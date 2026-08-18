"""Exhaustive enumeration of the bounded Autogram grammar."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Dict, Iterable, List, Sequence

from ..dsl import ast as A
from ..dsl.evaluate import (
    typed_binary_domain,
    typed_group_key,
    typed_sort_key,
    typed_unique,
)
from ..dsl.grammar import Grammar
from ..dsl.typecheck import is_admissible
from ..logic.solver import is_trivial, legacy_is_trivial


_SYMMETRIC_OPS = {"~=", "==", "!=", "<|>"}


class SearchSpaceTruncatedError(RuntimeError):
    """Raised when an explicit rule ceiling would make enumeration incomplete."""


# Trusted ceiling on the number of condition candidates the grammar may materialise. Condition
# columns are categorical/Boolean with small domains; a candidate space beyond this is a declaration
# error (a continuous column mislabelled as a condition), so fail loud rather than exhaust memory.
_MAX_CONDITION_CANDIDATES = 1_000_000


def _term_key(t: A.Term) -> str:
    return t.unparse()


def _term_sort_key(t: A.Term) -> tuple:
    rank = {A.Ref: 0, A.Agg: 1, A.RelatedAgg: 1, A.Scale: 2, A.Add: 3, A.Const: 4}
    return (rank.get(type(t), 9), t.unparse())


def _is_zero(t: A.Term) -> bool:
    return isinstance(t, A.Const) and float(t.value) == 0.0


def normalize_term(t: A.Term) -> A.Term:
    if isinstance(t, A.Rolling):
        inner = normalize_term(t.term)
        if int(t.window) == 1:
            return inner
        return A.Rolling(inner, int(t.window), t.kind)
    if isinstance(t, A.Lag):
        return A.Lag(normalize_term(t.term), int(t.steps))
    if isinstance(t, A.Diff):
        return A.Diff(normalize_term(t.term), int(t.steps))
    if isinstance(t, A.Scale):
        inner = normalize_term(t.term)
        if isinstance(inner, A.Const):
            return A.Const(t.coeff * inner.value)
        if float(t.coeff) == 1.0:
            return inner
        return A.Scale(float(t.coeff), inner)
    if isinstance(t, A.Add):
        parts: list[A.Term] = []
        const = 0.0
        for x in t.terms:
            nx = normalize_term(x)
            if isinstance(nx, A.Add):
                parts.extend(nx.terms)
            elif isinstance(nx, A.Const):
                const += nx.value
            else:
                parts.append(nx)
        if abs(const) > 0.0:
            parts.append(A.Const(const))
        parts = sorted(parts, key=_term_sort_key)
        if not parts:
            return A.Const(0.0)
        if len(parts) == 1:
            return parts[0]
        return A.Add(tuple(parts))
    return t


def normalize_rule(rule: A.Rule) -> A.Rule:
    if not isinstance(rule.atom, A.Compare):
        return A.Rule(
            rule.binder,
            rule.atom,
            tag=rule.tag,
            condition=_normalize_condition(rule.condition),
        )
    left = normalize_term(rule.atom.left)
    right = normalize_term(rule.atom.right)
    op = rule.atom.op
    if op in _SYMMETRIC_OPS and _term_key(right) < _term_key(left):
        left, right = right, left
    elif op == "<=" and _is_zero(left) and not _is_zero(right):
        left, right, op = right, left, ">="
    elif op == ">=" and _is_zero(left) and not _is_zero(right):
        left, right, op = right, left, "<="
    return A.Rule(
        rule.binder,
        A.Compare(left, op, right),
        tag=rule.tag,
        condition=_normalize_condition(rule.condition),
    )


def _normalize_condition(condition):
    if condition is None:
        return condition
    if condition.op in ("in", "not in"):
        return A.Condition(
            condition.column,
            condition.op,
            tuple(sorted(
                condition.values,
                key=lambda value: typed_sort_key(typed_group_key(value)),
            )),
        )
    if condition.op != "all":
        return condition
    children = tuple(sorted(
        (
            _normalize_condition(child)
            for child in condition.values
            if isinstance(child, A.Condition)
        ),
        key=lambda child: child.unparse(),
    ))
    return A.Condition("", "all", children)


def _term_ref_roles(term: A.Term) -> set[str]:
    if isinstance(term, A.Ref):
        return {term.role}
    if isinstance(term, (A.Scale, A.Lag, A.Diff, A.Rolling)):
        return _term_ref_roles(term.term)
    if isinstance(term, A.Add):
        return set().union(*(
            _term_ref_roles(child)
            for child in term.terms
        ))
    if isinstance(term, (A.Mul, A.Div)):
        left = term.left if isinstance(term, A.Mul) else term.num
        right = term.right if isinstance(term, A.Mul) else term.den
        return _term_ref_roles(left) | _term_ref_roles(right)
    return set()


def _condition_columns(condition: A.Condition) -> set[object]:
    if condition.op == "all":
        return set().union(*(
            _condition_columns(child)
            for child in condition.values
            if isinstance(child, A.Condition)
        ))
    return {condition.column}


def _typed_column_roles(
    column_roles: Mapping[object, str] | Iterable[tuple] | None,
) -> dict[tuple[str | None, tuple], frozenset[str]] | None:
    if column_roles is None:
        return None
    items = (
        column_roles.items()
        if isinstance(column_roles, Mapping)
        else column_roles
    )
    out: dict[tuple, set[str]] = {}
    for item in items:
        if len(item) == 2:
            column, role = item
            binder = None
        elif len(item) == 3:
            binder, column, role = item
            binder = str(binder)
        else:
            raise ValueError(
                "column role mappings require (column, role) or "
                "(binder, column, role) entries"
            )
        key = (binder, typed_group_key(column))
        out.setdefault(key, set()).add(str(role))
    return {
        key: frozenset(roles)
        for key, roles in out.items()
    }


def _self_conditioned(
    rule: A.Rule,
    column_roles: Mapping[
        tuple[str | None, tuple],
        frozenset[str],
    ] | None,
) -> bool:
    if rule.condition is None:
        return False
    if isinstance(rule.atom, A.Compare):
        roles = (
            _term_ref_roles(rule.atom.left)
            | _term_ref_roles(rule.atom.right)
        )
    elif isinstance(rule.atom, A.BandDefinition):
        roles = _term_ref_roles(rule.atom.term)
    else:
        return False
    condition_roles = set()
    for column in _condition_columns(rule.condition):
        mapped = set()
        if column_roles is not None:
            column_key = typed_group_key(column)
            mapped.update(
                column_roles.get((rule.binder, column_key), ())
            )
            mapped.update(
                column_roles.get((None, column_key), ())
            )
        if mapped:
            condition_roles.update(mapped)
        elif column_roles is None and isinstance(column, str):
            condition_roles.add(column)
    return bool(roles & condition_roles)


class EnumerationProposer:
    """Enumerate every admissible rule in a finite grammar bound.

    ``n`` is accepted for compatibility with the discovery loop but does not limit enumeration;
    use ``Grammar.max_complexity`` to bound the hypothesis space.

    ``column_roles`` is the authoritative profiled column-to-role mapping. Runtime entries are
    binder-scoped triples; mappings and two-item entries remain binder-agnostic for manual grammars.
    Iterables preserve typed column identities that compare equal in Python.
    """

    def __init__(
        self,
        G: Grammar,
        *,
        column_roles: (
            Mapping[object, str]
            | Iterable[tuple]
            | None
        ) = None,
    ):
        self.G = G
        if column_roles is None:
            grammar_column_roles = getattr(G, "column_roles", None)
            column_roles = grammar_column_roles or None
        self._column_roles = _typed_column_roles(
            column_roles
        )
        self._cache: List[A.Rule] | None = None

    def propose(self, n: int = 0, seeds: Sequence[A.Rule] = (), rng=None) -> List[A.Rule]:
        if self._cache is None:
            self._cache = self._enumerate()
        return list(self._cache)

    def _base_terms_for(self, binder: str) -> List[A.Term]:
        cap = self.G.complexity_cap(binder)
        terms: Dict[str, A.Term] = {}
        for role in self.G.refs_for(binder):
            t = A.Ref(role)
            if t.complexity() <= cap:
                terms[_term_key(t)] = t
        for fam in self.G.fams_for(binder):
            for kind in self.G.agg_kinds:          # proposer-chosen: SUM/AVG/MIN/MAX
                t = A.Agg(kind, fam)
                if t.complexity() <= cap:
                    terms[_term_key(t)] = t
        for role in self.G.related_for(binder):
            term = A.RelatedAgg(role)
            if term.complexity() <= cap:
                terms[_term_key(term)] = term
        return list(terms.values())

    def _scaled_terms_for(self, binder: str, base_terms: Sequence[A.Term]) -> List[A.Term]:
        cap = int(self.G.max_linear_leaves)
        if cap > 0 and len(base_terms) > cap:
            raise SearchSpaceTruncatedError(
                f"linear grammar for binder {binder!r} exposes "
                f"{len(base_terms)} leaves, exceeding "
                f"max_linear_leaves={cap}; raise the ceiling or tighten "
                "the declared grammar"
            )
        cap = self.G.complexity_cap(binder)
        terms: Dict[str, A.Term] = {}
        for t in base_terms:
            for coeff in self.G.scale_coeffs:
                st = normalize_term(A.Scale(float(coeff), t))
                if st.complexity() <= cap:
                    terms[_term_key(st)] = st
        return list(terms.values())

    def _add_terms_for(self, binder: str) -> List[A.Term]:
        comp_cap = self.G.complexity_cap(binder)
        arity_cap = self.G.add_arity_cap(binder)
        boolean_roles = set(self.G.booleans_for(binder))
        refs = [
            A.Ref(role)
            for role in self.G.refs_for(binder)
            if role not in boolean_roles
        ]
        aggs = [A.Agg(kind, fam)
                for fam in self.G.fams_for(binder)
                for kind in self.G.agg_kinds]        # proposer-chosen aggregations
        leaves = refs + aggs
        cap = int(self.G.max_linear_leaves)
        if cap > 0 and len(leaves) > cap:
            raise SearchSpaceTruncatedError(
                f"linear grammar for binder {binder!r} exposes "
                f"{len(leaves)} leaves, exceeding "
                f"max_linear_leaves={cap}; raise the ceiling or tighten "
                "the declared grammar"
            )
        terms: Dict[str, A.Term] = {}
        for arity in range(2, max(1, arity_cap) + 1):
            for combo in itertools.combinations(leaves, arity):
                at = normalize_term(A.Add(tuple(combo)))
                if at.complexity() <= comp_cap:
                    terms[_term_key(at)] = at
        return list(terms.values())

    @staticmethod
    def _is_scaled_slack(left: A.Term, op: str, right: A.Term) -> bool:
        if op not in ("<=", ">=") or not isinstance(left, A.Scale) or isinstance(right, A.Const):
            return False
        return left.coeff < 0.0 or abs(left.coeff) < 1.0

    def _candidate_rules(self):
        if self.G.legacy_compat:
            yield from self._legacy_candidate_rules()
            return
        for binder in self.G.binders:
            zero = A.Const(0.0)
            all_base_terms = self._base_terms_for(binder)
            boolean_roles = set(self.G.booleans_for(binder))
            boolean_related_roles = set(
                self.G.boolean_related_for(binder)
            )
            boolean_terms = [
                term
                for term in all_base_terms
                if (
                    isinstance(term, A.Ref)
                    and term.role in boolean_roles
                ) or (
                    isinstance(term, A.RelatedAgg)
                    and term.role in boolean_related_roles
                )
            ]
            base_terms = [
                term
                for term in all_base_terms
                if not ((
                    isinstance(term, A.Ref)
                    and term.role in boolean_roles
                ) or (
                    isinstance(term, A.RelatedAgg)
                    and term.role in boolean_related_roles
                ))
            ]
            scaled_terms = self._scaled_terms_for(binder, base_terms)
            add_terms = self._add_terms_for(binder)
            temporal_terms = (
                self._temporal_terms_for(binder, base_terms)
                if self.G.temporal_enabled else []
            )
            nonlinear_leaves = base_terms + [
                term for term in temporal_terms if isinstance(term, A.Rolling)
            ]
            nonlinear_leaves = self._bounded_nonlinear_leaves(
                binder,
                nonlinear_leaves,
            )
            nonlinear_terms = (self._nonlinear_terms_for(binder, nonlinear_leaves)
                               if self.G.degree_cap(binder) >= 2 else [])
            nonlinear_terms = sorted(
                nonlinear_terms,
                key=self._nonlinear_priority,
            )
            measured = base_terms + add_terms + temporal_terms + nonlinear_terms
            simple_terms = [zero] + base_terms
            # Put high-value nonlinear identities and definitions before broad linear combinations
            # so a configured max_rules budget cannot starve ratio/temporal targets.
            if self.G.band_enabled and (
                binder == "record" or "record" not in self.G.binders
            ):
                for ref in (
                    term for term in base_terms
                    if isinstance(term, A.Ref)
                    and term.role not in set(self.G.booleans_for(binder))
                ):
                    # Emitted bare: conditioning happens in one place (`_enumerate`), so every
                    # conditioned variant -- band definitions included -- is counted against
                    # `max_conditioned_rules` instead of slipping past it.
                    yield A.Rule(binder, A.BandDefinition(ref, None))
            if "~∝" in self.G.ops:
                refs = [
                    term for term in base_terms
                    if isinstance(term, A.Ref)
                ]
                for left in refs:
                    for right in refs:
                        if left != right:
                            yield A.Rule(
                                binder,
                                A.Compare(left, "~∝", right),
                            )
            one_sided_terms = [
                term for term in measured
                if isinstance(term, (A.Ref, A.Diff))
            ]
            for t in one_sided_terms:
                if isinstance(t, A.Diff):
                    operators = ["~=", "=="]
                    if ">" in self.G.ops:
                        operators.append(">")
                    if "<" in self.G.ops:
                        operators.append("<")
                    operators.extend([">=", "<="])
                else:
                    operators = [">=", "<="]
                    if ">" in self.G.ops:
                        operators.append(">")
                    if "<" in self.G.ops:
                        operators.append("<")
                for operator in operators:
                    yield A.Rule(binder, A.Compare(t, operator, zero))
            for left in nonlinear_terms:
                for right in simple_terms:
                    for op in ("~=", "=="):
                        yield A.Rule(binder, A.Compare(left, op, right))
            # Base-vs-base carries the full operator set including same-family separations.
            for i, left in enumerate(simple_terms):
                for j, right in enumerate(simple_terms):
                    if i == j:
                        continue
                    for op in self.G.ops:
                        if op in _SYMMETRIC_OPS and j <= i:
                            continue
                        yield A.Rule(binder, A.Compare(left, op, right))
            for left_index, left in enumerate(boolean_terms):
                for right in boolean_terms[left_index + 1:]:
                    for op in ("==", "!=", "<|>"):
                        if op in self.G.ops:
                            yield A.Rule(
                                binder,
                                A.Compare(left, op, right),
                            )
            if self.G.advanced_enabled:
                yield from self._definition_rules(
                    binder,
                    [
                        term
                        for term in all_base_terms
                        if isinstance(term, A.Ref)
                    ],
                )
            temporal_ops = ["~=", "==", "<=", ">="]
            if "<" in self.G.ops:
                temporal_ops.append("<")
            if ">" in self.G.ops:
                temporal_ops.append(">")
            for temporal in temporal_terms:
                for right in simple_terms:
                    for op in temporal_ops:
                        yield A.Rule(
                            binder,
                            A.Compare(temporal, op, right),
                        )

            # Bounded linear forms are compared to base terms; `!=` remains atomic only.
            for left in scaled_terms + add_terms:
                for right in simple_terms:
                    if isinstance(left, A.Scale) and isinstance(right, A.Const):
                        continue
                    for op in ("~=", "==", "<=", ">="):
                        if self._is_scaled_slack(left, op, right):
                            continue
                        yield A.Rule(binder, A.Compare(left, op, right))
            for i, left in enumerate(add_terms):
                for j, right in enumerate(add_terms):
                    if j <= i:
                        continue
                    for op in ("~=", "==", "<=", ">="):
                        yield A.Rule(binder, A.Compare(left, op, right))

    def _legacy_candidate_rules(self):
        for binder in self.G.binders:
            zero = A.Const(0.0)
            base_terms = sorted(
                self._base_terms_for(binder),
                key=_term_key,
            )
            scaled_terms = sorted(
                self._scaled_terms_for(binder, base_terms),
                key=_term_key,
            )
            add_terms = sorted(
                self._add_terms_for(binder),
                key=_term_key,
            )
            nonlinear_terms = (
                sorted(
                    self._nonlinear_terms_for(binder, base_terms),
                    key=_term_key,
                )
                if self.G.degree_cap(binder) >= 2
                else []
            )
            measured = base_terms + add_terms + nonlinear_terms
            simple_terms = [zero] + base_terms
            for term in measured:
                yield A.Rule(
                    binder,
                    A.Compare(term, ">=", zero),
                )
                yield A.Rule(
                    binder,
                    A.Compare(term, "<=", zero),
                )
            for i, left in enumerate(simple_terms):
                for j, right in enumerate(simple_terms):
                    if i == j:
                        continue
                    for op in self.G.ops:
                        if op in _SYMMETRIC_OPS and j <= i:
                            continue
                        yield A.Rule(
                            binder,
                            A.Compare(left, op, right),
                        )
            for left in scaled_terms + add_terms:
                for right in simple_terms:
                    if isinstance(left, A.Scale) and isinstance(
                        right,
                        A.Const,
                    ):
                        continue
                    for op in ("~=", "==", "<=", ">="):
                        if self._is_scaled_slack(left, op, right):
                            continue
                        yield A.Rule(
                            binder,
                            A.Compare(left, op, right),
                        )
            for i, left in enumerate(add_terms):
                for j, right in enumerate(add_terms):
                    if j <= i:
                        continue
                    for op in ("~=", "==", "<=", ">="):
                        yield A.Rule(
                            binder,
                            A.Compare(left, op, right),
                        )
            for left in nonlinear_terms:
                for right in simple_terms:
                    for op in ("~=", "==", "<=", ">="):
                        yield A.Rule(
                            binder,
                            A.Compare(left, op, right),
                        )

    def _nonlinear_terms_for(self, binder: str, base_terms: Sequence[A.Term]) -> List[A.Term]:
        """Products a*b (a!=b) and ratios a/b, gated by the binder's degree cap (item 6)."""
        deg_cap = self.G.degree_cap(binder)
        comp_cap = self.G.complexity_cap(binder)
        terms: Dict[str, A.Term] = {}
        leaves = list(base_terms)
        for i, a in enumerate(leaves):
            for j, b in enumerate(leaves):
                if i == j:
                    continue
                if i < j:
                    m = A.Mul(a, b)
                    if (m.degree() <= deg_cap
                            and m.complexity() <= comp_cap):
                        terms[_term_key(m)] = m
                d = A.Div(a, b)
                if (d.degree() <= deg_cap
                        and d.complexity() <= comp_cap):
                    terms[_term_key(d)] = d
        return list(terms.values())

    def _bounded_nonlinear_leaves(
        self,
        binder: str,
        terms: Sequence[A.Term],
    ) -> List[A.Term]:
        cap = int(self.G.max_nonlinear_leaves)
        if cap <= 0 or len(terms) <= cap:
            return list(terms)
        raise SearchSpaceTruncatedError(
            f"nonlinear grammar for binder {binder!r} exposes "
            f"{len(terms)} leaves, exceeding "
            f"max_nonlinear_leaves={cap}; raise the ceiling or tighten "
            "the declared grammar"
        )

    @staticmethod
    def _nonlinear_priority(term: A.Term) -> tuple:
        ratio = isinstance(term, A.Div)
        matching_window_ratio = (
            isinstance(term, A.Div)
            and isinstance(term.num, A.Rolling)
            and isinstance(term.den, A.Rolling)
            and term.num.window == term.den.window
        )
        direct_ref_ratio = (
            isinstance(term, A.Div)
            and isinstance(term.num, A.Ref)
            and isinstance(term.den, A.Ref)
        )
        return (
            0 if matching_window_ratio else
            1 if direct_ref_ratio else
            2 if ratio else
            3,
        )

    def _temporal_terms_for(
        self,
        binder: str,
        base_terms: Sequence[A.Term],
    ) -> List[A.Term]:
        comp_cap = self.G.complexity_cap(binder)
        terms: Dict[str, A.Term] = {}
        refs = [term for term in base_terms if isinstance(term, A.Ref)]
        lags = range(1, int(self.G.max_lag) + 1)
        for ref in refs:
            for steps in lags:
                for term in (A.Lag(ref, steps), A.Diff(ref, steps)):
                    if term.complexity() <= comp_cap:
                        terms[_term_key(term)] = term
            for window in self.G.windows:
                term = A.Rolling(ref, int(window), "SUM")
                if term.complexity() <= comp_cap:
                    terms[_term_key(term)] = term
        return list(terms.values())

    def _enumerate(self) -> List[A.Rule]:
        out: List[A.Rule] = []
        seen = set()
        conditioned_emitted = 0
        conditions = (
            self._conditions()
            if self.G.conditional_enabled
            else []
        )
        precount = bool(
            conditions
            and self.G.conditional_enabled
            and self.G.max_conditioned_rules > 0
        )
        # Pre-count the conditioned expansion before building any of it. ``conditioned_emitted``
        # below counts conditioned VARIANTS -- before dedup, admissibility and triviality screening
        # -- so `conditionable x |conditions|` is not an estimate but exactly the number the ceiling
        # is compared against. Discovering that deep inside expansion means paying for the whole
        # blow-up first; refusing here names both factors, so a mis-declared condition column is
        # immediately diagnosable. The count STREAMS the base rules and raises the moment the
        # running product crosses the ceiling, so a blow-up is refused without ever materialising
        # the candidate list -- materialising it would trade one memory blow-up for another and
        # defeat the purpose.
        if precount:
            conditionable = 0
            for raw in self._candidate_rules():
                if not (
                    isinstance(raw.atom, (A.Compare, A.BandDefinition))
                    and self._conditional_candidate(raw)
                ):
                    continue
                conditionable += 1
                if conditionable * len(conditions) > self.G.max_conditioned_rules:
                    raise SearchSpaceTruncatedError(
                        f"conditioned grammar would expand at least {conditionable} conditionable "
                        f"rules over {len(conditions)} conditions = "
                        f"{conditionable * len(conditions)} conditioned candidates, exceeding "
                        f"max_conditioned_rules={self.G.max_conditioned_rules}; raise the ceiling "
                        "or tighten explicit condition bounds"
                    )
        for raw in self._candidate_rules():
            variants = [raw]
            if (
                isinstance(raw.atom, (A.Compare, A.BandDefinition))
                and self.G.conditional_enabled
                and self._conditional_candidate(raw)
            ):
                for condition in conditions:
                    if (
                        self.G.max_conditioned_rules > 0
                        and conditioned_emitted >= self.G.max_conditioned_rules
                    ):
                        raise SearchSpaceTruncatedError(
                            "conditioned grammar exceeds "
                            f"max_conditioned_rules={self.G.max_conditioned_rules}; "
                            "raise the ceiling or tighten explicit condition bounds"
                        )
                    variants.append(
                        A.Rule(
                            raw.binder,
                            raw.atom,
                            tag=raw.tag,
                            condition=condition,
                        )
                    )
                    conditioned_emitted += 1
            for candidate in variants:
                rule = normalize_rule(candidate)
                if _self_conditioned(rule, self._column_roles):
                    continue
                if rule.complexity() > self.G.complexity_cap(rule.binder):
                    continue
                ok, _ = is_admissible(rule, self.G)
                if not ok:
                    continue
                if (
                    legacy_is_trivial(rule)
                    if self.G.legacy_compat
                    else is_trivial(rule)
                ):
                    continue
                sig = rule.signature()
                if sig in seen:
                    continue
                seen.add(sig)
                if self.G.max_rules and len(out) >= self.G.max_rules:
                    raise SearchSpaceTruncatedError(
                        "bounded grammar exceeds "
                        f"max_rules={self.G.max_rules}; raise the ceiling or "
                        "tighten explicit grammar complexity bounds"
                    )
                out.append(rule)
        return out

    def _definition_rules(self, binder: str, refs: Sequence[A.Ref]):
        boolean_roles = set(self.G.booleans_for(binder))
        targets = [ref for ref in refs if ref.role in boolean_roles]
        numeric = [ref for ref in refs if ref.role not in boolean_roles]
        for target in targets:
            for ref in numeric:
                for op in ("<", "<=", ">", ">="):
                    for threshold in (None, 0.0):
                        yield A.Rule(
                            binder,
                            A.BooleanDefinition(
                                target,
                                A.Bound(ref, op, threshold),
                            ),
                        )
                for window in self.G.run_lengths:
                    for op in ("<", "<=", ">", ">="):
                        yield A.Rule(
                            binder,
                            A.BooleanDefinition(
                                target,
                                A.Sustained(
                                    A.Bound(
                                        ref,
                                        op,
                                        None,
                                    ),
                                    int(window),
                                ),
                            ),
                        )
            # Generic bounded conjunctions over the numeric columns, each conjunct a fixed-0 sign
            # bound. Learned-threshold conjunctions are admitted by the grammar and fitted exactly by
            # the evaluator (see ``_definition_predicate_enumerable`` and ``_evaluate_boolean_definition``),
            # but the default bounded proposer does not exhaustively emit them over every column
            # combination: grounding-and-fitting a learned-threshold conjunction per column tuple is
            # prohibitively expensive at realistic profile scale (the joint two-threshold fit is
            # O(|candidates|^2) per rule). Single learned thresholds enter through the single-bound and
            # sustained definitions above and the bounded compound tier below; broader learned-
            # threshold conjunction search is a RegimeSpec/config trade-off, not the default budget.
            max_arity = min(
                int(self.G.max_conjunction_terms),
                len(numeric),
            )
            for arity in range(2, max_arity + 1):
                for selected in itertools.combinations(numeric, arity):
                    choices = [
                        tuple(
                            A.Bound(ref, op, 0.0)
                            for op in ("<", "<=", ">", ">=")
                        )
                        for ref in selected
                    ]
                    for predicates in itertools.product(*choices):
                        yield A.Rule(
                            binder,
                            A.BooleanDefinition(
                                target,
                                A.Conjunction(tuple(predicates)),
                            ),
                        )
            # Bounded compound-operand conjunction tier. The generic conjunctions above range only
            # over bare columns; a fully generic enumeration that also ranged compound temporal
            # operands (rolling sums, finite differences, pairwise-difference rolling sums) over every
            # column pair, op, and window would provably blow past ``max_rules`` for realistic
            # profiles. This tier supplies those compound operands under the SAME generic admissibility
            # as the bare-column conjunctions (see ``_definition_predicate_enumerable``), parameterised
            # over all columns/windows -- no conjunct is tied to a specific named invariant.
            if self.G.max_conjunction_terms >= 3 and len(numeric) >= 2:
                for ratio in numeric:
                    for input_ref, output_ref in itertools.permutations(
                        numeric,
                        2,
                    ):
                        for window in self.G.windows:
                            if int(window) > self.G.max_lag:
                                continue
                            yield A.Rule(
                                binder,
                                A.BooleanDefinition(
                                    target,
                                    A.Conjunction((
                                        A.Bound(ratio, "<", None),
                                        A.Bound(
                                            A.Rolling(
                                                A.Add((
                                                    input_ref,
                                                    A.Scale(-1.0, output_ref),
                                                )),
                                                int(window),
                                                "SUM",
                                            ),
                                            ">",
                                            0.0,
                                        ),
                                        A.Bound(
                                            A.Diff(ratio, int(window)),
                                            "<=",
                                            0.0,
                                        ),
                                    )),
                                ),
                            )

        if binder != "record" and "record" in self.G.binders:
            return

        conditions = self.G.condition_columns
        bool_columns = [
            column
            for column, values in conditions.items()
            if typed_binary_domain(values)
        ]
        category_columns = [
            column
            for column, values in conditions.items()
            if values and not typed_binary_domain(values)
        ]
        for target_column in category_columns:
            target_values = typed_unique(conditions[target_column])
            for default in target_values:
                default_key = typed_group_key(default)
                labels = tuple(
                    value
                    for value in target_values
                    if typed_group_key(value) != default_key
                )
                if not labels or len(labels) > len(bool_columns):
                    continue
                for columns in itertools.permutations(bool_columns, len(labels)):
                    for emitted in itertools.permutations(labels):
                        yield A.Rule(
                            binder,
                            A.CategoryDefinition(
                                target_column,
                                tuple(zip(columns, emitted)),
                                default,
                            ),
                        )

    def _conditions(self) -> List[A.Condition]:
        out: List[A.Condition] = []
        cap = max(1, int(self.G.max_condition_values))
        ranked = sorted(
            (
                (column, typed_unique(values))
                for column, values in self.G.condition_columns.items()
            ),
            key=lambda item: item[0],
        )
        # Fail loud *before* materialising a combinatorial blow-up: the subset ("in") enumeration is
        # sum_{s=2..min(cap, v-1)} C(v, s) per column, which for a wide domain and a high value cap
        # would eagerly allocate trillions of conditions before any rule ceiling could fire. Count
        # the candidates first and refuse an unbounded grammar rather than exhaust memory.
        import math as _math

        categorical_columns = {
            column
            for column, values in self.G.condition_columns.items()
            if values and not typed_binary_domain(values)
        }
        total = 0
        simple_per_column = []
        for column, raw_values in ranked:
            v = len(raw_values)
            if v <= 1:
                continue
            total += v
            if column in categorical_columns:
                simple_per_column.append(v)
            for subset_size in range(2, min(cap, v - 1) + 1):
                total += _math.comb(v, subset_size)
        # The cross-column conjunctions built below are conditions too, and they are quadratic in
        # the number of simple categorical equalities. Counting only the per-column subsets let
        # millions of conditions materialise before the ceiling could fire. Only pairs from
        # DIFFERENT columns are generated, so same-column pairs must not be counted or the ceiling
        # would refuse a grammar it could actually enumerate.
        running = 0
        for v in simple_per_column:
            total += running * v
            running += v
        ceiling = _MAX_CONDITION_CANDIDATES
        if total > ceiling:
            raise SearchSpaceTruncatedError(
                f"condition grammar would enumerate {total} candidates, exceeding the trusted "
                f"ceiling {ceiling}; tighten the declared condition domains or value cap"
            )
        for column, raw_values in ranked:
            values = typed_unique(raw_values)
            if len(values) <= 1:
                continue
            for value in values:
                out.append(A.Condition(column, "==", (value,)))
            max_subset = min(cap, len(values) - 1)
            for subset_size in range(2, max_subset + 1):
                for subset in itertools.combinations(
                    values,
                    subset_size,
                ):
                    out.append(A.Condition(
                        column,
                        "in",
                        tuple(sorted(
                            subset,
                            key=lambda value: typed_sort_key(
                                typed_group_key(value)
                            ),
                        )),
                    ))
        simple = [
            condition for condition in out
            if condition.op == "==" and condition.values
            and condition.column in categorical_columns
        ]
        for left, right in itertools.combinations(simple, 2):
            if left.column != right.column:
                out.append(A.Condition("", "all", (left, right)))
        return out

    def _conditional_candidate(self, rule: A.Rule) -> bool:
        atom = rule.atom
        # A band definition is conditionable: `x ~band c where regime == A` is a regime-restricted
        # concentration claim, and it must pass through the same ceiling accounting as the rest.
        if isinstance(atom, A.BandDefinition):
            return bool(self.G.band_enabled)
        if not isinstance(atom, A.Compare):
            return False
        # Proportional laws are conditionable.
        if atom.op == "\u007e\u221d":
            return True
        # A generic near-equality / equality between two DISTINCT atomic measurements is a
        # conditionable relation (e.g. a regime-conditioned balance ``x ~= y where regime == A``).
        # This is not tied to any named invariant -- any measured ref pair qualifies.
        if (
            atom.op in ("~=", "==")
            and isinstance(atom.left, A.Ref)
            and isinstance(atom.right, A.Ref)
            and atom.left != atom.right
        ):
            return True
        # A finite-difference term compared to zero under ANY operator: conditional monotonicity /
        # positivity (``DELTA_k(x) > 0``) and conditional zero-change (``DELTA_k(x) ~= 0``).
        return (
            (isinstance(atom.left, A.Diff) and _is_zero(atom.right))
            or (isinstance(atom.right, A.Diff) and _is_zero(atom.left))
        )


# Backward-compatible name for callers that still ask for a random proposer; it now enumerates.
RandomProposer = EnumerationProposer


def _term_has_temporal(*terms: A.Term) -> bool:
    def visit(term: A.Term) -> bool:
        if isinstance(term, (A.Lag, A.Diff, A.Rolling)):
            return True
        if isinstance(term, A.Scale):
            return visit(term.term)
        if isinstance(term, A.Add):
            return any(visit(child) for child in term.terms)
        if isinstance(term, A.Mul):
            return visit(term.left) or visit(term.right)
        if isinstance(term, A.Div):
            return visit(term.num) or visit(term.den)
        return False

    return any(visit(term) for term in terms)