"""Exhaustive enumeration of the bounded Autogram grammar."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..dsl import ast as A
from ..dsl.evaluate import (
    typed_binary_domain,
    typed_condition_key,
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
_MAX_CATEGORY_CANDIDATES = 1_000_000


@dataclass(frozen=True)
class ExecutableConditionGrammar:
    """Condition forms the bounded proposer can attach exhaustively."""

    simple_ops: tuple[str, ...] = ("==", "in")
    conjunction_arities: tuple[int, ...] = (2,)
    conjunction_child_ops: tuple[str, ...] = ("==",)
    conjunction_allows_binary_domains: bool = False
    # Definition and broad algebraic families stay unconditioned to preserve the bounded null
    # surface; known catalogs use this same allow-list and fail before induction for every omission.
    conditioned_relation_kinds: frozenset[str] = frozenset({
        "healthy_band",
        "pair",
        "proportional",
        "delta_bound",
        "delta_zero",
    })


EXECUTABLE_CONDITION_GRAMMAR = ExecutableConditionGrammar()


def condition_family_is_enumerable(kind: str | None) -> bool:
    """Whether the proposer attaches conditions to this relation family."""
    return kind in EXECUTABLE_CONDITION_GRAMMAR.conditioned_relation_kinds


def condition_conjunction_arity_error() -> str:
    arities = EXECUTABLE_CONDITION_GRAMMAR.conjunction_arities
    if len(arities) == 1:
        return (
            "condition conjunction must contain exactly "
            f"{arities[0]} child conditions"
        )
    return (
        "condition conjunction arity must be one of "
        f"{arities!r}"
    )


def canonical_executable_condition(
    condition: A.Condition,
    *,
    condition_columns: Mapping[object, Sequence[object]] | None = None,
    max_condition_values: int | None = None,
    _inside_conjunction: bool = False,
) -> A.Condition:
    """Validate and canonicalize one condition in the executable grammar."""
    policy = EXECUTABLE_CONDITION_GRAMMAR
    if not isinstance(condition, A.Condition):
        raise ValueError("condition conjunction must contain condition children")
    if condition.op == "all":
        if _inside_conjunction:
            raise ValueError("condition conjunctions may not nest")
        if len(condition.values) not in policy.conjunction_arities:
            raise ValueError(condition_conjunction_arity_error())
        children = tuple(
            canonical_executable_condition(
                child,
                condition_columns=condition_columns,
                max_condition_values=max_condition_values,
                _inside_conjunction=True,
            )
            for child in condition.values
        )
        if any(
            child.op not in policy.conjunction_child_ops
            for child in children
        ):
            raise ValueError(
                "condition conjunction children must use equality conditions"
            )
        columns = tuple(
            typed_group_key(child.column)
            for child in children
        )
        if len(columns) != len(set(columns)):
            raise ValueError(
                "condition conjunction must use distinct columns"
            )
        return A.Condition(
            "",
            "all",
            tuple(sorted(
                children,
                key=lambda child: repr(typed_condition_key(child)),
            )),
        )
    if condition.op not in policy.simple_ops:
        raise ValueError(
            f"condition operator {condition.op!r} is not executable"
        )
    if (
        _inside_conjunction
        and condition.op not in policy.conjunction_child_ops
    ):
        raise ValueError(
            "condition conjunction children must use equality conditions"
        )
    if condition.op == "==" and len(condition.values) != 1:
        raise ValueError(
            "equality condition requires exactly one value"
        )
    values = typed_unique(condition.values)
    if not values:
        raise ValueError("condition must contain at least one value")
    op = condition.op
    if op == "in":
        if len(values) == 1:
            op = "=="
        elif (
            max_condition_values is not None
            and len(values) > max(1, int(max_condition_values))
        ):
            raise ValueError("condition exceeds the value cap")
    if op == "in":
        values = tuple(sorted(
            values,
            key=lambda value: typed_sort_key(
                typed_group_key(value)
            ),
        ))
    if condition_columns is not None:
        if condition.column not in condition_columns:
            raise ValueError(
                f"condition column {condition.column!r} is not enabled"
            )
        if (
            _inside_conjunction
            and not policy.conjunction_allows_binary_domains
            and typed_binary_domain(condition_columns[condition.column])
        ):
            raise ValueError(
                "condition conjunction children require non-binary "
                "categorical columns"
            )
        allowed_values = {
            typed_group_key(value)
            for value in typed_unique(
                condition_columns[condition.column]
            )
        }
        condition_values = {
            typed_group_key(value)
            for value in values
        }
        if not condition_values <= allowed_values:
            raise ValueError(
                "condition value is not observed for the declared column"
            )
        if op == "in" and len(condition_values) >= len(allowed_values):
            raise ValueError(
                "membership condition requires a proper multi-value subset"
            )
    elif (
        _inside_conjunction
        and not policy.conjunction_allows_binary_domains
        and typed_binary_domain(values)
    ):
        raise ValueError(
            "condition conjunction children require non-binary "
            "categorical columns"
        )
    return A.Condition(condition.column, op, tuple(values))


def enumerate_executable_conditions(
    condition_columns: Mapping[object, Sequence[object]],
    max_condition_values: int,
) -> List[A.Condition]:
    """Enumerate exactly the declarative executable condition grammar."""
    policy = EXECUTABLE_CONDITION_GRAMMAR
    cap = max(1, int(max_condition_values))
    ranked = sorted(
        (
            (column, typed_unique(values))
            for column, values in condition_columns.items()
        ),
        key=lambda item: typed_sort_key(typed_group_key(item[0])),
    )

    simple_counts: list[int] = []
    equality_counts: list[int] = []
    for _column, values in ranked:
        value_count = len(values)
        if value_count <= 1:
            simple_counts.append(0)
            equality_counts.append(0)
            continue
        subset_count = sum(
            math.comb(value_count, subset_size)
            for subset_size in range(
                2,
                min(cap, value_count - 1) + 1,
            )
        )
        simple_counts.append(value_count + subset_count)
        equality_counts.append(
            value_count
            if (
                policy.conjunction_allows_binary_domains
                or not typed_binary_domain(values)
            )
            else 0
        )

    total = sum(simple_counts)
    max_arity = max(policy.conjunction_arities, default=0)
    elementary = [0] * (max_arity + 1)
    elementary[0] = 1
    for count in equality_counts:
        for arity in range(max_arity, 0, -1):
            elementary[arity] += elementary[arity - 1] * count
    total += sum(
        elementary[arity]
        for arity in policy.conjunction_arities
    )
    if total > _MAX_CONDITION_CANDIDATES:
        raise SearchSpaceTruncatedError(
            f"condition grammar would enumerate {total} candidates, "
            f"exceeding the trusted ceiling {_MAX_CONDITION_CANDIDATES}; "
            "tighten the declared condition domains or value cap"
        )

    out: List[A.Condition] = []
    equalities_by_column: list[tuple[object, tuple[A.Condition, ...]]] = []
    for column, values in ranked:
        if len(values) <= 1:
            continue
        equalities = tuple(
            canonical_executable_condition(
                A.Condition(column, "==", (value,)),
                max_condition_values=cap,
            )
            for value in values
        )
        out.extend(equalities)
        if (
            policy.conjunction_allows_binary_domains
            or not typed_binary_domain(values)
        ):
            equalities_by_column.append((column, equalities))
        for subset_size in range(
            2,
            min(cap, len(values) - 1) + 1,
        ):
            for subset in itertools.combinations(
                values,
                subset_size,
            ):
                out.append(canonical_executable_condition(
                    A.Condition(column, "in", subset),
                    max_condition_values=cap,
                ))

    for arity in policy.conjunction_arities:
        for selected in itertools.combinations(
            equalities_by_column,
            arity,
        ):
            canonical_executable_condition(
                A.Condition(
                    "",
                    "all",
                    tuple(
                        conditions[0]
                        for _column, conditions in selected
                    ),
                ),
                condition_columns=condition_columns,
                max_condition_values=cap,
            )
            for children in itertools.product(*(
                conditions
                for _column, conditions in selected
            )):
                out.append(A.Condition(
                    "",
                    "all",
                    tuple(children),
                ))
    return out


def _onto_assignment_count(item_count: int, block_count: int) -> int:
    """Number of assignments onto ``block_count`` ordered non-empty blocks."""
    if block_count <= 0 or item_count < block_count:
        return 0
    return sum(
        (-1) ** (block_count - used)
        * math.comb(block_count, used)
        * used ** item_count
        for used in range(block_count + 1)
    )


def _label_block_assignment_count(
    label_count: int,
    block_count: int,
) -> int:
    """Adjacent-distinct block labels that use every available output label."""
    if label_count <= 0 or block_count < label_count:
        return 0
    return sum(
        (-1) ** (label_count - used)
        * math.comb(label_count, used)
        * (
            0
            if used == 0
            else used * (used - 1) ** (block_count - 1)
        )
        for used in range(label_count + 1)
    )


def _category_semantic_candidate_count(
    column_count: int,
    label_count: int,
    max_arity: int,
) -> int:
    """Count canonical priority maps for one target/default pair."""
    return sum(
        math.comb(column_count, arity)
        * _onto_assignment_count(arity, block_count)
        * _label_block_assignment_count(label_count, block_count)
        for arity in range(label_count, max_arity + 1)
        for block_count in range(label_count, arity + 1)
    )


def _label_block_assignments(labels: Sequence[object], block_count: int):
    """Yield adjacent-distinct block-label sequences that cover ``labels``."""
    label_count = len(labels)
    required = (1 << label_count) - 1

    def visit(prefix: tuple[int, ...], used: int):
        remaining = block_count - len(prefix)
        if (required & ~used).bit_count() > remaining:
            return
        if not remaining:
            if used == required:
                yield tuple(labels[index] for index in prefix)
            return
        previous = prefix[-1] if prefix else None
        for index in range(label_count):
            if previous is not None and index == previous:
                continue
            yield from visit(
                prefix + (index,),
                used | (1 << index),
            )

    yield from visit((), 0)


def _ordered_nonempty_blocks(
    columns: Sequence[str],
    block_count: int,
):
    """Yield each ordered partition once, with columns sorted inside every block."""
    columns = tuple(columns)
    if block_count == 1:
        yield (columns,)
        return
    max_first_size = len(columns) - block_count + 1
    positions = tuple(range(len(columns)))
    for first_size in range(1, max_first_size + 1):
        for selected in itertools.combinations(positions, first_size):
            selected_set = set(selected)
            first = tuple(columns[index] for index in selected)
            remaining = tuple(
                column
                for index, column in enumerate(columns)
                if index not in selected_set
            )
            for rest in _ordered_nonempty_blocks(
                remaining,
                block_count - 1,
            ):
                yield (first, *rest)


def _term_key(t: A.Term) -> tuple:
    return A.term_identity(t)


def _legacy_term_order_key(t: A.Term) -> tuple:
    """Legacy presentation order with an exact structural tie-break."""
    return (t.unparse(), _term_key(t))


def _term_sort_key(t: A.Term) -> tuple:
    rank = {A.Ref: 0, A.Agg: 1, A.RelatedAgg: 1, A.Scale: 2, A.Add: 3, A.Const: 4}
    return (rank.get(type(t), 9), _legacy_term_order_key(t))


def _rule_semantic_key(rule: A.Rule):
    if isinstance(rule.atom, A.CategoryDefinition):
        return (
            "rule",
            rule.binder,
            A.category_definition_semantic_key(
                rule.atom,
                typed_group_key,
            ),
            typed_condition_key(rule.condition),
        )
    return rule.signature()


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
    if (
        op in _SYMMETRIC_OPS
        and _legacy_term_order_key(right)
        < _legacy_term_order_key(left)
    ):
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
        terms: Dict[tuple, A.Term] = {}
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
        terms: Dict[tuple, A.Term] = {}
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
        terms: Dict[tuple, A.Term] = {}
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
                key=_legacy_term_order_key,
            )
            scaled_terms = sorted(
                self._scaled_terms_for(binder, base_terms),
                key=_legacy_term_order_key,
            )
            add_terms = sorted(
                self._add_terms_for(binder),
                key=_legacy_term_order_key,
            )
            nonlinear_terms = (
                sorted(
                    self._nonlinear_terms_for(binder, base_terms),
                    key=_legacy_term_order_key,
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
        terms: Dict[tuple, A.Term] = {}
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
        terms: Dict[tuple, A.Term] = {}
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
                sig = _rule_semantic_key(rule)
                if sig in seen:
                    continue
                seen.add(sig)
                if (
                    not isinstance(rule.atom, A.CategoryDefinition)
                    and (
                        legacy_is_trivial(rule)
                        if self.G.legacy_compat
                        else is_trivial(rule)
                    )
                ):
                    continue
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

        target_columns = self.G.condition_columns
        bool_columns = sorted(set(self.G.category_cases_for(binder)))
        category_columns = [
            column
            for column, values in target_columns.items()
            if values and not typed_binary_domain(values)
        ]
        category_specs = []
        semantic_total = 0
        for target_column in category_columns:
            target_values = typed_unique(target_columns[target_column])
            for default in target_values:
                default_key = typed_group_key(default)
                labels = tuple(
                    value
                    for value in target_values
                    if typed_group_key(value) != default_key
                )
                max_arity = min(
                    len(bool_columns),
                    int(self.G.max_conjunction_terms),
                    max(0, self.G.complexity_cap(binder) - 3),
                )
                if not labels or len(labels) > max_arity:
                    continue
                semantic_count = _category_semantic_candidate_count(
                    len(bool_columns),
                    len(labels),
                    max_arity,
                )
                semantic_total += semantic_count
                category_specs.append((
                    target_column,
                    default,
                    labels,
                    max_arity,
                ))
        if (
            self.G.max_rules > 0
            and semantic_total > self.G.max_rules
        ):
            raise SearchSpaceTruncatedError(
                "categorical grammar would enumerate "
                f"{semantic_total} semantic candidates, exceeding "
                f"max_rules={self.G.max_rules}; raise the ceiling or tighten "
                "the categorical arity/domain bounds"
            )
        if semantic_total > _MAX_CATEGORY_CANDIDATES:
            raise SearchSpaceTruncatedError(
                "categorical grammar would enumerate "
                f"{semantic_total} semantic candidates, exceeding the trusted "
                f"ceiling {_MAX_CATEGORY_CANDIDATES}; tighten the categorical "
                "arity/domain bounds"
            )
        for target_column, default, labels, max_arity in category_specs:
            label_count = len(labels)
            assignments_by_block_count = {
                block_count: tuple(_label_block_assignments(
                    labels,
                    block_count,
                ))
                for block_count in range(
                    label_count,
                    max_arity + 1,
                )
            }
            for arity in range(label_count, max_arity + 1):
                for columns in itertools.combinations(
                    bool_columns,
                    arity,
                ):
                    for block_count in range(
                        label_count,
                        arity + 1,
                    ):
                        assignments = assignments_by_block_count[
                            block_count
                        ]
                        if not assignments:
                            continue
                        for blocks in _ordered_nonempty_blocks(
                            columns,
                            block_count,
                        ):
                            for emitted in assignments:
                                cases = tuple(
                                    (column, value)
                                    for block, value in zip(
                                        blocks,
                                        emitted,
                                    )
                                    for column in block
                                )
                                cases = A.category_definition_canonical_cases(
                                    cases,
                                    typed_group_key,
                                )
                                yield A.Rule(
                                    binder,
                                    A.CategoryDefinition(
                                        target_column,
                                        cases,
                                        default,
                                    ),
                                )

    def _conditions(self) -> List[A.Condition]:
        return enumerate_executable_conditions(
            self.G.condition_columns,
            self.G.max_condition_values,
        )

    def _conditional_candidate(self, rule: A.Rule) -> bool:
        atom = rule.atom
        if isinstance(atom, A.BandDefinition):
            kind = "healthy_band" if self.G.band_enabled else None
            return condition_family_is_enumerable(kind)
        if not isinstance(atom, A.Compare):
            return False
        kind = None
        if (
            atom.op == "\u007e\u221d"
            and isinstance(atom.left, A.Ref)
            and isinstance(atom.right, A.Ref)
            and atom.left != atom.right
        ):
            kind = "proportional"
        if (
            atom.op in ("~=", "==")
            and isinstance(atom.left, A.Ref)
            and isinstance(atom.right, A.Ref)
            and atom.left != atom.right
        ):
            kind = "pair"
        if (
            (isinstance(atom.left, A.Diff) and _is_zero(atom.right))
            or (isinstance(atom.right, A.Diff) and _is_zero(atom.left))
        ):
            kind = (
                "delta_zero"
                if atom.op in ("~=", "==")
                else "delta_bound"
            )
        return condition_family_is_enumerable(kind)


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