"""Exhaustive enumeration of the bounded Autogram grammar."""

from __future__ import annotations

import itertools
from typing import Dict, List, Sequence

from ..dsl import ast as A
from ..dsl.grammar import Grammar
from ..dsl.typecheck import is_admissible
from ..logic.solver import is_trivial


_SYMMETRIC_OPS = {"~=", "==", "!=", "<|>"}


def _term_key(t: A.Term) -> str:
    return t.unparse()


def _term_sort_key(t: A.Term) -> tuple:
    rank = {A.Ref: 0, A.Agg: 1, A.Scale: 2, A.Add: 3, A.Const: 4}
    return (rank.get(type(t), 9), t.unparse())


def _is_zero(t: A.Term) -> bool:
    return isinstance(t, A.Const) and float(t.value) == 0.0


def normalize_term(t: A.Term) -> A.Term:
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
    left = normalize_term(rule.atom.left)
    right = normalize_term(rule.atom.right)
    op = rule.atom.op
    if op in _SYMMETRIC_OPS and _term_key(right) < _term_key(left):
        left, right = right, left
    elif op == "<=" and _is_zero(left) and not _is_zero(right):
        left, right, op = right, left, ">="
    elif op == ">=" and _is_zero(left) and not _is_zero(right):
        left, right, op = right, left, "<="
    return A.Rule(rule.binder, A.Compare(left, op, right), tag=rule.tag)


class EnumerationProposer:
    """Enumerate every admissible rule in a finite grammar bound.

    ``n`` is accepted for compatibility with the discovery loop but does not limit enumeration;
    use ``Grammar.max_complexity`` to bound the hypothesis space.
    """

    def __init__(self, G: Grammar):
        self.G = G
        self._cache: List[A.Rule] | None = None

    def propose(self, n: int = 0, seeds: Sequence[A.Rule] = (), rng=None) -> List[A.Rule]:
        if self._cache is None:
            self._cache = self._enumerate()
        return list(self._cache)

    def _base_terms_for(self, binder: str) -> List[A.Term]:
        terms: Dict[str, A.Term] = {}
        for role in self.G.refs_for(binder):
            t = A.Ref(role)
            if t.complexity() <= self.G.max_complexity:
                terms[_term_key(t)] = t
        for fam in self.G.fams_for(binder):
            for kind in ("SUM",):
                if kind not in self.G.agg_kinds:
                    continue
                t = A.Agg(kind, fam)
                if t.complexity() <= self.G.max_complexity:
                    terms[_term_key(t)] = t
        return [terms[k] for k in sorted(terms)]

    def _scaled_terms_for(self, base_terms: Sequence[A.Term]) -> List[A.Term]:
        terms: Dict[str, A.Term] = {}
        for t in base_terms:
            for coeff in self.G.scale_coeffs:
                st = normalize_term(A.Scale(float(coeff), t))
                if st.complexity() <= self.G.max_complexity:
                    terms[_term_key(st)] = st
        return [terms[k] for k in sorted(terms)]

    def _add_terms_for(self, binder: str) -> List[A.Term]:
        refs = [A.Ref(r) for r in self.G.refs_for(binder)]
        aggs = [A.Agg("SUM", fam) for fam in self.G.fams_for(binder) if "SUM" in self.G.agg_kinds]
        leaves = refs + aggs
        terms: Dict[str, A.Term] = {}
        for arity in range(2, max(1, self.G.max_add_arity) + 1):
            for combo in itertools.combinations(leaves, arity):
                at = normalize_term(A.Add(tuple(combo)))
                if at.complexity() <= self.G.max_complexity:
                    terms[_term_key(at)] = at
        return [terms[k] for k in sorted(terms)]

    @staticmethod
    def _is_scaled_slack(left: A.Term, op: str, right: A.Term) -> bool:
        if op not in ("<=", ">=") or not isinstance(left, A.Scale) or isinstance(right, A.Const):
            return False
        return left.coeff < 0.0 or abs(left.coeff) < 1.0

    def _candidate_rules(self):
        for binder in self.G.binders:
            zero = A.Const(0.0)
            base_terms = self._base_terms_for(binder)
            scaled_terms = self._scaled_terms_for(base_terms)
            add_terms = self._add_terms_for(binder)
            measured = base_terms + add_terms
            simple_terms = [zero] + base_terms
            for t in measured:
                yield A.Rule(binder, A.Compare(t, ">=", zero))
                yield A.Rule(binder, A.Compare(t, "<=", zero))
            # Base-vs-base carries the full operator set including same-family separations.
            for i, left in enumerate(simple_terms):
                for j, right in enumerate(simple_terms):
                    if i == j:
                        continue
                    for op in self.G.ops:
                        if op in _SYMMETRIC_OPS and j <= i:
                            continue
                        yield A.Rule(binder, A.Compare(left, op, right))
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

    def _enumerate(self) -> List[A.Rule]:
        out: List[A.Rule] = []
        seen = set()
        for raw in self._candidate_rules():
            rule = normalize_rule(raw)
            if rule.complexity() > self.G.max_complexity:
                continue
            ok, _ = is_admissible(rule, self.G)
            if not ok:
                continue
            if is_trivial(rule):
                continue
            sig = rule.signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(rule)
        return out


# Backward-compatible name for callers that still ask for a random proposer; it now enumerates.
RandomProposer = EnumerationProposer
