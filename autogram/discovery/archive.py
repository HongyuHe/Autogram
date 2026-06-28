"""A simple Pareto archive keyed by ``(binder, length)`` cells (collapses the QD/Thompson stack).

The archive keeps a diverse portfolio of accepted invariants: one elite per ``(binder, length)``
behaviour cell, chosen by a single parsimony-adjusted quantity (MDL gain, tie-broken by
coverage).  The portfolio's own *progress* over rounds -- the improvement of its held-out
coverage x parsimony front -- is the leakage-safe success signal that replaces a catalogue
answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .evaluate import Evaluation
from ..dsl.typecheck import _leaf_set


def _rule_leaves(rule) -> frozenset:
    return frozenset(_leaf_set(rule.atom.left) | _leaf_set(rule.atom.right))


def _implies(simpler: Evaluation, other: Evaluation, cov_slack: float = 0.02) -> bool:
    """True if ``simpler`` (already kept) makes ``other`` redundant on the same binder."""
    if simpler.rule.binder != other.rule.binder:
        return False
    ls, lo = _rule_leaves(simpler.rule), _rule_leaves(other.rule)
    same_or_simpler_leaves = ls < lo or (ls == lo and simpler.rule.length() <= other.rule.length())
    return same_or_simpler_leaves and other.coverage <= simpler.coverage + cov_slack


@dataclass
class ParetoArchive:
    cells: Dict[tuple, Evaluation] = field(default_factory=dict)

    def add(self, ev: Evaluation) -> bool:
        """Insert ``ev`` if accepted and it beats its cell incumbent.  Returns improved?"""
        if not ev.accepted:
            return False
        key = ev.descriptor
        cur = self.cells.get(key)
        if cur is None or self._better(ev, cur):
            self.cells[key] = ev
            return True
        return False

    @staticmethod
    def _better(a: Evaluation, b: Evaluation) -> bool:
        if a.mdl_gain != b.mdl_gain:
            return a.mdl_gain > b.mdl_gain
        return a.coverage > b.coverage

    def elites(self) -> List[Evaluation]:
        return list(self.cells.values())

    def front(self) -> List[Evaluation]:
        """Non-dominated elites on (coverage maximize, length minimize)."""
        es = self.elites()
        out: List[Evaluation] = []
        for e in es:
            dominated = any(
                (o.coverage >= e.coverage and o.rule.length() <= e.rule.length()
                 and (o.coverage > e.coverage or o.rule.length() < e.rule.length()))
                for o in es if o is not e)
            if not dominated:
                out.append(e)
        return out

    def progress(self) -> float:
        """Scalar summary of the front: rewards more cells, higher coverage, more parsimony."""
        total = 0.0
        for e in self.elites():
            total += e.coverage * (1.0 + max(0.0, e.mdl_gain))
        return total

    def portfolio(self, non_redundant: bool = False) -> List[Evaluation]:
        """Accepted elites, de-duplicated by signature and ranked by parsimony then coverage.

        With ``non_redundant`` set, drop any rule whose measured leaves are a strict superset of
        an already-kept, higher-ranked rule on the same binder with comparable coverage -- i.e.
        a rule implied by a simpler accepted one (the novelty / non-redundancy criterion).
        """
        best: Dict[str, Evaluation] = {}
        for e in self.elites():
            sig = e.rule.signature()
            if sig not in best or self._better(e, best[sig]):
                best[sig] = e
        ranked = sorted(best.values(),
                        key=lambda e: (e.mdl_gain, e.coverage, -e.rule.length()), reverse=True)
        if not non_redundant:
            return ranked
        kept: List[Evaluation] = []
        for e in ranked:
            if not any(_implies(k, e) for k in kept):
                kept.append(e)
        return kept

    def signatures(self) -> set:
        return {e.rule.signature() for e in self.elites()}
