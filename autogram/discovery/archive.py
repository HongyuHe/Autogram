"""Archive for solver-certified rules.

Equivalent rules are collapsed with Z3.  Subsumed longer forms are discarded when a shorter kept
rule already implies them.  MDL is used only as the final tie-break inside a logical/statistical tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .evaluate import Evaluation
from ..logic.solver import equivalent, subsumes
from ..dsl.typecheck import _leaf_set


def _leaves(rule) -> frozenset:
    return frozenset(_leaf_set(rule.atom.left) | _leaf_set(rule.atom.right))


def _better(a: Evaluation, b: Evaluation) -> bool:
    primary_a = (a.hold_rate_lo, a.hold_rate, _op_strength(a.rule.atom.op), -a.rule.length())
    primary_b = (b.hold_rate_lo, b.hold_rate, _op_strength(b.rule.atom.op), -b.rule.length())
    if primary_a != primary_b:
        return primary_a > primary_b
    return a.mdl_gain > b.mdl_gain


def _op_strength(op: str) -> int:
    if op in ("==", "~=", "<|>"):
        return 3
    if op == "!=":
        return 2
    if op in ("<=", ">="):
        return 1
    return 0


def _term_has_scaled_slack(term) -> bool:
    from ..dsl import ast as A

    if isinstance(term, A.Scale):
        return term.coeff < 0.0 or abs(term.coeff) < 1.0
    if isinstance(term, A.Add):
        return any(_term_has_scaled_slack(t) for t in term.terms)
    return False


def _is_scaled_slack(ev: Evaluation) -> bool:
    return ev.rule.atom.op in ("<=", ">=") and (
        _term_has_scaled_slack(ev.rule.atom.left) or _term_has_scaled_slack(ev.rule.atom.right)
    )


def _is_zero_const(term) -> bool:
    from ..dsl import ast as A

    return isinstance(term, A.Const) and float(term.value) == 0.0


def _is_atomic_ref(term) -> bool:
    from ..dsl import ast as A

    return isinstance(term, A.Ref)


def _is_bloated_one_sided(ev: Evaluation) -> bool:
    atom = ev.rule.atom
    if atom.op not in ("<=", ">="):
        return False
    left_zero, right_zero = _is_zero_const(atom.left), _is_zero_const(atom.right)
    if not (left_zero or right_zero):
        return True
    measured = atom.right if left_zero else atom.left
    return not _is_atomic_ref(measured)


@dataclass
class ParetoArchive:
    cells: Dict[str, Evaluation] = field(default_factory=dict)

    def add(self, ev: Evaluation) -> bool:
        if not ev.accepted:
            return False
        if _is_scaled_slack(ev) or _is_bloated_one_sided(ev):
            return False
        ev_leaves = _leaves(ev.rule)
        for sig, cur in list(self.cells.items()):
            cur_leaves = _leaves(cur.rule)
            if cur.rule.binder != ev.rule.binder or cur_leaves != ev_leaves:
                continue
            if equivalent(cur.rule, ev.rule):
                if _better(ev, cur):
                    self.cells[sig] = ev
                    return True
                return False
            if cur.rule.length() <= ev.rule.length() and subsumes(cur.rule, ev.rule):
                return False
            if ev.rule.length() <= cur.rule.length() and subsumes(ev.rule, cur.rule):
                del self.cells[sig]
        self.cells[ev.rule.signature()] = ev
        return True

    def representatives(self) -> List[Evaluation]:
        """The kept rules: one representative per behaviour cell (not an evolutionary elite)."""
        return list(self.cells.values())

    def front(self) -> List[Evaluation]:
        es = self.representatives()
        out: List[Evaluation] = []
        for e in es:
            dominated = any(
                (o.hold_rate_lo >= e.hold_rate_lo and o.rule.length() <= e.rule.length()
                 and (o.hold_rate_lo > e.hold_rate_lo or o.rule.length() < e.rule.length()))
                for o in es if o is not e)
            if not dominated:
                out.append(e)
        return out

    def progress(self) -> float:
        return sum(e.hold_rate_lo for e in self.representatives())

    def portfolio(self, non_redundant: bool = False) -> List[Evaluation]:
        ranked = sorted(
            self.representatives(),
            key=lambda e: (e.hold_rate_lo, e.hold_rate, _op_strength(e.rule.atom.op), -e.rule.length(), e.mdl_gain),
            reverse=True,
        )
        if not non_redundant:
            return ranked
        kept: List[Evaluation] = []
        for e in ranked:
            e_leaves = _leaves(e.rule)
            if any(k.rule.length() <= e.rule.length()
                   and k.rule.binder == e.rule.binder
                   and _leaves(k.rule) == e_leaves
                   and subsumes(k.rule, e.rule) for k in kept):
                continue
            kept.append(e)
        return kept

    def signatures(self) -> set:
        return set(self.cells)
