"""Archive for solver-certified rules.

Equivalent rules are collapsed with Z3.  Subsumed longer forms are discarded when a shorter kept
rule already implies them.  MDL is used only as the final tie-break inside a logical/statistical tie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .evaluate import Evaluation
from ..dsl import ast as A
from ..logic.solver import (
    equivalent,
    legacy_equivalent,
    legacy_subsumes,
    subsumes,
)
from ..dsl.typecheck import _leaf_set


def _leaves(rule) -> frozenset:
    from ..dsl import ast as A

    if isinstance(rule.atom, A.Compare):
        leaves = _leaf_set(rule.atom.left) | _leaf_set(rule.atom.right)
    elif isinstance(rule.atom, A.BooleanDefinition):
        leaves = _leaf_set(rule.atom.target)
        leaves |= _predicate_leaves(rule.atom.predicate)
        leaves.add(("predicate", rule.atom.predicate.unparse()))
    elif isinstance(rule.atom, A.CategoryDefinition):
        leaves = {
            ("category", rule.atom.target_column),
            *(("condition", column) for column, _value in rule.atom.cases),
        }
    elif isinstance(rule.atom, A.BandDefinition):
        leaves = _leaf_set(rule.atom.term)
    else:
        leaves = set()
    return frozenset(leaves)


def _predicate_leaves(predicate) -> set:
    from ..dsl import ast as A

    if isinstance(predicate, A.Bound):
        return _leaf_set(predicate.term)
    if isinstance(predicate, A.Sustained):
        return _predicate_leaves(predicate.predicate)
    if isinstance(predicate, A.Conjunction):
        out = set()
        for item in predicate.predicates:
            out |= _predicate_leaves(item)
        return out
    return set()


def _has_temporal(rule) -> bool:
    from ..dsl import ast as A

    def term_has(term):
        if isinstance(term, (A.Lag, A.Diff, A.Rolling)):
            return True
        if isinstance(term, A.Scale):
            return term_has(term.term)
        if isinstance(term, A.Add):
            return any(term_has(child) for child in term.terms)
        if isinstance(term, A.Mul):
            return term_has(term.left) or term_has(term.right)
        if isinstance(term, A.Div):
            return term_has(term.num) or term_has(term.den)
        return False

    if isinstance(rule.atom, A.Compare):
        return term_has(rule.atom.left) or term_has(rule.atom.right)
    return False


def _better(a: Evaluation, b: Evaluation) -> bool:
    primary_a = (a.hold_rate_lo, a.hold_rate, _atom_strength(a.rule.atom), -a.rule.length())
    primary_b = (b.hold_rate_lo, b.hold_rate, _atom_strength(b.rule.atom), -b.rule.length())
    if primary_a != primary_b:
        return primary_a > primary_b
    return a.mdl_gain > b.mdl_gain


def _same_fitted_semantics(a: Evaluation, b: Evaluation) -> bool:
    semantic_keys = {
        "center",
        "coefficient",
        "coefficients",
        "thresholds",
    }
    a_parameters = {
        key: value
        for key, value in dict(
            getattr(a, "parameters", {})
        ).items()
        if key in semantic_keys
    }
    b_parameters = {
        key: value
        for key, value in dict(
            getattr(b, "parameters", {})
        ).items()
        if key in semantic_keys
    }
    return (
        a.rule.atom == b.rule.atom
        and a.eps == b.eps
        and a.strictness == b.strictness
        and a_parameters == b_parameters
    )


def _cross_condition_subsumption_ok(keeper: Evaluation, candidate: Evaluation) -> bool:
    """Whether ``keeper`` may subsume ``candidate`` when their conditions differ.

    Z3 subsumption treats a condition as an opaque implication guard, so the only cross-condition
    subsumption it can prove is the tautology ``atom |= (C -> atom)`` -- an *unconditional* rule
    logically subsuming its own *conditioned* refinement. That subsumption is only genuine when the
    two share identical fitted semantics (same operator, epsilon, and any fitted center / ratio
    coefficient / learned threshold) *and* the unconditional rule holds at least as strongly; a
    weaker unconditional shadow (e.g. a 0.67 hold-rate approximation of a law that is exact under
    one regime, or a differently-fitted proportional coefficient) must never evict the stronger
    conditioned refinement, which carries strictly more information. When the conditions match this
    guard is irrelevant and ordinary same-condition subsumption applies.
    """
    if keeper.rule.condition == candidate.rule.condition:
        return True
    if keeper.rule.condition is not None:
        # A conditioned keeper cannot tautologically subsume a rule with a different condition.
        return False
    return (
        _same_fitted_semantics(keeper, candidate)
        and keeper.hold_rate + 1e-12 >= candidate.hold_rate
    )


def _op_strength(op: str) -> int:
    if op in ("==", "~=", "~∝", "<|>"):
        return 3
    if op == "!=":
        return 2
    if op in ("<=", ">=", "<", ">"):
        return 1
    return 0


def _atom_strength(atom) -> int:
    from ..dsl import ast as A

    if isinstance(atom, (A.BooleanDefinition, A.CategoryDefinition, A.BandDefinition)):
        return 4
    return _op_strength(atom.op)


def _term_has_scaled_slack(term) -> bool:
    from ..dsl import ast as A

    if isinstance(term, A.Scale):
        return term.coeff < 0.0 or abs(term.coeff) < 1.0
    if isinstance(term, A.Add):
        return any(_term_has_scaled_slack(t) for t in term.terms)
    return False


def _is_scaled_slack(ev: Evaluation) -> bool:
    from ..dsl import ast as A

    if not isinstance(ev.rule.atom, A.Compare):
        return False
    return ev.rule.atom.op in ("<=", ">=", "<", ">") and (
        _term_has_scaled_slack(ev.rule.atom.left) or _term_has_scaled_slack(ev.rule.atom.right)
    )


def _is_zero_const(term) -> bool:
    from ..dsl import ast as A

    return isinstance(term, A.Const) and float(term.value) == 0.0


def _is_atomic_ref(term) -> bool:
    from ..dsl import ast as A

    # A bare column, a finite difference, or a temporal *shift* of either is an atomic one-sided
    # target -- ``LAG_k(x) OP 0`` is a legitimate (if usually redundant) sign law over a single
    # column, not a bloated compound term. It is RETAINED here so a genuinely independent lag law
    # (whose raw column's atomic bound is not itself accepted -- e.g. a counter that dips negative
    # only in its final rows) survives discovery; ``portfolio(non_redundant=True)`` then suppresses
    # the trivial lag shadows that an EXACT atomic already proves, which keeps the temporal-null
    # control at zero without discarding the non-redundant lag laws.
    if isinstance(term, (A.Ref, A.Diff)):
        return True
    if isinstance(term, A.Lag):
        return _is_atomic_ref(term.term)
    return False


def _sign_bound_role_direction(rule):
    """``(binder, role, direction, atomic_strict, is_lag)`` for a one-sided sign bound ``T OP 0``.

    ``T`` is a bare ``Ref`` (atomic) or ``LAG_k(Ref)`` (lagged); returns ``None`` for anything else.
    ``direction`` is ``"lower"`` for a ``>=``/``>`` sign law and ``"upper"`` for ``<=``/``<``. This
    lets the portfolio match a lagged shadow against its atomic across archive cells (a lag and its
    atomic occupy different behaviour cells, so they are never compared during ``add``).
    """
    from ..dsl import ast as A

    atom = rule.atom
    if not isinstance(atom, A.Compare) or atom.op not in ("<", "<=", ">", ">="):
        return None
    left_zero = _is_zero_const(atom.left)
    right_zero = _is_zero_const(atom.right)
    if not (left_zero or right_zero):
        return None
    measured = atom.right if left_zero else atom.left
    if isinstance(measured, A.Ref):
        role, is_lag = measured.role, False
    elif isinstance(measured, A.Lag) and isinstance(measured.term, A.Ref):
        role, is_lag = measured.term.role, True
    else:
        return None
    points_toward = atom.op in ("<", "<=") if left_zero else atom.op in (">", ">=")
    direction = "lower" if points_toward else "upper"
    return rule.binder, role, direction, atom.op in (">", "<"), is_lag


def _suppress_lag_shadows(kept: List[Evaluation]) -> List[Evaluation]:
    """Drop ``LAG_k(x) OP 0`` bounds an EXACT atomic ``x OP 0`` already proves redundant.

    An atomic sign law that holds tolerance-free on every row (``raw_exact_sign``) forces its lagged
    shift on every valid lagged row, so the lag adds nothing and is only a shadow. A lag whose atomic
    is missing, or accepted only within tolerance (``hold_rate`` may reach 1.0 by absorbing a
    violation against a large scale), carries independent information and is KEPT -- that is the case
    a lag search must not miss. Restricting suppression to *raw-exact* atomics also holds the
    temporal-null control at zero: a non-negative null column's atomic sign law is exact and evicts
    its lag shadows.
    """
    exact_atomic = set()
    for evaluation in kept:
        info = _sign_bound_role_direction(evaluation.rule)
        if (
            info is not None
            and not info[4]
            and evaluation.rule.condition is None
            and getattr(evaluation, "raw_exact_sign", False)
        ):
            binder, role, direction, strict, _is_lag = info
            exact_atomic.add((binder, role, direction, strict))
    if not exact_atomic:
        return kept

    def _shadowed(evaluation: Evaluation) -> bool:
        info = _sign_bound_role_direction(evaluation.rule)
        if info is None or not info[4] or evaluation.rule.condition is not None:
            return False
        binder, role, direction, lag_strict, _is_lag = info
        # An exact atomic implies the lag iff same direction and its strictness is at least the
        # lag's (an exact ``x >= 0`` cannot witness a strict ``LAG > 0`` because of a zero value).
        return any(
            b == binder and r == role and d == direction and (astrict or not lag_strict)
            for (b, r, d, astrict) in exact_atomic
        )

    return [evaluation for evaluation in kept if not _shadowed(evaluation)]


def _is_bloated_one_sided(ev: Evaluation) -> bool:
    from ..dsl import ast as A

    atom = ev.rule.atom
    if not isinstance(atom, A.Compare):
        return False
    if atom.op not in ("<=", ">=", "<", ">"):
        return False
    left_zero, right_zero = _is_zero_const(atom.left), _is_zero_const(atom.right)
    if not (left_zero or right_zero):
        return True
    measured = atom.right if left_zero else atom.left
    return not _is_atomic_ref(measured)


def _is_atomic_one_sided(rule) -> bool:
    from ..dsl import ast as A

    if not isinstance(rule.atom, A.Compare):
        return False
    if rule.atom.op not in ("<=", ">=", "<", ">"):
        return False
    left_zero = _is_zero_const(rule.atom.left)
    right_zero = _is_zero_const(rule.atom.right)
    measured = rule.atom.right if left_zero else rule.atom.left
    return (left_zero or right_zero) and isinstance(measured, A.Ref)


def _atomic_reference_bound(rule):
    if not _is_atomic_one_sided(rule):
        return None
    atom = rule.atom
    zero_on_left = _is_zero_const(atom.left)
    points_toward_ref = atom.op in ("<", "<=") if zero_on_left else atom.op in (">", ">=")
    direction = "lower" if points_toward_ref else "upper"
    return direction, atom.op in (">", "<")


def _is_exact_zero_equality(ev: Evaluation) -> bool:
    from ..dsl import ast as A

    atom = ev.rule.atom
    if not isinstance(atom, A.Compare) or atom.op not in ("==", "~="):
        return False
    left_zero = _is_zero_const(atom.left)
    right_zero = _is_zero_const(atom.right)
    measured = atom.right if left_zero else atom.left
    return (
        (left_zero or right_zero)
        and isinstance(measured, A.Ref)
        and ev.hold_rate == 1.0
        and ev.eps <= 1e-8
    )


@dataclass
class ParetoArchive:
    cells: Dict[str, Evaluation] = field(default_factory=dict)
    legacy_compat: bool = False
    cell_index: Dict[tuple, Dict[str, Evaluation]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        for signature, evaluation in self.cells.items():
            self.cell_index.setdefault(
                self._cell_key(evaluation.rule),
                {},
            )[signature] = evaluation

    @staticmethod
    def _cell_key(rule) -> tuple:
        return rule.binder, _leaves(rule)

    def _store(
        self,
        signature: str,
        evaluation: Evaluation,
        cell_key: tuple,
    ) -> None:
        self.cells[signature] = evaluation
        self.cell_index.setdefault(cell_key, {})[
            signature
        ] = evaluation

    def _delete(self, signature: str, cell_key: tuple) -> None:
        del self.cells[signature]
        indexed = self.cell_index[cell_key]
        del indexed[signature]
        if not indexed:
            del self.cell_index[cell_key]

    def add(self, ev: Evaluation) -> bool:
        if self.legacy_compat:
            return self._add_legacy(ev)
        if not ev.accepted:
            return False
        if _is_scaled_slack(ev) or _is_bloated_one_sided(ev):
            return False
        cell_key = self._cell_key(ev.rule)
        for sig, cur in list(
            self.cell_index.get(cell_key, {}).items()
        ):
            if cur.rule.condition != ev.rule.condition:
                # An unconditional law logically subsumes a conditioned law with the same
                # atom + fitted semantics ONLY when it holds at least as strongly: if the
                # unconditional rule holds everywhere (hold-rate >= the conditioned rule's),
                # the condition adds nothing. A weaker unconditional shadow (e.g. a 0.67
                # hold-rate approximation of a law that is exact under one regime) must NOT
                # evict the exact conditioned refinement, which carries strictly more
                # information.
                if (
                    cur.rule.condition is None
                    and ev.rule.condition is not None
                    and _same_fitted_semantics(cur, ev)
                    and cur.hold_rate + 1e-12 >= ev.hold_rate
                ):
                    return False
                if (
                    ev.rule.condition is None
                    and cur.rule.condition is not None
                    and _same_fitted_semantics(cur, ev)
                    and ev.hold_rate + 1e-12 >= cur.hold_rate
                ):
                    self._delete(sig, cell_key)
                    continue
                continue
            cur_bound = _atomic_reference_bound(cur.rule)
            ev_bound = _atomic_reference_bound(ev.rule)
            if (
                cur_bound is not None
                and ev_bound is not None
                and cur_bound[0] == ev_bound[0]
                and cur.rule.condition == ev.rule.condition
            ):
                # Between a strict (``x > 0``) and a non-strict (``x >= 0``) bound in the same
                # direction, keep the STRICT one: it entails the non-strict, so the survivor covers
                # the discarded rule (archive completeness), and a strict known is recoverable only
                # from a strict portfolio rule. The bound tuple's second field is True when strict.
                if cur_bound[1] and not ev_bound[1]:
                    return False
                if not cur_bound[1] and ev_bound[1]:
                    self._delete(sig, cell_key)
                    continue
            if (
                isinstance(cur.rule.atom, A.Compare)
                and isinstance(ev.rule.atom, A.Compare)
                and {cur.rule.atom.op, ev.rule.atom.op} == {"==", "~="}
            ):
                continue
            if equivalent(cur.rule, ev.rule):
                if _better(ev, cur):
                    self._store(sig, ev, cell_key)
                    return True
                return False
            if _has_temporal(cur.rule) or _has_temporal(ev.rule):
                continue
            if (
                _is_atomic_one_sided(cur.rule)
                != _is_atomic_one_sided(ev.rule)
                and not (
                    _is_exact_zero_equality(cur)
                    or _is_exact_zero_equality(ev)
                )
            ):
                continue
            if cur.rule.length() <= ev.rule.length() and subsumes(cur.rule, ev.rule):
                return False
            if ev.rule.length() <= cur.rule.length() and subsumes(ev.rule, cur.rule):
                self._delete(sig, cell_key)
        self._store(ev.rule.signature(), ev, cell_key)
        return True

    def _add_legacy(self, ev: Evaluation) -> bool:
        if not ev.accepted:
            return False
        if _is_scaled_slack(ev) or _is_bloated_one_sided(ev):
            return False
        cell_key = self._cell_key(ev.rule)
        for sig, cur in list(
            self.cell_index.get(cell_key, {}).items()
        ):
            if legacy_equivalent(cur.rule, ev.rule):
                if _better(ev, cur):
                    self._store(sig, ev, cell_key)
                    return True
                return False
            if (
                cur.rule.length() <= ev.rule.length()
                and legacy_subsumes(cur.rule, ev.rule)
            ):
                return False
            if (
                ev.rule.length() <= cur.rule.length()
                and legacy_subsumes(ev.rule, cur.rule)
            ):
                self._delete(sig, cell_key)
        self._store(ev.rule.signature(), ev, cell_key)
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
            key=lambda e: (e.hold_rate_lo, e.hold_rate, _atom_strength(e.rule.atom), -e.rule.length(), e.mdl_gain),
            reverse=True,
        )
        if not non_redundant:
            return ranked
        subsumption = legacy_subsumes if self.legacy_compat else subsumes
        kept: List[Evaluation] = []
        kept_by_cell: Dict[tuple, List[Evaluation]] = {}
        for e in ranked:
            cell_key = self._cell_key(e.rule)
            comparable = kept_by_cell.get(cell_key, ())
            if self.legacy_compat:
                redundant = any(
                    k.rule.length() <= e.rule.length()
                    and subsumption(k.rule, e.rule)
                    for k in comparable
                )
            else:
                redundant = any(
                    k.rule.length() <= e.rule.length()
                    and not (
                        _has_temporal(k.rule)
                        or _has_temporal(e.rule)
                    )
                    and (
                        _is_atomic_one_sided(k.rule)
                        == _is_atomic_one_sided(e.rule)
                        or _is_exact_zero_equality(k)
                        or _is_exact_zero_equality(e)
                    )
                    and _cross_condition_subsumption_ok(k, e)
                    and subsumption(k.rule, e.rule)
                    for k in comparable
                )
            if redundant:
                continue
            kept.append(e)
            kept_by_cell.setdefault(cell_key, []).append(e)
        return _suppress_lag_shadows(kept)

    def signatures(self) -> set:
        return set(self.cells)
