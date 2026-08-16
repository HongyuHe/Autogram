"""Guarantees-first evaluator for numeric relations and total definitions.

Z3 decides logical triviality.  The only data statistic is hold-rate with a Wilson confidence
interval.  MDL is computed for tie-breaking, never as an acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DiscoveryConfig
from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings
from ..dsl.evaluate import (
    _blowup,
    _condition_mask,
    _consecutive_window_ends,
    _ordered_groups,
    _union_overflow,
    _window_overflow,
    _typed_object_array,
    eval_term,
    eval_term_overflow,
    ground,
    robust_median,
    typed_group_key,
)
from ..dsl.typecheck import _has_boolean_ref
from ..evaluator.band import fit_band_auto, violation_magnitude
from ..evaluator.metrics import mdl_gain, wilson, z_for_alpha
from ..evaluator.threshold import rule_threshold
from ..logic.solver import is_trivial, legacy_is_trivial


@dataclass
class Evaluation:
    rule: A.Rule
    accepted: bool
    reason: str
    eps: float
    hold_rate: float
    hold_rate_lo: float
    hold_rate_hi: float
    statistic: str
    support: float
    n_points: int
    n_bindings: int
    mdl_gain: float
    strictness: str
    descriptor: tuple
    # Compatibility aliases for export/report callers.
    threshold: float = 0.0
    coverage: float = 0.0
    coverage_lo: float = 0.0
    coverage_hi: float = 0.0
    operating_cov: float = 0.0
    stability_std: float = 0.0
    stability_min: float = 1.0
    support_margin: float = 0.0
    stability_margin: float = 0.0
    # True only for an atomic sign bound ``x OP 0`` that holds with ZERO tolerance on every row.
    # ``hold_rate`` can be 1.0 while a violation is absorbed by the acceptance tolerance against a
    # large scale, so this flag -- not ``hold_rate`` -- is what proves an atomic sign law exact.
    raw_exact_sign: bool = False
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        self.coverage = self.hold_rate
        self.coverage_lo = self.hold_rate_lo
        self.coverage_hi = self.hold_rate_hi
        self.operating_cov = self.hold_rate
        self.support_margin = self.hold_rate_lo
        self.stability_margin = self.hold_rate_lo

    def summary(self, adapter=None) -> str:
        from ..dsl.render import render_rule
        params = f" params={self.parameters}" if self.parameters else ""
        return (f"{render_rule(self.rule, adapter):<54s} {self.strictness:<10s} "
                f"hold={self.hold_rate:.3f}[{self.hold_rate_lo:.2f},{self.hold_rate_hi:.2f}] "
                f"mdl={self.mdl_gain:+.2f}{params} {self.reason}")


def _strictness(op: str, eps: float, rel: np.ndarray, hold: np.ndarray) -> str:
    if op == "<|>":
        return "existence"
    if op == "!=":
        return "separation"
    if op == "~∝":
        return "proportional"
    if op in (">=", "<=", ">", "<"):
        return "one-sided"
    if rel.size and float(np.max(rel)) <= 1e-12:
        return "exact"
    if np.all(hold):
        return "soft"
    return "loose"


def _finite_ulp(values: np.ndarray) -> np.ndarray:
    """Finite one-ULP magnitude, including at either float64 ceiling."""
    values = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        spacing = np.abs(np.spacing(values))
        inward = np.abs(values - np.nextafter(values, 0.0))
    return np.where(np.isfinite(spacing), spacing, inward)


def _positive_scale_floor(values: np.ndarray) -> float:
    return max(
        1e-6 * (robust_median(values) if values.size else 1.0),
        float(np.nextafter(0.0, 1.0)),
    )


def _group_labels(frame, name_model, row_indices=None):
    keys = tuple(
        getattr(
            getattr(name_model, "adapter", None),
            "group_keys",
            (),
        )
    )
    if not keys or not all(
        key in frame.row_context
        for key in keys
    ):
        return None
    if len(keys) == 1:
        labels = _typed_object_array(frame.row_context[keys[0]])
    else:
        columns = [
            _typed_object_array(frame.row_context[key])
            for key in keys
        ]
        labels = np.empty(frame.n_rows, dtype=object)
        for index, values in enumerate(zip(*columns)):
            labels[index] = tuple(values)
    if row_indices is not None:
        labels = labels[np.asarray(row_indices, dtype=int)]
    return labels


class _GlobalGroup:
    """Sentinel for "the data is not grouped".

    A string sentinel collides with a real group whose label happens to be that string, which then
    reports one group's coefficient under the other's name.
    """

    __slots__ = ()

    def __repr__(self) -> str:                       # pragma: no cover - display only
        return "global"


GLOBAL_GROUP = _GlobalGroup()


def _typed_label(label):
    """Backward-compatible alias for the one shared recursive typed identity."""
    return typed_group_key(label)


def _typed_equal_array(left, right) -> np.ndarray:
    """Elementwise categorical equality under recursive typed identity."""
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("typed categorical operands must have the same shape")
    return np.fromiter(
        (
            _typed_label(left_value) == _typed_label(right_value)
            for left_value, right_value in zip(left.flat, right.flat)
        ),
        dtype=bool,
        count=left.size,
    ).reshape(left.shape)


def _typed_scalar_mask(values, target) -> np.ndarray:
    """Rows of ``values`` equal to ``target`` under recursive typed identity."""
    values = np.asarray(values, dtype=object)
    target_key = _typed_label(target)
    return np.fromiter(
        (_typed_label(value) == target_key for value in values.flat),
        dtype=bool,
        count=values.size,
    ).reshape(values.shape)


def _untyped_label(typed):
    """Recover the displayable label from a typed identity."""
    kind, value = typed
    if kind == "tuple":
        return tuple(_untyped_label(item) for item in value)
    return value


def _display_keys(labels) -> dict:
    """Map TYPED group identities to collision-free display keys for reporting.

    Reported parameters are persisted to JSON and to the learned `.dl` portfolio, so they have to
    round-trip the *identity* of a group. Stringifying is lossy: distinct labels ``1`` and ``"1"``
    collapse to one key and one group's fitted coefficient silently overwrites the other's, leaving
    an accepted per-group law that cannot be audited or reproduced. Plain ``str`` is kept while it
    is unambiguous -- that is the readable common case -- and the *whole set* falls back to ``repr``
    otherwise. Falling back only for the labels that collide is not enough: ``1``, ``"1"`` and
    ``"'1'"`` render as ``1``, ``'1'`` and ``'1'``, so a per-label fallback re-collides with a label
    that was never ambiguous to begin with.
    """
    typed = list(dict.fromkeys(labels))
    displayed = [_untyped_label(item) for item in typed]
    for render in (str, repr):
        keys = [render(label) for label in displayed]
        if len(set(keys)) == len(typed):
            return dict(zip(typed, keys))
    # Neither rendering separates them (``True``/``1`` share both, and two objects can share a
    # repr), so qualify by type and then by position. Injectivity is what the audit trail needs.
    keys = [f"{item[0]}:{label!r}" for item, label in zip(typed, displayed)]
    if len(set(keys)) != len(typed):
        keys = [f"{key}#{index}" for index, key in enumerate(keys)]
    assert len(set(keys)) == len(typed)
    return dict(zip(typed, keys))


def _binding_key(binding: dict) -> tuple:
    """Hashable identity of one binding, for per-binding bookkeeping."""
    return tuple(sorted(binding.items()))


def _group_hold_gate(
    holds,
    groups,
    *,
    z: float,
    threshold: float,
    use_wilson: bool = False,
    failed_groups=None,
) -> tuple[bool, dict]:
    if failed_groups is not None:
        failed_groups = np.asarray(failed_groups, dtype=object)
        if failed_groups.size:
            if groups is None:
                return False, {}
            holds = np.concatenate((
                np.asarray(holds, dtype=bool),
                np.zeros(failed_groups.size, dtype=bool),
            ))
            groups = np.concatenate((
                np.asarray(groups, dtype=object),
                failed_groups,
            ))
    if groups is None:
        return True, {}
    groups = np.asarray(groups, dtype=object)
    if groups.shape != np.asarray(holds).shape:
        return False, {}
    rates = {}
    lows = {}
    accepted = True
    ordered_typed = list(dict.fromkeys(_typed_label(item) for item in groups.tolist()))
    display = _display_keys(ordered_typed)
    for typed in ordered_typed:
        label = typed[1]
        mask = np.asarray([
            _typed_label(item) == typed
            for item in groups
        ], dtype=bool)
        count = int(np.count_nonzero(mask))
        group_lo, _group_hi, group_rate = wilson(
            int(np.count_nonzero(np.asarray(holds)[mask])),
            count,
            z=z,
        )
        key = display[typed]
        rates[key] = group_rate
        lows[key] = group_lo
        gate_value = group_lo if use_wilson else group_rate
        accepted &= gate_value >= threshold
    return accepted, {
        "group_hold_rates": rates,
        "group_hold_rate_lows": lows,
    }


def _term_intrinsic_error(term: A.Term) -> Optional[str]:
    if isinstance(term, (A.Const, A.Ref, A.RelatedAgg)):
        return None
    if isinstance(term, A.Agg):
        if term.kind not in A.AGG_KINDS:
            return f"unknown intrinsic aggregation {term.kind!r}"
        return None
    if isinstance(term, (A.Scale, A.Lag, A.Diff, A.Rolling)):
        if (
            isinstance(term, (A.Lag, A.Diff))
            and (
                not isinstance(term.steps, int)
                or isinstance(term.steps, bool)
                or term.steps <= 0
            )
        ):
            return "temporal steps must be positive"
        if (
            isinstance(term, A.Rolling)
            and (
                not isinstance(term.window, int)
                or isinstance(term.window, bool)
                or term.window <= 0
            )
        ):
            return "rolling window must be positive"
        if (
            isinstance(term, A.Rolling)
            and term.kind not in ("SUM", "MIN", "MAX", "AVG")
        ):
            return f"unknown intrinsic aggregation {term.kind!r}"
        return _term_intrinsic_error(term.term)
    if isinstance(term, A.Add):
        children = term.terms
    elif isinstance(term, (A.Mul, A.Div)):
        children = (
            (term.left, term.right)
            if isinstance(term, A.Mul)
            else (term.num, term.den)
        )
    else:
        return f"unknown intrinsic term {type(term).__name__!r}"
    for child in children:
        error = _term_intrinsic_error(child)
        if error is not None:
            return error
    return None


def _predicate_intrinsic_error(predicate: A.Predicate) -> Optional[str]:
    if isinstance(predicate, A.Bound):
        if predicate.op not in ("<", "<=", ">", ">="):
            return f"unknown intrinsic operator {predicate.op!r}"
        return _term_intrinsic_error(predicate.term)
    if isinstance(predicate, A.Sustained):
        return _predicate_intrinsic_error(predicate.predicate)
    if isinstance(predicate, A.Conjunction):
        for child in predicate.predicates:
            error = _predicate_intrinsic_error(child)
            if error is not None:
                return error
        return None
    return f"unknown intrinsic predicate {type(predicate).__name__!r}"


def _rule_intrinsic_error(rule: A.Rule) -> Optional[str]:
    atom = rule.atom
    if isinstance(atom, A.Compare):
        if atom.op not in A.OPS:
            return f"unknown intrinsic operator {atom.op!r}"
        terms = (atom.left, atom.right)
    elif isinstance(atom, A.BandDefinition):
        terms = (atom.term,)
    elif isinstance(atom, A.BooleanDefinition):
        terms = (atom.target,)
        error = _predicate_intrinsic_error(atom.predicate)
        if error is not None:
            return error
    elif isinstance(atom, A.CategoryDefinition):
        terms = ()
    else:
        return f"unknown intrinsic atom {type(atom).__name__!r}"
    for term in terms:
        error = _term_intrinsic_error(term)
        if error is not None:
            return error
    return None


class DataOnlyEvaluator:
    """Evaluate a rule against observable data only."""

    def __init__(self, ds, cfg: DiscoveryConfig = None):
        self.ds = ds
        self.cfg = cfg or DiscoveryConfig()
        adapter = getattr(ds.name_model, "adapter", None)
        uses_extended_capabilities = (
            bool(getattr(adapter, "temporal_enabled", False))
            or bool(getattr(adapter, "conditional_enabled", False))
            or bool(getattr(adapter, "advanced_enabled", False))
            or bool(getattr(adapter, "band_enabled", False))
            or bool(getattr(adapter, "related_templates", {}))
            or "~\u221d" in tuple(getattr(adapter, "ops", ()))
            or "<" in tuple(getattr(adapter, "ops", ()))
            or ">" in tuple(getattr(adapter, "ops", ()))
        )
        self.legacy_compat = (
            getattr(adapter, "codec_kind", "") == "dict_gt_hidden"
            and not uses_extended_capabilities
        )

    def evaluate(self, rule: A.Rule, *, logically_screened: bool = False) -> Evaluation:
        intrinsic_error = _rule_intrinsic_error(rule)
        if intrinsic_error is not None:
            return self._reject(rule, intrinsic_error)
        if isinstance(rule.atom, A.Compare):
            adapter = self.ds.name_model.adapter
            left_boolean = _has_boolean_ref(
                rule.atom.left,
                rule.binder,
                adapter,
            )
            right_boolean = _has_boolean_ref(
                rule.atom.right,
                rule.binder,
                adapter,
            )
            if (left_boolean or right_boolean) and not (
                isinstance(
                    rule.atom.left,
                    (A.Ref, A.RelatedAgg),
                )
                and isinstance(
                    rule.atom.right,
                    (A.Ref, A.RelatedAgg),
                )
                and left_boolean
                and right_boolean
                and rule.atom.op in ("==", "!=", "<|>")
            ):
                return self._reject(
                    rule,
                    "Boolean refs cannot enter numeric comparisons",
                )
        if (
            self.legacy_compat
            and isinstance(rule.atom, A.Compare)
            and rule.condition is None
        ):
            return self._evaluate_legacy_compare(
                rule,
                logically_screened=logically_screened,
            )
        cfg = self.cfg
        nm = self.ds.name_model
        frame = self.ds.observed
        if not logically_screened and is_trivial(rule):
            return self._reject(rule, "solver-trivial tautology/contradiction")
        if isinstance(rule.atom, (A.BooleanDefinition, A.CategoryDefinition)):
            # Definitions are subject to the SAME conditional minimum-support floor as comparisons
            # and bands: a rule restricted to a handful of rows is not evidence of a law, no matter
            # how cleanly it separates them. Guard before scoring so a thin condition can never be
            # accepted through the definition path.
            support_rejection = self._condition_support_rejection(rule)
            if support_rejection is not None:
                return support_rejection
        if isinstance(rule.atom, (A.BooleanDefinition, A.BandDefinition)):
            # ... and to the same finite-arithmetic guard, for the same reason: a definition marks a
            # non-finite row invalid, so without this it would be scored on whatever its own
            # overflow left behind.
            overflow_rejection, tainted_rows = self._definition_overflow_rejection(rule)
            if overflow_rejection is not None:
                return overflow_rejection
        else:
            tainted_rows = None
        if isinstance(rule.atom, A.BooleanDefinition):
            return self._evaluate_boolean_definition(rule, tainted_rows)
        if isinstance(rule.atom, A.CategoryDefinition):
            return self._evaluate_category_definition(rule)
        if isinstance(rule.atom, A.BandDefinition):
            return self._evaluate_band_definition(rule, tainted_rows)
        op = rule.atom.op

        g = ground(rule, frame, nm, subsample=cfg.subsample, seed=cfg.seed)
        overflow_rejection = self._overflow_rejection(rule, g)
        if overflow_rejection is not None:
            return overflow_rejection
        if rule.condition is not None and (
            g.graded_points < int(cfg.min_condition_points)
            or g.graded_condition_support < float(cfg.min_condition_fraction)
        ):
            return self._reject(
                rule,
                "condition support below minimum "
                f"({g.graded_points} points, {g.graded_condition_support:.3f} of rows)",
            )
        if g.degenerate or g.n_points == 0:
            return self._reject(
                rule,
                f"grounded 0 points for binder {rule.binder!r} "
                f"({g.n_bindings}/{g.n_candidates} non-degenerate bindings)",
            )

        rho = g.rho
        scale = g.scale
        overflow_support = None
        parameters = {}
        evaluation_groups = _group_labels(
            frame,
            nm,
            g.row_indices,
        )
        overflow_rows = getattr(g, "overflow_row_indices", None)
        failed_groups = (
            _group_labels(frame, nm, overflow_rows)
            if overflow_rows is not None and overflow_rows.size
            else None
        )
        if op == "~∝":
            fitted = _fit_proportional(g, frame, nm, cfg)
            if fitted is None:
                return self._reject(rule, "proportional coefficient could not be fit")
            (
                coefficients,
                coefficient_by_point,
                evaluation_mask,
                evaluation_groups,
            ) = fitted
            with np.errstate(over="ignore", invalid="ignore"):
                rho = g.left - coefficient_by_point * g.right
                scale = np.maximum(
                    np.abs(g.left), np.abs(coefficient_by_point * g.right),
                )
            # The fitted coefficient introduces arithmetic that `ground()` never saw, so the
            # finite-arithmetic guard has to be re-applied to the POST-FIT residual: a large
            # coefficient can overflow the product or the difference on rows whose operands were
            # perfectly finite. Three things make the check honest. It runs on the FULL graded
            # population, not the coreset, because a subsample that happens to miss the blown-up row
            # would otherwise accept the rule. It covers every grounded row rather than the
            # evaluation split, because a blow-up is a failure of the rule's own expression wherever
            # it occurs. And it is measured against the same attempted-row denominator as the base
            # guard and added to it, so two disjoint sub-cap overflows cannot pass a single cap
            # between them.
            full_left = g.full_left if g.full_left is not None else g.left
            full_right = g.full_right if g.full_right is not None else g.right
            full_coefficient = self._coefficient_for_points(
                coefficients, frame, nm, g, full_left.size,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                full_scaled = full_coefficient * full_right
                full_rho = full_left - full_scaled
            fit_overflow = _union_overflow(
                _blowup(full_scaled, full_coefficient, full_right),
                _blowup(full_rho, full_left, full_scaled),
            )
            fit_blown = 0 if fit_overflow is None else int(np.count_nonzero(fit_overflow))
            if fit_blown:
                fit_rows = (
                    g.full_row_indices
                    if g.full_row_indices is not None
                    else g.row_indices
                )
                fit_failed_groups = _group_labels(
                    frame,
                    nm,
                    fit_rows[fit_overflow],
                )
                if fit_failed_groups is not None:
                    failed_groups = (
                        fit_failed_groups
                        if failed_groups is None
                        else np.concatenate((
                            np.asarray(failed_groups, dtype=object),
                            np.asarray(fit_failed_groups, dtype=object),
                        ))
                    )
                attempted = int(g.attempted_points) or int(full_left.size)
                blown = int(g.overflow_points) + fit_blown
                rejection = self._overflow_reject_if(
                    rule,
                    overflow_points=blown,
                    graded_points=max(0, int(g.graded_points) - fit_blown),
                    fraction=(float(blown) / float(attempted)) if attempted else 0.0,
                )
                if rejection is not None:
                    return rejection
                # Tolerated overflow is still not evidence. The blown-up rows leave the scored
                # population, and the reported support shrinks with them; otherwise a rule that is
                # allowed a small overflow budget would be graded on the survivors while still
                # claiming the full population, which is the very inflation the guard exists to
                # prevent.
                with np.errstate(over="ignore", invalid="ignore"):
                    scored_scaled = coefficient_by_point * g.right
                scored_overflow = _union_overflow(
                    _blowup(scored_scaled, coefficient_by_point, g.right),
                    _blowup(rho, g.left, scored_scaled),
                )
                if scored_overflow is not None:
                    evaluation_mask = evaluation_mask & ~scored_overflow
                    if not np.any(evaluation_mask):
                        return self._reject(
                            rule,
                            "proportional law overflowed on every evaluated row",
                        )
                overflow_support = self._support_excluding(g, fit_blown)
                support_rejection = self._graded_support_rejection(
                    rule, g, fit_blown,
                )
                if support_rejection is not None:
                    return support_rejection
            positive = scale[scale > 0]
            floor = _positive_scale_floor(positive)
            scale = np.maximum(scale, floor)
            parameters = {
                "coefficient": float(robust_median(np.asarray(list(coefficients.values()), dtype=float))),
                "coefficients": _reported_coefficients(coefficients),
            }
            rho = rho[evaluation_mask]
            scale = scale[evaluation_mask]
            # `_fit_proportional` sliced the group labels with the mask it returned; narrowing the
            # mask above means they have to be re-sliced or they would be row-misaligned.
            all_labels = _group_labels(frame, nm, g.row_indices)
            if all_labels is not None:
                evaluation_groups = all_labels[evaluation_mask]

        rel = np.abs(rho) / scale
        if op == "<|>":
            eps = cfg.presence_tolerance
            floor = np.maximum(g.scale, 1.0) * eps
            left_present = np.abs(g.left) > floor
            right_present = np.abs(g.right) > floor
            if not ((np.any(left_present) and np.any(~left_present))
                    or (np.any(right_present) and np.any(~right_present))):
                return self._reject(rule, "existence has no observed presence/absence variation")
            holds = left_present == right_present
            rel = np.where(holds, 0.0, 1.0)
        elif op == "!=":
            eps = cfg.separation_tolerance
            holds = rel > eps
        elif op in (">", "<"):
            eps = cfg.ordering_tolerance
            signed = rho / scale
            holds = signed > eps if op == ">" else signed < -eps
        elif op == "==":
            ulp = np.maximum(_finite_ulp(g.left), _finite_ulp(g.right))
            exact_tolerance = 4.0 * ulp
            if not np.all(np.isfinite(exact_tolerance)):
                return self._reject(
                    rule,
                    "exact equality tolerance is non-finite",
                )
            holds = np.abs(rho) <= exact_tolerance
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                normalized_tolerance = exact_tolerance / scale
            if not np.all(np.isfinite(normalized_tolerance)):
                return self._reject(
                    rule,
                    "normalized exact equality tolerance is non-finite",
                )
            eps = float(np.max(normalized_tolerance))
        else:
            if cfg.band_mode == "adaptive" and op in ("~=", "~∝"):
                # item 4: per-candidate self-calibrated band (knee + split-conformal holdout),
                # capped at the global tolerance so it can only *tighten*, never widen-to-accept.
                bf, _cov = fit_band_auto(op, rho, scale, cfg.band_holdout_frac, cfg.seed,
                                         cap=cfg.tolerance,
                                         groups=evaluation_groups)
                if evaluation_groups is not None and bf.n_eval == 0:
                    return self._reject(
                        rule,
                        "adaptive band could not retain every declared group",
                    )
                eps = max(float(bf.eps), 1e-9)
                rho = rho[bf.eval_indices]
                scale = scale[bf.eval_indices]
                rel = np.abs(rho) / scale
                if evaluation_groups is not None:
                    evaluation_groups = evaluation_groups[
                        bf.eval_indices
                    ]
            else:
                eps = cfg.tolerance
            holds = violation_magnitude(op, rho, scale) <= eps + 1e-15
        k = int(np.count_nonzero(holds))
        z = z_for_alpha(cfg.ci_alpha)
        lo, hi, phat = wilson(k, int(holds.size), z=z)
        gain = mdl_gain(rule, eps, rel)
        strict = _strictness(op, eps, rel, holds)
        descriptor = (rule.binder, rule.length())
        thr = rule_threshold(cfg, op=op, strictness=strict, complexity=rule.complexity(),
                             n_bindings=g.n_bindings, eps=eps)
        ok = lo >= thr
        group_ok, group_parameters = _group_hold_gate(
            holds,
            evaluation_groups,
            z=z,
            threshold=thr,
            use_wilson=op == "~\u221d",
            failed_groups=failed_groups,
        )
        ok &= group_ok
        parameters.update(group_parameters)
        reason = ("hold-rate above Wilson threshold" if ok
                  else "hold-rate Wilson lower bound below threshold")
        return Evaluation(
            rule=rule, accepted=ok, reason=reason, eps=eps,
            hold_rate=phat, hold_rate_lo=lo, hold_rate_hi=hi, statistic="hold_rate",
            support=(g.support if overflow_support is None else overflow_support),
            n_points=int(holds.size), n_bindings=g.n_bindings,
            mdl_gain=gain, strictness=strict, descriptor=descriptor, threshold=thr,
            raw_exact_sign=g.raw_exact_sign,
            parameters=parameters)

    def _evaluate_legacy_compare(
        self,
        rule: A.Rule,
        *,
        logically_screened: bool,
    ) -> Evaluation:
        cfg = self.cfg
        if not logically_screened and legacy_is_trivial(rule):
            return self._reject(
                rule,
                "solver-trivial tautology/contradiction",
            )
        g = ground(
            rule,
            self.ds.observed,
            self.ds.name_model,
            subsample=cfg.subsample,
            seed=cfg.seed,
        )
        overflow_rejection = self._overflow_rejection(rule, g)
        if overflow_rejection is not None:
            return overflow_rejection
        if g.degenerate or g.n_points == 0:
            return self._reject(
                rule,
                f"grounded 0 points for binder {rule.binder!r} "
                f"({g.n_bindings}/{g.n_candidates} non-degenerate bindings)",
            )

        op = rule.atom.op
        rel = np.abs(g.rho) / g.scale
        if op == "<|>":
            eps = cfg.presence_tolerance
            floor = np.maximum(g.scale, 1.0) * eps
            left_present = np.abs(g.left) > floor
            right_present = np.abs(g.right) > floor
            if not (
                (
                    np.any(left_present)
                    and np.any(~left_present)
                )
                or (
                    np.any(right_present)
                    and np.any(~right_present)
                )
            ):
                return self._reject(
                    rule,
                    "existence has no observed presence/absence variation",
                )
            holds = left_present == right_present
            rel = np.where(holds, 0.0, 1.0)
        elif op == "!=":
            eps = cfg.separation_tolerance
            holds = rel > eps
        else:
            if cfg.band_mode == "adaptive":
                band, _coverage = fit_band_auto(
                    op,
                    g.rho,
                    g.scale,
                    cfg.band_holdout_frac,
                    cfg.seed,
                    cap=cfg.tolerance,
                )
                eps = max(float(band.eps), 1e-9)
            else:
                eps = cfg.tolerance
            holds = (
                violation_magnitude(op, g.rho, g.scale)
                <= eps + 1e-15
            )

        k = int(np.count_nonzero(holds))
        z = z_for_alpha(cfg.ci_alpha)
        lo, hi, phat = wilson(k, int(holds.size), z=z)
        gain = mdl_gain(rule, eps, rel)
        strict = _strictness(op, eps, rel, holds)
        threshold = rule_threshold(
            cfg,
            op=op,
            strictness=strict,
            complexity=rule.complexity(),
            n_bindings=g.n_bindings,
            eps=eps,
        )
        accepted = lo >= threshold
        return Evaluation(
            rule=rule,
            accepted=accepted,
            reason=(
                "hold-rate above Wilson threshold"
                if accepted
                else "hold-rate Wilson lower bound below threshold"
            ),
            eps=eps,
            hold_rate=phat,
            hold_rate_lo=lo,
            hold_rate_hi=hi,
            statistic="hold_rate",
            support=g.support,
            n_points=g.n_points,
            n_bindings=g.n_bindings,
            mdl_gain=gain,
            strictness=strict,
            descriptor=(rule.binder, rule.length()),
            threshold=threshold,
        )

    def _reject(self, rule: A.Rule, reason: str) -> Evaluation:
        return Evaluation(
            rule=rule, accepted=False, reason=reason, eps=0.0,
            hold_rate=0.0, hold_rate_lo=0.0, hold_rate_hi=0.0, statistic="hold_rate",
            support=0.0, n_points=0, n_bindings=0, mdl_gain=0.0,
            strictness="reject", descriptor=(rule.binder, rule.length()))

    def _coefficient_for_points(self, coefficients, frame, nm, g, n_points: int):
        """Map each point of the FULL graded population to its group's fitted coefficient.

        ``_fit_proportional`` returns a coefficient per group aligned to the (possibly subsampled)
        scoring population.  The finite-arithmetic guard is a universal claim, so it needs the same
        mapping over every graded row.  When the data is ungrouped, or a group has no fitted
        coefficient, the largest fitted magnitude is used -- an upper bound, so the guard can only
        be conservative.
        """
        largest = max((abs(value) for value in coefficients.values()), default=0.0)
        rows = g.full_row_indices if g.full_row_indices is not None else g.row_indices
        labels = _group_labels(frame, nm, rows) if rows is not None else None
        if labels is None or labels.size != n_points:
            return np.full(n_points, largest, dtype=float)
        mapped = np.full(n_points, largest, dtype=float)
        for index, label in enumerate(labels.tolist()):
            typed = _typed_label(label)
            if typed in coefficients:
                mapped[index] = coefficients[typed]
        return mapped

    @staticmethod
    def _support_excluding(g, blown: int) -> float:
        """``g.support`` with ``blown`` further rows removed from the graded population.

        Tolerated overflow is still not evidence: the rows leave the scored population, so the
        reported support has to leave with them.
        """
        graded = int(g.graded_points)
        if graded <= 0:
            return g.support
        remaining = max(0, graded - int(blown))
        return float(g.support) * (float(remaining) / float(graded))

    def _graded_support_rejection(self, rule: A.Rule, g, blown: int):
        """Re-apply the conditioned support floor after overflowed rows are excluded, else ``None``."""
        if rule.condition is None:
            return None
        graded = max(0, int(g.graded_points) - int(blown))
        attempted = int(g.n_bindings) * int(self.ds.observed.n_rows)
        support = (float(graded) / float(attempted)) if attempted else 0.0
        if (
            graded < int(self.cfg.min_condition_points)
            or support < float(self.cfg.min_condition_fraction)
        ):
            return self._reject(
                rule,
                "condition support below minimum after overflow "
                f"({graded} points, {support:.3f} of rows)",
            )
        return None

    def _overflow_rejection(self, rule: A.Rule, g):
        """Refuse a candidate whose own arithmetic exceeded float64, else ``None``.

        An overflowed row is not missing data: the operands were finite and the candidate's
        expression blew up on them.  Dropping such rows would leave a rule graded on whatever
        survived while its reported support still described the full population -- a false
        discovery presented with full confidence -- so the candidate is refused outright with a
        reason that names the blow-up rather than being quietly scored on a shrunken population.
        """
        return self._overflow_reject_if(
            rule,
            overflow_points=int(getattr(g, "overflow_points", 0)),
            graded_points=int(getattr(g, "graded_points", 0)),
            fraction=float(getattr(g, "overflow_fraction", 0.0)),
        )

    def _overflow_reject_if(self, rule: A.Rule, *, overflow_points: int,
                            graded_points: int, fraction: float):
        if overflow_points <= 0 or fraction <= float(self.cfg.max_overflow_fraction):
            return None
        return self._reject(
            rule,
            "arithmetic overflowed float64 on "
            f"{overflow_points} of {overflow_points + graded_points} grounded rows "
            f"({fraction:.3f} of attempted rows)",
        )

    def _definition_overflow_rejection(self, rule: A.Rule):
        """Finite-arithmetic guard for a definition -> ``(rejection, tainted_rows)``.

        Definitions do not go through :func:`ground`; they mark a non-finite row invalid and score
        the survivors, which is the identical silent-shrink hazard the comparison path had. Every
        term a definition evaluates -- its target and each bound inside its predicate -- is checked,
        so a blown-up predicate cannot quietly narrow the population a definition claims to define.

        Overflow *within* ``max_overflow_fraction`` is tolerated but is still not evidence, so the
        tainted rows are returned for exclusion. Leaving them in is worse than for a comparison: a
        ``SUSTAINED`` predicate turns a tainted window into a confident ``False``, which then scores
        as a correct prediction and inflates both the agreement and the support.
        """
        terms = _definition_terms(rule.atom)
        if not terms:
            return None, None
        frame = self.ds.observed
        nm = self.ds.name_model
        condition = (
            _condition_mask(rule.condition, frame)
            if rule.condition is not None
            else None
        )
        selected = (
            np.ones(frame.n_rows, dtype=bool) if condition is None
            else np.asarray(condition, dtype=bool)
        )
        n_selected = int(np.count_nonzero(selected))
        overflow_points = 0
        graded_points = 0
        attempted = 0
        tainted: dict = {}
        for binding in enumerate_bindings(rule.binder, nm):
            binding_overflow = None
            grounded = True
            for term, window in terms:
                values, overflow = eval_term_overflow(term, rule.binder, binding, frame, nm)
                if values is None:
                    grounded = False
                    break
                if overflow is not None and window:
                    # A sustained predicate reads this term over a trailing window, so the taint
                    # spreads to every window the blown-up row belongs to.
                    overflow = _window_overflow(overflow, int(window), frame, nm)
                if overflow is not None:
                    binding_overflow = (
                        overflow if binding_overflow is None
                        else (binding_overflow | overflow)
                    )
            if not grounded:
                continue
            attempted += n_selected
            if binding_overflow is None:
                graded_points += n_selected
                continue
            blown = int(np.count_nonzero(binding_overflow & selected))
            overflow_points += blown
            graded_points += n_selected - blown
            # Keyed PER BINDING: a blow-up in one binding says nothing about another, and applying
            # one binding's mask to all of them both hides a different binding's genuine failures
            # and excludes rows it never blew up on.
            tainted[_binding_key(binding)] = binding_overflow & selected
        rejection = self._overflow_reject_if(
            rule,
            overflow_points=overflow_points,
            graded_points=graded_points,
            fraction=(float(overflow_points) / float(attempted)) if attempted else 0.0,
        )
        return rejection, (tainted or None)

    def _definition_failed_groups(self, tainted_rows):
        if not tainted_rows:
            return None
        source_groups = _group_labels(
            self.ds.observed,
            self.ds.name_model,
        )
        if source_groups is None:
            return None
        chunks = [
            source_groups[np.asarray(mask, dtype=bool)]
            for mask in tainted_rows.values()
            if np.any(mask)
        ]
        return np.concatenate(chunks) if chunks else None

    def _condition_support_rejection(self, rule: A.Rule):
        """Reject a conditioned rule whose condition selects too few rows, else ``None``.

        The same minimum-support floor that guards conditioned comparisons and bands: a rule that
        only speaks about a handful of rows is not evidence of a law regardless of how cleanly it
        separates them, so definitions must clear it too before any scoring happens.
        """
        if rule.condition is None:
            return None
        frame = self.ds.observed
        mask = _condition_mask(rule.condition, frame)
        if mask is None:
            return self._reject(rule, "definition condition could not be grounded")
        selected = int(np.count_nonzero(mask))
        support = float(selected) / max(1, frame.n_rows)
        if (
            selected < int(self.cfg.min_condition_points)
            or support < float(self.cfg.min_condition_fraction)
        ):
            return self._reject(
                rule,
                "condition support below minimum "
                f"({selected} points, {support:.3f} of rows)",
            )
        return None

    def _graded_condition_rejection(self, rule: A.Rule, valid: np.ndarray):
        """Re-apply the condition floor to the rows a definition can actually grade, else ``None``.

        ``_condition_support_rejection`` runs before grounding, so it can only count the rows the
        condition selects. A definition grades a narrower population than that: rolling windows,
        non-finite targets and unevaluable predicate operands all shrink the validity mask. Without
        this second check a condition that nominally selects enough rows can still be scored on a
        handful of them, which is exactly the thin evidence the floor exists to reject.
        """
        if rule.condition is None:
            return None
        graded = np.asarray(valid, dtype=bool)
        if graded.size == 0:
            return None
        selected = int(np.count_nonzero(graded))
        support = float(selected) / float(graded.size)
        if (
            selected < int(self.cfg.min_condition_points)
            or support < float(self.cfg.min_condition_fraction)
        ):
            return self._reject(
                rule,
                "condition support below minimum after grounding "
                f"({selected} points, {support:.3f} of rows)",
            )
        return None

    def _evaluate_boolean_definition(self, rule: A.Rule, tainted_rows=None) -> Evaluation:
        atom = rule.atom
        learned = _learned_bounds(atom.predicate)
        if len(learned) > 2:
            return self._reject(rule, "Boolean definition has too many learned thresholds")
        window_by_bound = dict(_learned_bound_contexts(atom.predicate))
        target, valid, groups, effective, n_bindings = _definition_grounding(
            rule,
            learned,
            window_by_bound,
            self.ds,
            tainted_rows=tainted_rows,
        )
        if target.size == 0:
            return self._reject(rule, "Boolean definition grounded no valid points")
        thin = self._graded_condition_rejection(rule, valid)
        if thin is not None:
            return thin
        # Split the gradeable points once on the threshold-independent validity mask, then draw
        # candidate thresholds from the fit rows only. This keeps the evaluation split's labels out
        # of threshold fitting (holdout discipline) and makes the fit/score split identical.
        fit_mask, evaluation_mask = _parameter_masks(
            valid,
            self.cfg,
            split=bool(learned),
            groups=groups,
        )
        candidate_sets = [
            _fit_threshold_candidates(
                effective[bound],
                fit_mask,
                self.cfg,
                descriptor=bound.unparse(),
            )
            for bound in learned
        ]
        if any(not candidates for candidates in candidate_sets):
            return self._reject(rule, "Boolean threshold could not be fit")
        assignments = [{}]
        if learned:
            import itertools

            assignments = [
                dict(zip(learned, values))
                for values in itertools.product(*candidate_sets)
            ]
        best = None
        for thresholds in assignments:
            _t, predicted, _v, _n, _g = _boolean_population(
                rule,
                self.ds,
                thresholds,
            )
            if predicted.size != target.size:
                continue
            score = (
                float(np.mean(target[fit_mask] == predicted[fit_mask]))
                if np.any(fit_mask) else 0.0
            )
            preference = (score, -sum(abs(value) for value in thresholds.values()))
            if best is None or preference > best[0]:
                best = (preference, thresholds, predicted)
        if best is None:
            return self._reject(rule, "Boolean definition grounded no valid points")
        _, thresholds, predicted = best
        evaluation_target = target[evaluation_mask]
        holds = evaluation_target == predicted[evaluation_mask]
        positive_rate = float(np.mean(evaluation_target)) if evaluation_target.size else 0.0
        baseline = max(positive_rate, 1.0 - positive_rate)
        return self._definition_evaluation(
            rule,
            holds,
            int(target.size),
            n_bindings,
            parameters={
                "thresholds": {
                    bound.unparse(): float(value)
                    for bound, value in thresholds.items()
                }
            },
            baseline=baseline,
            groups=(
                groups[evaluation_mask]
                if groups is not None
                else None
            ),
            failed_groups=self._definition_failed_groups(
                tainted_rows,
            ),
        )

    def _evaluate_category_definition(self, rule: A.Rule) -> Evaluation:
        frame = self.ds.observed
        atom = rule.atom
        if atom.target_column not in frame.row_context:
            return self._reject(rule, "categorical target is absent from row context")
        target = np.asarray(frame.row_context[atom.target_column], dtype=object)
        valid = ~pd.isna(target)
        if rule.condition is not None:
            condition = _condition_mask(rule.condition, frame)
            valid &= condition if condition is not None else False
        case_masks = []
        for column, value in atom.cases:
            if column not in frame.row_context:
                return self._reject(rule, f"categorical case column {column!r} is absent")
            raw_active = np.asarray(
                frame.row_context[column],
                dtype=object,
            )
            present = ~pd.isna(raw_active)
            active = np.zeros(target.size, dtype=bool)
            active[present] = raw_active[present].astype(bool)
            valid &= present
            case_masks.append((active, value))
        thin = self._graded_condition_rejection(rule, valid)
        if thin is not None:
            return thin
        missing_edges = []
        for left_index, (left_mask, left_value) in enumerate(case_masks):
            higher_active = np.zeros(target.size, dtype=bool)
            for higher_mask, _higher_value in case_masks[:left_index]:
                higher_active |= higher_mask
            for right_index, (right_mask, right_value) in enumerate(
                case_masks[left_index + 1:],
                start=left_index + 1,
            ):
                identifiable = (
                    _typed_label(left_value) == _typed_label(right_value)
                    or np.any(
                        left_mask
                        & right_mask
                        & ~higher_active
                        & valid
                    )
                )
                if not identifiable:
                    missing_edges.append((left_index, right_index))
        if missing_edges:
            return self._reject(
                rule,
                "categorical priority is not identifiable for every precedence edge",
            )
        predicted = np.full(target.size, atom.default, dtype=object)
        for active, value in reversed(case_masks):
            predicted[active] = value
        category_groups = _group_labels(
            frame,
            self.ds.name_model,
        )
        return self._definition_evaluation(
            rule,
            _typed_equal_array(target[valid], predicted[valid]),
            int(target.size),
            1,
            parameters={},
            baseline=max(
                float(np.mean(_typed_scalar_mask(target[valid], value)))
                for value in (
                    _untyped_label(key)
                    for key in dict.fromkeys(
                        _typed_label(item)
                        for item in target[valid].tolist()
                    )
                )
            ) if np.any(valid) else 1.0,
            groups=(
                category_groups[valid]
                if category_groups is not None
                else None
            ),
        )

    def _definition_evaluation(
        self,
        rule: A.Rule,
        holds: np.ndarray,
        population_size: int,
        n_bindings: int,
        *,
        parameters: dict,
        baseline: float,
        groups=None,
        failed_groups=None,
    ) -> Evaluation:
        if holds.size == 0:
            return self._reject(rule, "definition grounded no valid points")
        k = int(np.count_nonzero(holds))
        z = z_for_alpha(self.cfg.ci_alpha)
        lo, hi, phat = wilson(k, int(holds.size), z=z)
        strict = "definition"
        threshold = rule_threshold(
            self.cfg,
            op=":=",
            strictness=strict,
            complexity=rule.complexity(),
            n_bindings=n_bindings,
            eps=0.0,
        )
        required = max(
            threshold,
            float(baseline)
            + float(self.cfg.definition_min_lift)
            * (1.0 - float(baseline)),
        )
        accepted = lo >= required
        group_ok, group_parameters = _group_hold_gate(
            holds,
            groups,
            z=z,
            threshold=required,
            failed_groups=failed_groups,
        )
        accepted &= group_ok
        residual = np.where(holds, 0.0, 1.0)
        gain = mdl_gain(rule, 1e-9, residual)
        support = holds.size / max(1, population_size)
        return Evaluation(
            rule=rule,
            accepted=accepted,
            reason=(
                "definition agreement above Wilson threshold and baseline"
                if accepted
                else "definition agreement Wilson lower bound below threshold or baseline lift"
            ),
            eps=0.0,
            hold_rate=phat,
            hold_rate_lo=lo,
            hold_rate_hi=hi,
            statistic="hold_rate",
            support=support,
            n_points=int(holds.size),
            n_bindings=n_bindings,
            mdl_gain=gain,
            strictness=strict,
            descriptor=(rule.binder, rule.length()),
            threshold=required,
            parameters={
                **parameters,
                **group_parameters,
                "baseline_agreement": float(baseline),
            },
        )

    def _evaluate_band_definition(self, rule: A.Rule, tainted_rows=None) -> Evaluation:
        values = []
        groups = []
        source_groups = _group_labels(
            self.ds.observed,
            self.ds.name_model,
        )
        failed_groups = self._definition_failed_groups(tainted_rows)
        condition_mask = None
        if rule.condition is not None:
            condition_mask = _condition_mask(
                rule.condition,
                self.ds.observed,
            )
            if condition_mask is None:
                return self._reject(rule, "band condition could not be grounded")
        n_candidates = 0
        for binding in enumerate_bindings(rule.binder, self.ds.name_model):
            n_candidates += 1
            vector = eval_term(
                rule.atom.term,
                rule.binder,
                binding,
                self.ds.observed,
                self.ds.name_model,
            )
            if vector is None:
                continue
            mask = np.isfinite(vector)
            if condition_mask is not None:
                mask &= condition_mask
            binding_taint = (
                None if tainted_rows is None
                else tainted_rows.get(_binding_key(binding))
            )
            if binding_taint is not None:
                mask &= ~np.asarray(binding_taint, dtype=bool)
            values.append(np.asarray(vector, dtype=float)[mask])
            if source_groups is not None:
                groups.append(source_groups[mask])
        if not values:
            return self._reject(rule, "band grounded no valid points")
        population = np.concatenate(values)
        population_groups = (
            np.concatenate(groups)
            if groups
            else None
        )
        # Reported support must describe the rows the band was SCORED on, exactly as
        # ``Grounded.support`` does for comparisons: a band graded on ten finite rows out of a
        # hundred is not supported by the whole hundred, and neither is one that grounded on half
        # its bindings. Measured unconditionally, not only under a condition, because a non-finite
        # term shrinks the evidence either way.
        n_bindings = len(values)
        graded_rows = float(population.size) / float(
            max(1, n_bindings * self.ds.observed.n_rows)
        )
        graded_support = graded_rows * (
            float(n_bindings) / float(n_candidates) if n_candidates else 0.0
        )
        if rule.condition is not None:
            # Measure the floor on the rows the band can actually grade, not on the rows the
            # condition merely selects: non-finite terms shrink the evidence further.
            if (
                population.size < int(self.cfg.min_condition_points)
                or graded_rows < float(self.cfg.min_condition_fraction)
            ):
                return self._reject(
                    rule,
                    "condition support below minimum "
                    f"({population.size} points, {graded_rows:.3f} of rows)",
                )
        valid = np.ones(population.size, dtype=bool)
        learned = rule.atom.center is None
        fit_mask, evaluation_mask = _parameter_masks(
            valid,
            self.cfg,
            split=learned,
            groups=population_groups,
        )
        if not np.any(evaluation_mask):
            return self._reject(rule, "band center could not be fit")
        center = (
            robust_median(population[fit_mask])
            if learned
            else float(rule.atom.center)
        )
        # The centre introduces arithmetic the grounding pass never saw, exactly as a fitted
        # proportional coefficient does: ``value - centre`` overflows for a centre and an
        # observation at opposite ends of the float64 range. Checked over the WHOLE graded
        # population, refused above the cap, and -- when tolerated -- the blown-up rows leave the
        # scored population and the reported support with it.
        with np.errstate(over="ignore", invalid="ignore"):
            centred_all = population - center
        centre_overflow = _blowup(
            centred_all, population, np.full(population.shape, center),
        )
        if centre_overflow is not None:
            blown = int(np.count_nonzero(centre_overflow))
            # Measured against the rows the band was OFFERED under its condition, exactly as the
            # base guard's `attempted_points` is -- using every frame row instead would divide a
            # conditioned band's overflow by a population it never claimed.
            selected_rows = (
                self.ds.observed.n_rows if condition_mask is None
                else int(np.count_nonzero(condition_mask))
            )
            attempted = max(1, n_bindings * selected_rows)
            rejection = self._overflow_reject_if(
                rule,
                overflow_points=blown,
                graded_points=max(0, int(population.size) - blown),
                fraction=float(blown) / float(attempted),
            )
            if rejection is not None:
                return rejection
            if population_groups is not None:
                centre_failed_groups = population_groups[
                    centre_overflow
                ]
                failed_groups = (
                    centre_failed_groups
                    if failed_groups is None
                    else np.concatenate((
                        failed_groups,
                        centre_failed_groups,
                    ))
                )
            evaluation_mask = evaluation_mask & ~centre_overflow
            if not np.any(evaluation_mask):
                return self._reject(
                    rule, "band centre overflowed on every evaluated row",
                )
            graded_points = int(population.size) - blown
            graded_rows = float(graded_points) / float(
                max(1, n_bindings * self.ds.observed.n_rows)
            )
            graded_support = graded_rows * (
                float(n_bindings) / float(n_candidates) if n_candidates else 0.0
            )
            # A tolerated overflow still shrinks the evidence, so a conditioned band has to clear
            # the support floor again on what is left of it.
            if rule.condition is not None and (
                graded_points < int(self.cfg.min_condition_points)
                or graded_rows < float(self.cfg.min_condition_fraction)
            ):
                return self._reject(
                    rule,
                    "condition support below minimum after overflow "
                    f"({graded_points} points, {graded_rows:.3f} of rows)",
                )
        observed = population[evaluation_mask]
        scale = np.maximum(np.abs(observed), abs(center))
        positive = scale[scale > 0]
        floor = _positive_scale_floor(positive)
        scale = np.maximum(scale, floor)
        with np.errstate(over="ignore", invalid="ignore"):
            relative = np.abs(observed - center) / scale
        holds = relative <= float(self.cfg.tolerance) + 1e-15
        k = int(np.count_nonzero(holds))
        lo, hi, phat = wilson(k, int(holds.size), z=z_for_alpha(self.cfg.ci_alpha))
        threshold = rule_threshold(
            self.cfg,
            op="~band",
            strictness="band",
            complexity=rule.complexity(),
            n_bindings=n_bindings,
            eps=float(self.cfg.tolerance),
        )
        accepted = lo >= threshold
        group_ok, group_parameters = _group_hold_gate(
            holds,
            (
                population_groups[evaluation_mask]
                if population_groups is not None
                else None
            ),
            z=z_for_alpha(self.cfg.ci_alpha),
            threshold=threshold,
            failed_groups=failed_groups,
        )
        accepted &= group_ok
        return Evaluation(
            rule=rule,
            accepted=accepted,
            reason=(
                "band hold-rate above Wilson threshold"
                if accepted
                else "band hold-rate Wilson lower bound below threshold"
            ),
            eps=float(self.cfg.tolerance),
            hold_rate=phat,
            hold_rate_lo=lo,
            hold_rate_hi=hi,
            statistic="hold_rate",
            support=graded_support,
            n_points=int(holds.size),
            n_bindings=n_bindings,
            mdl_gain=mdl_gain(rule, float(self.cfg.tolerance), relative),
            strictness="band",
            descriptor=(rule.binder, rule.length()),
            threshold=threshold,
            parameters={
                "center": center,
                **group_parameters,
            },
        )


def _fit_proportional(g, frame, nm, cfg):
    """Fit a robust through-origin slope globally or once per declared row group."""

    keys = tuple(getattr(getattr(nm, "adapter", None), "group_keys", ()))
    if keys and all(key in frame.row_context for key in keys):
        if len(keys) == 1:
            labels = _typed_object_array(
                frame.row_context[keys[0]]
            )[g.row_indices]
        else:
            columns = [
                _typed_object_array(frame.row_context[key])[g.row_indices]
                for key in keys
            ]
            labels = np.empty(g.n_points, dtype=object)
            for index, values in enumerate(zip(*columns)):
                labels[index] = tuple(values)
    else:
        labels = np.full(g.n_points, GLOBAL_GROUP, dtype=object)

    coefficients: dict[str, float] = {}
    coefficient_by_point = np.full(g.n_points, np.nan, dtype=float)
    evaluation_mask = np.zeros(g.n_points, dtype=bool)
    for group_index, typed in enumerate(
        dict.fromkeys(_typed_label(item) for item in labels.tolist())
    ):
        label = typed[1]
        mask = np.asarray(
            [_typed_label(item) == typed for item in labels], dtype=bool,
        )
        left = g.left[mask]
        right = g.right[mask]
        finite = np.isfinite(left) & np.isfinite(right)
        usable = finite & (right != 0.0)
        local_positions = np.flatnonzero(mask)
        usable_positions = local_positions[usable]
        zero_predictor_positions = local_positions[
            finite & ~usable
        ]
        if usable_positions.size < int(cfg.min_proportional_points):
            if usable_positions.size:
                return None
            coefficient_by_point[zero_predictor_positions] = 0.0
            evaluation_mask[zero_predictor_positions] = True
            continue
        rng = np.random.default_rng(int(cfg.seed) + group_index)
        shuffled = rng.permutation(usable_positions)
        n_eval = max(1, int(round(float(cfg.parameter_holdout_frac) * shuffled.size)))
        n_eval = min(n_eval, shuffled.size - 2)
        if n_eval < 1:
            continue
        eval_positions = shuffled[:n_eval]
        fit_positions = shuffled[n_eval:]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratios = g.left[fit_positions] / g.right[fit_positions]
        # ``np.median`` averages the two central ratios, and that intermediate sum overflows for a
        # sample near the float64 ceiling -- which would discard a perfectly well-determined
        # coefficient and fall through to the least-squares branch (a different estimator) for a
        # purely numerical reason. ``robust_median`` reproduces the median exactly whenever the
        # ratios are finite, so the least-squares fallback is reached only when they genuinely are
        # not.
        coefficient = robust_median(ratios) if np.all(np.isfinite(ratios)) else float("nan")
        if not np.isfinite(coefficient):
            fit_right = g.right[fit_positions]
            fit_left = g.left[fit_positions]
            denom = float(np.dot(fit_right, fit_right))
            if denom <= 0.0:
                continue
            coefficient = float(np.dot(fit_right, fit_left) / denom)
        # Keyed by the RAW group label, not its string form: distinct groups whose labels stringify
        # identically (``1`` and ``"1"``) would otherwise collide, one group's coefficient would
        # silently replace the other's, and the finite-arithmetic guard would then check the wrong
        # coefficient. Stringification happens only when the parameters are reported.
        coefficients[typed] = coefficient
        coefficient_by_point[mask] = coefficient
        evaluation_mask[eval_positions] = True
        evaluation_mask[zero_predictor_positions] = True
    if not coefficients or not np.any(evaluation_mask):
        return None
    return (
        coefficients,
        coefficient_by_point,
        evaluation_mask,
        labels[evaluation_mask],
    )


def _reported_coefficients(coefficients: dict) -> dict:
    """Serialisable view of the per-group coefficients (typed labels -> collision-free keys)."""
    display = _display_keys(coefficients)
    reported = {
        ("global" if isinstance(typed[1], _GlobalGroup) else display[typed]): value
        for typed, value in coefficients.items()
    }
    assert len(reported) == len(coefficients)
    return reported


def _predicate_terms(predicate: A.Predicate, window=None) -> list[tuple]:
    """Every numeric term a Boolean predicate evaluates, paired with the window it is read under.

    A ``SUSTAINED`` predicate reads its term over a trailing window, so ONE blown-up row taints
    every window that contains it. Reporting the raw row count would undercount the population the
    definition actually fails to describe, which is how a mathematically wrong definition passed a
    1% overflow cap on a single bad row.
    """
    if isinstance(predicate, A.Bound):
        return [(predicate.term, window)]
    if isinstance(predicate, A.Sustained):
        return _predicate_terms(predicate.predicate, int(predicate.window))
    if isinstance(predicate, A.Conjunction):
        out: list[tuple] = []
        for item in predicate.predicates:
            out.extend(_predicate_terms(item, window))
        return list(dict.fromkeys(out))
    return []


def _definition_terms(atom) -> list[tuple]:
    """Every ``(term, window)`` a definition evaluates, for the finite-arithmetic guard.

    A category definition reads Boolean context columns rather than evaluating arithmetic, so it
    contributes no terms and can never overflow.
    """
    if isinstance(atom, A.BooleanDefinition):
        return list(dict.fromkeys([
            (atom.target, None),
            *_predicate_terms(atom.predicate),
        ]))
    if isinstance(atom, A.BandDefinition):
        return [(atom.term, None)]
    return []


def _learned_bounds(predicate: A.Predicate) -> list[A.Bound]:
    if isinstance(predicate, A.Bound):
        return [predicate] if predicate.threshold is None else []
    if isinstance(predicate, A.Sustained):
        return _learned_bounds(predicate.predicate)
    if isinstance(predicate, A.Conjunction):
        out: list[A.Bound] = []
        for item in predicate.predicates:
            out.extend(_learned_bounds(item))
        return list(dict.fromkeys(out))
    return []


def _learned_bound_contexts(predicate: A.Predicate, window=None) -> list[tuple]:
    """Each learned bound paired with the sustained window it is evaluated under (or ``None``).

    A learned bound inside ``ALWAYS_w(...)`` is not a stump on its raw term: the sustained output
    is ``all(term OP theta)`` over the window, which for ``<``/``<=`` equals ``rolling_max < theta``
    and for ``>``/``>=`` equals ``rolling_min > theta``. Carrying the window lets candidate
    generation localise on that effective series so the exact optimum is found.
    """
    if isinstance(predicate, A.Bound):
        return [(predicate, window)] if predicate.threshold is None else []
    if isinstance(predicate, A.Sustained):
        return _learned_bound_contexts(predicate.predicate, predicate.window)
    if isinstance(predicate, A.Conjunction):
        out: list[tuple] = []
        for item in predicate.predicates:
            out.extend(_learned_bound_contexts(item, window))
        return out
    return []


def _bound_effective_series(bound: A.Bound, window, binder, binding, dataset):
    """Ground the series a learned bound's threshold actually separates.

    For a direct bound this is the term itself; for a sustained bound it is the per-group rolling
    max (``<``/``<=``) or rolling min (``>``/``>=``) aligned to the same valid window-end rows the
    sustained predicate uses, so a stump on it reproduces the sustained decision exactly.
    """
    frame = dataset.observed
    values = eval_term(bound.term, binder, binding, frame, dataset.name_model)
    if values is None:
        return None, None
    values = np.asarray(values, dtype=float)
    if not window:
        return values, np.isfinite(values)
    reducer = np.max if bound.op in ("<", "<=") else np.min
    groups = _ordered_groups(frame, dataset.name_model)
    if groups is None:
        return None, None
    effective = np.full(frame.n_rows, np.nan, dtype=float)
    valid = np.zeros(frame.n_rows, dtype=bool)
    for rows in groups:
        consecutive = _consecutive_window_ends(
            frame,
            dataset.name_model,
            rows,
            int(window),
        )
        for end in range(int(window) - 1, rows.size):
            if not consecutive[end]:
                continue
            window_rows = rows[end - int(window) + 1:end + 1]
            window_values = values[window_rows]
            if np.all(np.isfinite(window_values)):
                effective[rows[end]] = float(reducer(window_values))
                valid[rows[end]] = True
    return effective, valid


def _definition_grounding(rule: A.Rule, learned, window_by_bound, dataset, tainted_rows=None):
    """Single aligned pass over the definition population.

    Returns the concatenated Boolean target, the threshold-independent validity mask, the optional
    group labels, and each learned bound's *effective* series (the term itself, or its rolling
    window reduction under a sustained wrapper). Grounding everything in one binding loop keeps the
    per-bound series row-aligned with the target/valid arrays that ``_boolean_population`` produces,
    so the fit/evaluation split can be computed once and reused for candidate generation and
    scoring without leaking evaluation-split labels into the fit.
    """
    frame = dataset.observed
    zero = {bound: 0.0 for bound in learned}
    condition = (
        _condition_mask(rule.condition, frame)
        if rule.condition is not None
        else None
    )
    source_groups = _group_labels(frame, dataset.name_model)
    targets: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    effective: dict = {bound: [] for bound in learned}
    n_bindings = 0
    for binding in enumerate_bindings(rule.binder, dataset.name_model):
        target = eval_term(
            rule.atom.target,
            rule.binder,
            binding,
            frame,
            dataset.name_model,
        )
        if target is None:
            continue
        predicted, valid = _predicate_population(
            rule.atom.predicate,
            rule.binder,
            binding,
            dataset,
            zero,
        )
        if predicted is None:
            continue
        if condition is not None:
            valid = valid & (condition if condition is not None else False)
        # Rows this binding's own arithmetic blew up on are not evidence, even when the blow-up was
        # within the tolerated fraction: a sustained predicate would turn each one into a confident
        # ``False`` that then scores as a correct prediction.
        binding_taint = (
            None if tainted_rows is None
            else tainted_rows.get(_binding_key(binding))
        )
        if binding_taint is not None:
            valid = valid & ~np.asarray(binding_taint, dtype=bool)
        targets.append(np.asarray(target, dtype=float) != 0.0)
        valids.append(valid & np.isfinite(np.asarray(target, dtype=float)))
        if source_groups is not None:
            groups.append(source_groups.copy())
        for bound in learned:
            series, _series_valid = _bound_effective_series(
                bound,
                window_by_bound.get(bound),
                rule.binder,
                binding,
                dataset,
            )
            if series is None:
                series = np.full(frame.n_rows, np.nan, dtype=float)
            effective[bound].append(np.asarray(series, dtype=float))
        n_bindings += 1
    if not targets:
        empty = np.empty(0, dtype=bool)
        return empty, empty, None, {}, 0
    return (
        np.concatenate(targets),
        np.concatenate(valids),
        np.concatenate(groups) if groups else None,
        {bound: np.concatenate(chunks) for bound, chunks in effective.items()},
        n_bindings,
    )


def _fit_threshold_candidates(effective: np.ndarray, fit_mask: np.ndarray, cfg=None,
                              descriptor: str = "") -> list[float]:
    """Exhaustive, holdout-clean candidate thresholds for one learned bound.

    Candidates are drawn from the *fit* rows only (never the evaluation split) as the midpoints
    between consecutive distinct effective-term values plus two edge sentinels (below the minimum
    and above the maximum). When adjacent floats have no representable interior midpoint, both
    observed endpoints are included so strict and non-strict operators remain separable. Because a
    stump's prediction only changes at these boundaries, this discrete set contains the exact
    agreement-maximising threshold, and the Cartesian product across a conjunction's learned bounds
    therefore contains the exact *joint* optimum -- not merely a per-bound heuristic. A positive
    ``max_threshold_candidates`` fails loud rather than silently coarsening.
    """
    mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(effective)
    terms = effective[mask]
    if terms.size == 0:
        return []
    unique = np.unique(terms)
    if unique.size == 1:
        base = unique.astype(float)
        span = max(abs(float(unique[0])), 1.0)
    else:
        lo_vals = unique[:-1]
        hi_vals = unique[1:]
        # Overflow-safe midpoints between consecutive distinct values. ``a + (b - a)/2`` is exact for
        # same-sign values near the float maximum, but ``b - a`` still overflows to +/-inf for
        # *opposite-sign* extremes (e.g. -MAX vs +MAX). Where that happens, fall back to the half-sum
        # ``a/2 + b/2``, which cannot overflow (each half has magnitude <= MAX) and stays within
        # ``[a, b]``, so a finite separating threshold always exists between two finite values.
        with np.errstate(over="ignore"):
            base = lo_vals + (hi_vals - lo_vals) / 2.0
        overflow = ~np.isfinite(base)
        if overflow.any():
            base = np.where(overflow, lo_vals * 0.5 + hi_vals * 0.5, base)
        collapsed = (base <= lo_vals) | (base >= hi_vals)
        separators = np.concatenate((
            lo_vals[collapsed],
            hi_vals[collapsed],
        ))
        with np.errstate(over="ignore"):
            raw_span = float(hi_vals[-1] - lo_vals[0])
        span = raw_span if np.isfinite(raw_span) else 0.0
    # Edge sentinels strictly outside the observed range so an all-true / all-false stump is
    # reachable. ``nextafter`` handles the ordinary case; the span-scaled offset widens them. Both are
    # forced finite -- at the absolute float extremes ``nextafter`` and the offset run off to +/-inf,
    # and an infinite candidate is neither a usable threshold nor JSON-serialisable, so we clamp back
    # to the observed endpoint (a finite, degenerate all/none boundary).
    with np.errstate(over="ignore"):
        low = np.nextafter(unique[0], -np.inf)
        high = np.nextafter(unique[-1], np.inf)
        if span > 0:
            low = min(low, unique[0] - span * 1e-6)
            high = max(high, unique[-1] + span * 1e-6)
    if not np.isfinite(low):
        low = float(unique[0])
    if not np.isfinite(high):
        high = float(unique[-1])
    edges = np.array([low, high], dtype=float)
    candidates = np.unique(np.concatenate([
        base,
        edges,
        separators if unique.size > 1 else np.empty(0, dtype=float),
    ]))
    cap = int(getattr(cfg, "max_threshold_candidates", 0) or 0)
    if cap > 0 and candidates.size > cap:
        from .propose import SearchSpaceTruncatedError

        raise SearchSpaceTruncatedError(
            f"learned-threshold bound {descriptor!r} exposes {candidates.size} fit-split candidate "
            f"thresholds, exceeding max_threshold_candidates={cap}; raise the ceiling or tighten "
            "the declared grammar"
        )
    return [float(value) for value in candidates]


def _parameter_masks(
    valid: np.ndarray,
    cfg,
    *,
    split: bool,
    groups=None,
):
    valid = np.asarray(valid, dtype=bool)
    if not split:
        return valid.copy(), valid.copy()
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if groups.shape != valid.shape:
            empty = np.zeros(valid.size, dtype=bool)
            return empty, empty
        fit = np.zeros(valid.size, dtype=bool)
        evaluation = np.zeros(valid.size, dtype=bool)
        # Typed identity again: a raw split can merge two groups and leave one of them entirely out
        # of the evaluation half, where it can no longer fail the per-group gate.
        typed_groups = [_typed_label(item) for item in groups.tolist()]
        typed_array = np.empty(len(typed_groups), dtype=object)
        typed_array[:] = typed_groups
        for group_index, label in enumerate(
            dict.fromkeys(typed_array[valid].tolist())
        ):
            indices = np.flatnonzero(
                valid
                & np.asarray([
                    item == label
                    for item in typed_groups
                ], dtype=bool)
            )
            if indices.size == 1:
                evaluation[indices[0]] = True
                continue
            rng = np.random.default_rng(
                int(cfg.seed) + group_index
            )
            shuffled = rng.permutation(indices)
            n_eval = max(
                1,
                int(round(
                    float(cfg.parameter_holdout_frac)
                    * indices.size
                )),
            )
            n_eval = min(n_eval, indices.size - 1)
            evaluation[shuffled[:n_eval]] = True
            fit[shuffled[n_eval:]] = True
        if not np.any(fit) or not np.any(evaluation):
            empty = np.zeros(valid.size, dtype=bool)
            return empty, empty
        return fit, evaluation
    indices = np.flatnonzero(valid)
    if indices.size < 4:
        empty = np.zeros(valid.size, dtype=bool)
        return empty, empty
    rng = np.random.default_rng(int(cfg.seed))
    shuffled = rng.permutation(indices)
    n_eval = max(1, int(round(float(cfg.parameter_holdout_frac) * indices.size)))
    n_eval = min(n_eval, indices.size - 2)
    fit = np.zeros(valid.size, dtype=bool)
    evaluation = np.zeros(valid.size, dtype=bool)
    evaluation[shuffled[:n_eval]] = True
    fit[shuffled[n_eval:]] = True
    return fit, evaluation


def _boolean_population(rule: A.Rule, dataset, thresholds):
    targets = []
    predictions = []
    valid_masks = []
    groups = []
    source_groups = _group_labels(
        dataset.observed,
        dataset.name_model,
    )
    n_bindings = 0
    for binding in enumerate_bindings(rule.binder, dataset.name_model):
        target = eval_term(
            rule.atom.target,
            rule.binder,
            binding,
            dataset.observed,
            dataset.name_model,
        )
        if target is None:
            continue
        predicted, valid = _predicate_population(
            rule.atom.predicate,
            rule.binder,
            binding,
            dataset,
            thresholds,
        )
        if predicted is None:
            continue
        if rule.condition is not None:
            condition = _condition_mask(rule.condition, dataset.observed)
            valid &= condition if condition is not None else False
        targets.append(np.asarray(target, dtype=float) != 0.0)
        predictions.append(predicted)
        valid_masks.append(valid & np.isfinite(target))
        if source_groups is not None:
            groups.append(source_groups.copy())
        n_bindings += 1
    if not targets:
        empty = np.empty(0, dtype=bool)
        return empty, empty, empty, 0, None
    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        np.concatenate(valid_masks),
        n_bindings,
        np.concatenate(groups) if groups else None,
    )


def _predicate_population(predicate, binder, binding, dataset, thresholds):
    frame = dataset.observed
    if isinstance(predicate, A.Bound):
        values = eval_term(predicate.term, binder, binding, frame, dataset.name_model)
        if values is None:
            return None, None
        values = np.asarray(values, dtype=float)
        threshold = (
            thresholds[predicate]
            if predicate.threshold is None
            else float(predicate.threshold)
        )
        valid = np.isfinite(values)
        if predicate.op == "<":
            return values < threshold, valid
        if predicate.op == "<=":
            return values <= threshold, valid
        if predicate.op == ">":
            return values > threshold, valid
        if predicate.op == ">=":
            return values >= threshold, valid
        return None, None
    if isinstance(predicate, A.Sustained):
        base, base_valid = _predicate_population(
            predicate.predicate,
            binder,
            binding,
            dataset,
            thresholds,
        )
        if base is None:
            return None, None
        groups = _ordered_groups(frame, dataset.name_model)
        if groups is None:
            return None, None
        output = np.zeros(frame.n_rows, dtype=bool)
        valid = np.zeros(frame.n_rows, dtype=bool)
        for rows in groups:
            consecutive = _consecutive_window_ends(
                frame,
                dataset.name_model,
                rows,
                predicate.window,
            )
            for end in range(rows.size):
                row = rows[end]
                if end < predicate.window - 1 or not consecutive[end]:
                    # No run of `window` consecutive periods ends here -- either the group has
                    # not produced that many periods yet, or a cadence gap resets the run. The
                    # sustained predicate is then definitively unmet, not ungradeable, so the
                    # row must be scored as False rather than dropped from the population. If
                    # such rows were excluded, a definition whose target is wrongly True during
                    # warm-up or across a gap would never be charged for those mistakes.
                    valid[row] = True
                    output[row] = False
                    continue
                window_rows = rows[end - predicate.window + 1:end + 1]
                # A sustained run is a POSITIVE claim: the predicate must be observed to hold on
                # every period of the window. A period we could not evaluate does not establish
                # the run any more than a missing period does, so it breaks the run rather than
                # making the row undecidable. Together with the warm-up and gap cases above this
                # makes SUSTAINED total, which is what lets a run-length rule be compared against
                # its planted spans on every row.
                valid[row] = True
                output[row] = bool(
                    np.all(base_valid[window_rows] & base[window_rows])
                )
        return output, valid
    if isinstance(predicate, A.Conjunction):
        # Three-valued conjunction. A row is decided when every conjunct could be evaluated, OR
        # when some conjunct we *could* evaluate is already False -- an unevaluable conjunct can
        # never rescue a conjunction that is broken elsewhere. Only when nothing observable settles
        # the row does it stay ungraded. Dropping every row with a missing operand (the previous
        # behaviour) excused the rule on exactly the rows where a wrong target hides.
        #
        # The two alternatives were measured against the emitted GTIB trajectory alert, whose
        # operands need up to 45 periods of history:
        #   * drop the row whenever any operand is missing -- grades 1501/2160 rows at agreement
        #     1.000, hiding rows where the target fires;
        #   * this rule -- grades 1542/2160 at agreement 1.000;
        #   * treat a missing conjunct as vacuously true -- grades 1861/2160 but agreement falls to
        #     0.954, because it invents a verdict the data does not support.
        # So this is the most total reading that stays exact. Note the contrast with `Sustained`
        # above: there a missing period definitively falsifies the operator's own claim that a run
        # of observations occurred, so it is graded False rather than left undecided.
        #
        # Only a conjunct with NO learned threshold may settle a row. Gradability has to be
        # independent of the fitted parameters: if a learned bound could decide which rows are in
        # the population, the population -- and with it the fit/evaluation split -- would move as
        # the threshold moves, and `_definition_grounding`'s threshold-independent validity mask
        # would no longer describe the rows actually scored.
        output = np.ones(frame.n_rows, dtype=bool)
        all_observed = np.ones(frame.n_rows, dtype=bool)
        known_false = np.zeros(frame.n_rows, dtype=bool)
        for item in predicate.predicates:
            values, item_valid = _predicate_population(
                item,
                binder,
                binding,
                dataset,
                thresholds,
            )
            if values is None:
                return None, None
            output &= values
            all_observed &= item_valid
            if not _learned_bounds(item):
                known_false |= item_valid & ~values
        return output & ~known_false, all_observed | known_false
    return None, None
