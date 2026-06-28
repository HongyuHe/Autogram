"""Data-only quality metrics for a candidate invariant.

These are the pieces the discovery evaluator composes, none of which reads a clean oracle or a
catalogue:

* **confidence / support** -- :func:`wilson` (held-out coverage interval) and a separate
  support fraction computed upstream.
* **name-permutation null** -- :func:`name_blind_null` recomputes the candidate's residual
  under many shufflings of the right-hand column pairing (within the same measurement kind);
  the *lift* and the *percentile* of the true residual against that null kill tautologies
  without any magic ``lift_min`` threshold.
* **parsimony** -- :func:`mdl_gain`, the bits saved versus encoding the residuals raw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ..dsl import ast as A
from ..dsl import binders as B
from ..dsl.evaluate import eval_term


def z_for_alpha(alpha: float) -> float:
    """Two-sided normal critical value for a configured significance level."""
    a = min(max(float(alpha), 1e-12), 1.0 - 1e-12)
    return float(NormalDist().inv_cdf(1.0 - a / 2.0))


def wilson(k: int, n: int, z: float = None):
    """Wilson score interval for a binomial proportion -- robust at extreme p / small n."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    if z is None:
        z = z_for_alpha(0.05)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half), phat)


@dataclass
class LiftResult:
    lift: float            # null residual / true residual (>1 means the named pairing matters)
    percentile: float      # fraction of null residuals <= true residual (low => real)
    n_null: int
    null_coverage: float = 1.0
    null_coverage_hi: float = 1.0


def _true_rel_residuals(rule: A.Rule, frame, nm):
    """Pooled two-sided relative residual ``|L-R|/max(|L|,|R|)`` for the true pairing."""
    bindings = B.enumerate_bindings(rule.binder, nm)
    res, scales = [], []
    for b in bindings:
        L = eval_term(rule.atom.left, rule.binder, b, frame, nm)
        R = eval_term(rule.atom.right, rule.binder, b, frame, nm)
        if L is None or R is None:
            continue
        res.append(L - R)
        scales.append(np.maximum(np.abs(L), np.abs(R)))
    if not res:
        return None
    s = np.concatenate(scales)
    med = np.median(s[s > 0]) if np.any(s > 0) else 1.0
    s = np.maximum(s, 1e-6 * med)
    return np.abs(np.concatenate(res)) / s


def _permute_within_kind(frame, nm, rng) -> "object":
    """A frame where columns are shuffled *within each measurement kind*.

    This breaks the specific named pairing a rule relies on while preserving each kind's marginal
    distribution AND the magnitude of family aggregates (a ``SUM`` over a family still sums the
    same number of same-kind columns).  Re-grounding the rule on this frame is therefore a
    magnitude-fair name-permutation null -- unlike swapping a whole ``SUM`` for one column.
    """
    from ..loader.loader import Frame

    by_kind: dict = {}
    for j, name in enumerate(frame.names):
        sem = nm.by_name.get(name)
        key = sem.kind if sem is not None else "?"
        by_kind.setdefault(key, []).append(j)
    new = frame.matrix.copy()
    for _, idxs in by_kind.items():
        if len(idxs) < 2:
            continue
        perm = list(idxs)
        rng.shuffle(perm)
        new[:, idxs] = frame.matrix[:, perm]
    return Frame(new, frame.names)


def name_blind_null(rule: A.Rule, frame, nm, n_perm: int, seed: int,
                    eps: float = None, z: float = None) -> LiftResult:
    """Lift + percentile of the true name pairing against a within-kind permutation null.

    The null shuffles columns within each kind and re-grounds the rule (see
    :func:`_permute_within_kind`).  We report:

    * ``percentile`` = fraction of pooled null residual points <= the true median residual;
    * ``lift``       = median(pooled null residual) / true median residual.

    A genuine tight law has ``percentile`` near 0 and large lift.  A tautology, a loose/false
    equality (``v ~= 2v``) or a spurious aggregate identity leaves a residual that is *not* small
    relative to the magnitude-fair null, so it lands at a high percentile and is rejected -- no
    ``lift_min`` / ``eps_max`` constant required.
    """
    tr = _true_rel_residuals(rule, frame, nm)
    if tr is None or tr.size == 0:
        return LiftResult(1.0, 1.0, 0)
    true_med = float(np.median(tr)) + 1e-12

    import random as _random
    pyrng = _random.Random(seed)
    null_points = []
    null_medians = []
    for _ in range(max(1, n_perm)):
        pf = _permute_within_kind(frame, nm, pyrng)
        nr = _true_rel_residuals(rule, pf, nm)
        if nr is not None and nr.size:
            null_points.append(nr)
            null_medians.append(float(np.median(nr)))
    if not null_points:
        return LiftResult(1.0, 1.0, 0)
    pooled = np.concatenate(null_points)
    lift = float(np.median(pooled)) / true_med
    zero_constant_law = (
        isinstance(rule.atom.left, A.Const) and rule.atom.left.value == 0
        or isinstance(rule.atom.right, A.Const) and rule.atom.right.value == 0
    )
    if zero_constant_law:
        true_cov = float(np.mean(tr <= true_med + 1e-15))
        null_covs = np.asarray([np.mean(nr <= true_med + 1e-15) for nr in null_points])
        percentile = float(np.mean(null_covs >= true_cov))
    else:
        percentile = float(np.mean(pooled <= true_med))
    null_cov = 1.0
    null_hi = 1.0
    if eps is not None:
        k = int(np.count_nonzero(pooled <= eps + 1e-15))
        _lo, null_hi, null_cov = wilson(k, int(pooled.size), z=z)
    return LiftResult(lift=lift, percentile=percentile, n_null=int(pooled.size),
                      null_coverage=null_cov, null_coverage_hi=null_hi)


def mdl_gain(rule: A.Rule, eps: float, resid_over_s: np.ndarray) -> float:
    """Bits saved by the rule vs encoding the residuals raw (the parsimony-adjusted rank).

    A band of relative width ``eps`` describes each point in ~``-log2(eps)`` fewer bits than a
    raw O(1)-scale value, so a *tighter* band saves *more* bits; the typical residual magnitude
    contributes the same way.  The form's own description length is charged as a parsimony
    penalty, so shorter, tighter invariants rank higher.  Larger is better.
    """
    eps_c = min(1.0, max(1e-3, eps))
    bits_band = -math.log2(eps_c)
    if resid_over_s.size:
        disp = float(np.median(np.abs(resid_over_s)))
    else:
        disp = eps_c
    disp_c = min(1.0, max(1e-3, disp))
    bits_resid = -math.log2(disp_c)
    form_cost = 0.15 * rule.complexity()
    return 0.5 * (bits_band + bits_resid) - form_cost
