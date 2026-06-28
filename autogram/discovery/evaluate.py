"""The data-only evaluator: judge a candidate invariant from data alone (P1, P3, P5).

No clean oracle, no catalogue, no tuned constant.  Each candidate is scored by:

1. **Self-calibrated band** -- the tolerance ``eps`` and its operating *coverage* are read off
   the rule's own residual distribution by knee detection (:func:`band.fit_band_auto`).
   Coverage is an OUTPUT, not a target.
2. **Name-permutation lift** -- the true column pairing must beat a same-kind shuffle null,
   scored by a percentile / significance level ``alpha`` (FDR control), never ``lift_min``.
3. **Support + stability vs null** -- held-out and per-split Wilson lower bounds must stay
   above the name-permuted by-chance coverage upper bound.  The threshold is the data's own
   null reference, not a configured coverage floor or variance tolerance.
4. **Parsimony** -- candidates are ranked by MDL gain (bits saved vs encoding residuals raw).

Acceptance is the two-test rule of P5: ACCEPT iff (held-out coverage on a stable plateau) AND
(lift beats the null percentile) AND (support is sufficient) AND (MDL gain is positive).
Strictness (exact/soft) is a descriptive LABEL read off the fitted residual distribution and
operator -- not a gated decision.

Anti-invariants (``!=``) are intentionally not *accepted* here: certifying a separation from
data alone (without a directional null) would also fire on independent columns and break the
false-discovery guarantee on null data.  They remain admissible (the search may form them) but
the data-only grader does not award them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..config import DiscoveryConfig
from ..dsl import ast as A
from ..dsl.evaluate import ground
from ..evaluator.band import fit_band_auto, violation_magnitude
from ..evaluator.metrics import LiftResult, mdl_gain, name_blind_null, wilson, z_for_alpha

_ACCEPT_OPS = ("~=", "==")


@dataclass
class Evaluation:
    rule: A.Rule
    accepted: bool
    reason: str
    eps: float
    coverage: float           # held-out coverage at the self-calibrated operating point
    coverage_lo: float
    coverage_hi: float
    operating_cov: float      # the knee-chosen coverage the band was fit to
    lift: float
    lift_percentile: float
    support: float
    n_points: int
    n_bindings: int
    stability_std: float      # std of held-out coverage across splits/time (low = trustworthy)
    stability_min: float      # worst per-split held-out coverage
    support_null_hi: float    # upper CI of coverage expected from name-permuted null pairings
    support_margin: float     # full held-out coverage_lo - support_null_hi
    stability_margin: float   # worst split coverage_lo - support_null_hi
    mdl_gain: float           # bits saved vs encoding residuals raw (parsimony-adjusted rank)
    strictness: str           # exact | soft | loose | anti  (descriptive label)
    descriptor: tuple         # (binder, length) Pareto cell key

    def summary(self) -> str:
        return (f"{self.rule.unparse():<46s} {self.strictness:<6s} "
                f"cov={self.coverage:.3f}[{self.coverage_lo:.2f},{self.coverage_hi:.2f}] "
                f"lift={self.lift:6.1f} p={self.lift_percentile:.3f} "
                f"stab={self.stability_std:.3f} mdl={self.mdl_gain:+.2f}")


def _row_subsets(n_rows: int, n_splits: int, n_blocks: int, seed: int) -> List[np.ndarray]:
    subsets: List[np.ndarray] = []
    if n_rows >= 2 * n_splits:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_rows)
        subsets += [np.sort(s) for s in np.array_split(perm, n_splits)]
    if n_rows >= 2 * n_blocks:
        subsets += [b for b in np.array_split(np.arange(n_rows), n_blocks)]
    return [s for s in subsets if s.size >= 2]


def _strictness(op: str, eps: float, residuals: np.ndarray) -> str:
    if op == "!=":
        return "anti"
    if residuals.size == 0:
        return "reject"
    if eps <= float(np.min(residuals)) + 1e-15:
        return "exact"
    if eps < float(np.max(residuals)):
        return "soft"
    return "loose"


class DataOnlyEvaluator:
    """Evaluate a rule against a dataset using only observable data."""

    def __init__(self, ds, cfg: DiscoveryConfig = None):
        self.ds = ds
        self.cfg = cfg or DiscoveryConfig()

    def evaluate(self, rule: A.Rule) -> Evaluation:
        cfg = self.cfg
        nm = self.ds.name_model
        frame = self.ds.observed
        op = rule.atom.op

        g = ground(rule, frame, nm, subsample=cfg.subsample, seed=cfg.seed)
        if g.degenerate or g.n_points == 0:
            return self._reject(rule, "degenerate / too few points")

        band, operating_cov = fit_band_auto(op, g.rho, g.scale, cfg.holdout_frac, cfg.seed)
        z = z_for_alpha(cfg.alpha)
        lo, hi, phat = wilson(band.k_eval, max(band.n_eval, 1), z=z)
        resid_over_s = np.abs(g.rho) / g.scale
        gain = mdl_gain(rule, band.eps, resid_over_s)
        strict = _strictness(op, band.eps, resid_over_s)
        descriptor = (rule.binder, rule.length())

        # name-permutation null (the lift test) -- the gate that kills tautologies / spurious
        lift = name_blind_null(rule, frame, nm, cfg.n_perm, cfg.seed, eps=band.eps, z=z)
        support_margin = lo - lift.null_coverage_hi

        # stability across disjoint random splits and contiguous time blocks
        stab_std, stab_min, stab_margin = self._stability(
            rule, op, band.eps, lift.null_coverage, lift.null_coverage_hi, z)

        base = Evaluation(
            rule=rule, accepted=False, reason="", eps=band.eps,
            coverage=phat, coverage_lo=lo, coverage_hi=hi, operating_cov=operating_cov,
            lift=lift.lift, lift_percentile=lift.percentile, support=g.support,
            n_points=g.n_points, n_bindings=g.n_bindings,
            stability_std=stab_std, stability_min=stab_min,
            support_null_hi=lift.null_coverage_hi, support_margin=support_margin,
            stability_margin=stab_margin, mdl_gain=gain,
            strictness=strict, descriptor=descriptor)

        ok, reason = self._accept(op, base, lift, cfg)
        base.accepted = ok
        base.reason = reason
        return base

    def _stability(self, rule: A.Rule, op: str, eps: float,
                   null_coverage: float, null_coverage_hi: float, z: float) -> tuple:
        """Coverage reproducibility at a FIXED global tolerance across splits/time.

        The operating band ``eps`` is fit once on the full frame, then coverage is measured at
        that *same* ``eps`` on each disjoint random split and contiguous time block.  Each
        split's Wilson lower bound must exceed the name-permuted by-chance coverage upper bound,
        and the observed cross-split variance must fit inside the null's own finite-sample
        uncertainty band.  This self-calibrates the stability gate from the rule/data/null
        instead of comparing coverage variance to a fixed tolerance.
        """
        cfg = self.cfg
        nm = self.ds.name_model
        frame = self.ds.observed
        subsets = _row_subsets(frame.n_rows, cfg.n_splits, cfg.n_time_blocks, cfg.seed)
        covs = []
        lower_bounds = []
        split_sizes = []
        for rows in subsets:
            sub = frame.slice_rows(rows)
            gg = ground(rule, sub, nm, seed=cfg.seed)
            if gg.degenerate or gg.n_points == 0:
                continue
            v = violation_magnitude(op, gg.rho, gg.scale)
            if v.size == 0:
                continue
            k = int(np.count_nonzero(v <= eps + 1e-15))
            lo, _hi, phat = wilson(k, int(v.size), z=z)
            covs.append(phat)
            lower_bounds.append(lo)
            split_sizes.append(int(v.size))
        if len(covs) < 2:
            return (1.0, 0.0, -1.0)
        arr = np.asarray(covs)
        min_split = max(1, min(split_sizes))
        null_instability = z * float(np.sqrt(
            max(0.0, null_coverage_hi * (1.0 - null_coverage_hi)) / min_split))
        coverage_margin = float(min(lower_bounds) - null_coverage_hi)
        variance_margin = float(null_instability - np.std(arr))
        return (float(np.std(arr)), float(np.min(arr)),
                min(coverage_margin, variance_margin))

    def _accept(self, op: str, ev: Evaluation, lift: LiftResult,
                cfg: DiscoveryConfig) -> tuple:
        if op not in _ACCEPT_OPS:
            return False, "only two-sided coverage invariants are certified data-only"
        # parsimony is a proxy metric: negative gain means the rule increases description length.
        if cfg.require_parsimony and ev.mdl_gain <= 0.0:
            return False, "not parsimonious (negative MDL gain)"
        # Structural guard: two bindings are the minimum needed to estimate cross-binding
        # generalisation.  The coverage threshold itself is the null-derived margin below.
        if ev.n_bindings <= 1:
            return False, "not enough bindings to estimate cross-binding generalisation"
        if cfg.require_null_support:
            if ev.support_margin <= 0.0:
                return False, "held-out coverage does not beat name-permuted by-chance support"
        elif ev.coverage_lo <= 0.0:
            return False, "insufficient held-out coverage support"
        # lifts: the named pairing must beat the same-kind permutation null
        if cfg.require_lift and not (lift.lift > 1.0 and lift.percentile <= cfg.alpha):
            return False, "no name-permutation lift (tautology / spurious)"
        # stable plateau: every split/time block must beat the null coverage reference.
        if cfg.require_stability:
            if cfg.require_null_support:
                if ev.stability_margin <= 0.0:
                    return False, "coverage not stable above name-permuted by-chance level"
            elif ev.stability_min <= 0.0:
                return False, "coverage collapses on a held-out split"
        return True, "stable, lifts, supported"

    def _reject(self, rule: A.Rule, reason: str) -> Evaluation:
        return Evaluation(
            rule=rule, accepted=False, reason=reason, eps=0.0, coverage=0.0,
            coverage_lo=0.0, coverage_hi=0.0, operating_cov=0.0, lift=1.0,
            lift_percentile=1.0, support=0.0, n_points=0, n_bindings=0,
            stability_std=1.0, stability_min=0.0, support_null_hi=1.0,
            support_margin=-1.0, stability_margin=-1.0, mdl_gain=0.0,
            strictness="reject", descriptor=(rule.binder, rule.length()))
