"""Guarantees-first evaluator.

Z3 decides logical triviality.  The only data statistic is hold-rate with a Wilson confidence
interval.  MDL is computed for tie-breaking, never as an acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DiscoveryConfig
from ..dsl import ast as A
from ..dsl.evaluate import ground
from ..evaluator.band import violation_magnitude
from ..evaluator.metrics import mdl_gain, wilson, z_for_alpha
from ..logic.solver import is_trivial


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
    coverage: float = 0.0
    coverage_lo: float = 0.0
    coverage_hi: float = 0.0
    operating_cov: float = 0.0
    stability_std: float = 0.0
    stability_min: float = 1.0
    support_margin: float = 0.0
    stability_margin: float = 0.0

    def __post_init__(self):
        self.coverage = self.hold_rate
        self.coverage_lo = self.hold_rate_lo
        self.coverage_hi = self.hold_rate_hi
        self.operating_cov = self.hold_rate
        self.support_margin = self.hold_rate_lo
        self.stability_margin = self.hold_rate_lo

    def summary(self) -> str:
        return (f"{self.rule.unparse():<54s} {self.strictness:<10s} "
                f"hold={self.hold_rate:.3f}[{self.hold_rate_lo:.2f},{self.hold_rate_hi:.2f}] "
                f"mdl={self.mdl_gain:+.2f} {self.reason}")


def _strictness(op: str, eps: float, rel: np.ndarray, hold: np.ndarray) -> str:
    if op == "<|>":
        return "existence"
    if op == "!=":
        return "separation"
    if op in (">=", "<="):
        return "one-sided"
    if rel.size and float(np.max(rel)) <= 1e-12:
        return "exact"
    if np.all(hold):
        return "soft"
    return "loose"


class DataOnlyEvaluator:
    """Evaluate a rule against observable data only."""

    def __init__(self, ds, cfg: DiscoveryConfig = None):
        self.ds = ds
        self.cfg = cfg or DiscoveryConfig()

    def evaluate(self, rule: A.Rule) -> Evaluation:
        cfg = self.cfg
        nm = self.ds.name_model
        frame = self.ds.observed
        op = rule.atom.op

        if is_trivial(rule):
            return self._reject(rule, "solver-trivial tautology/contradiction")

        g = ground(rule, frame, nm, subsample=cfg.subsample, seed=cfg.seed)
        if g.degenerate or g.n_points == 0:
            return self._reject(
                rule,
                f"grounded 0 points for binder {rule.binder!r} "
                f"({g.n_bindings}/{g.n_candidates} non-degenerate bindings)",
            )

        rel = np.abs(g.rho) / g.scale
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
        else:
            eps = cfg.tolerance
            holds = violation_magnitude(op, g.rho, g.scale) <= eps + 1e-15
        k = int(np.count_nonzero(holds))
        z = z_for_alpha(cfg.ci_alpha)
        lo, hi, phat = wilson(k, int(holds.size), z=z)
        gain = mdl_gain(rule, eps, rel)
        strict = _strictness(op, eps, rel, holds)
        descriptor = (rule.binder, rule.length())
        ok = lo >= cfg.hold_rate_threshold
        reason = "hold-rate above Wilson threshold" if ok else "hold-rate Wilson lower bound below threshold"
        return Evaluation(
            rule=rule, accepted=ok, reason=reason, eps=eps,
            hold_rate=phat, hold_rate_lo=lo, hold_rate_hi=hi, statistic="hold_rate",
            support=g.support, n_points=g.n_points, n_bindings=g.n_bindings,
            mdl_gain=gain, strictness=strict, descriptor=descriptor)

    def _reject(self, rule: A.Rule, reason: str) -> Evaluation:
        return Evaluation(
            rule=rule, accepted=False, reason=reason, eps=0.0,
            hold_rate=0.0, hold_rate_lo=0.0, hold_rate_hi=0.0, statistic="hold_rate",
            support=0.0, n_points=0, n_bindings=0, mdl_gain=0.0,
            strictness="reject", descriptor=(rule.binder, rule.length()))
