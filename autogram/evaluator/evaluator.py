"""Top-level evaluator: ground -> fit band -> metrics -> gate -> score (design Sec. 10.1).

Produces an :class:`EvaluationResult` carrying the full Pareto vector plus a scalar
``combined_score`` used only to rank *within* a quality-diversity cell (Sec. 10.3-10.4).
The learner sees the *observed* (noisy) frame for coverage/tightness; the clean frame is
consulted only by the gate, mirroring the deployment/learning split of Sec. 5.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EvalConfig
from ..loader.loader import Dataset
from ..dsl import ast as A
from ..dsl.evaluate import ground
from ..noise.model import decompose
from .band import fit_band
from .gate import Verdict, classify
from .metrics import MDL, mdl, name_blind_lift, tightness, wilson

ACCEPT = {Verdict.EXACT, Verdict.SOFT_STRUCTURAL, Verdict.SOFT, Verdict.ANTI}


@dataclass
class EvaluationResult:
    rule: A.Rule
    verdict: Verdict
    reason: str
    eps: float
    kappa_hat: float                 # held-out coverage (confidence)
    kappa_lo: float
    kappa_hi: float
    support: float                   # fraction of in-scope bindings that grounded
    n_points: int
    n_bindings: int
    tightness: float
    lift: float
    delta: float                     # structural centre (signed bias)
    sigma_prop: float
    mdl: MDL
    combined_score: float
    accepted: bool = False
    descriptor: tuple = field(default_factory=tuple)

    def summary(self) -> str:
        return (f"{self.rule.unparse():<48s} {self.verdict.value:<16s} "
                f"eps={self.eps:.4f} kappa={self.kappa_hat:.3f} "
                f"supp={self.support:.2f} lift={self.lift:.2f} "
                f"delta={self.delta:+.4f} score={self.combined_score:.3f}")


def evaluate(rule: A.Rule, ds: Dataset, cfg: EvalConfig) -> EvaluationResult:
    op = rule.atom.op
    g = ground(rule, ds.observed, ds.name_model, subsample=cfg.subsample, seed=cfg.seed)
    if g.degenerate or g.n_points == 0:
        return _reject(rule, "degenerate grounding")

    bf = fit_band(op, g.rho, g.scale, cfg.target_coverage, cfg.holdout_frac, cfg.seed)
    nd = decompose(rule, ds.observed, ds.clean, ds.name_model)
    lift = name_blind_lift(rule, ds.observed, ds.name_model, cfg.n_perm, cfg.seed)

    resid_over_s = np.abs(g.rho) / g.scale
    dl = mdl(rule, bf.eps, resid_over_s, cfg.mdl_w_complexity, cfg.mdl_w_band, cfg.mdl_w_residual)
    verdict, reason, delta = classify(op, nd, bf.eps, lift, cfg)

    # Confidence: held-out coverage within the band (for != it is the separation fraction).
    if op == "!=":
        sep = (np.abs(g.rho) / g.scale) > max(nd.sigma_prop, cfg.eps_exact)
        k, n = int(np.count_nonzero(sep)), sep.size
    else:
        k, n = bf.k_eval, max(bf.n_eval, 1)
    lo, hi, phat = wilson(k, n)
    kappa = phat

    w = tightness(bf.eps if op != "!=" else 0.0)
    score = (cfg.w_confidence * kappa
             + cfg.w_tightness * w
             + cfg.w_support * g.support
             - cfg.w_mdl * dl.total / 100.0)
    if verdict not in ACCEPT:
        score *= 0.1

    descriptor = (min(rule.complexity(), 9), rule.binder, _tight_bucket(bf.eps))
    return EvaluationResult(
        rule=rule, verdict=verdict, reason=reason, eps=bf.eps,
        kappa_hat=kappa, kappa_lo=lo, kappa_hi=hi, support=g.support,
        n_points=g.n_points, n_bindings=g.n_bindings, tightness=w, lift=lift,
        delta=delta, sigma_prop=nd.sigma_prop, mdl=dl, combined_score=score,
        accepted=verdict in ACCEPT, descriptor=descriptor,
    )


def _tight_bucket(eps: float) -> str:
    if eps <= 0.01:
        return "exact"
    if eps <= 0.05:
        return "tight"
    if eps <= 0.15:
        return "loose"
    return "wide"


def _reject(rule: A.Rule, reason: str) -> EvaluationResult:
    return EvaluationResult(
        rule=rule, verdict=Verdict.REJECT, reason=reason, eps=0.0,
        kappa_hat=0.0, kappa_lo=0.0, kappa_hi=0.0, support=0.0, n_points=0,
        n_bindings=0, tightness=0.0, lift=1.0, delta=0.0, sigma_prop=0.0,
        mdl=MDL(0, 0, 0), combined_score=0.0, accepted=False, descriptor=(),
    )
