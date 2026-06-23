"""Noise-vs-structure gate: assign a verdict and strictness (design Sec. 10.1).

The gate is what stops the evaluator rewarding a candidate for *fitting noise*.  It reads
the propagated-noise decomposition (clean/noisy oracle, :mod:`autogram.noise.model`) -- not
the candidate's own observed residuals -- and decides:

* ``EXACT``           -- structure is essentially zero residual on clean data; any observed
  spread is at or below the propagated noise floor (e.g. I4 two-end, I7 flow-conservation).
* ``SOFT_STRUCTURAL`` -- a *persistent* one-sided structural bias larger than the noise floor
  survives noise removal (e.g. the ~1.9% I5/I6 deficit); reported with its centre ``delta``.
* ``SOFT``            -- approximate but real: a tight-ish band, name-semantic lift > 1, and
  the band is not merely tracking noise.
* ``REJECT``          -- no structural agreement; "holding" needs a band far beyond the noise
  floor (a false form / planted decoy) or the form is trivially true (lift ~ 1).

The verdict is *derived*, not searched, and it never sees the ground-truth catalogue.
"""

from __future__ import annotations

from enum import Enum

from ..config import EvalConfig
from ..noise.model import NoiseDecomposition


class Verdict(str, Enum):
    EXACT = "EXACT"
    SOFT_STRUCTURAL = "SOFT_STRUCTURAL"
    SOFT = "SOFT"
    REJECT = "REJECT"
    ANTI = "ANTI"


def classify(op: str, nd: NoiseDecomposition, eps: float, lift: float,
             cfg: EvalConfig) -> tuple[Verdict, str, float]:
    """Return (verdict, human reason, structural centre delta)."""
    if nd.degenerate:
        return Verdict.REJECT, "no groundable bindings", 0.0

    floor = cfg.gate_k * max(nd.sigma_prop, 1e-9)
    bias = nd.structural_bias
    scale = nd.structural_scale

    # Anti-invariant: reward *large* structural separation, not a tight band.
    if op == "!=":
        if scale > floor:
            return Verdict.ANTI, "persistent separation >> noise", bias
        return Verdict.REJECT, "ends not separable above noise", bias

    # One-sided bounds (I1-style): the only thing that matters is the *violation* band,
    # which fit_band already computes one-sided; the two-sided structural bias is moot.
    if op in (">=", "<="):
        if eps <= cfg.eps_exact:
            return Verdict.EXACT, "bound holds; ~zero violation", 0.0
        if eps <= cfg.eps_max:
            return Verdict.SOFT, "bound holds within tolerance", 0.0
        return Verdict.REJECT, "bound violated beyond tolerance", 0.0

    # Exact equality: clean residual is flat and centred within the noise floor.
    if scale <= floor and abs(bias) <= floor:
        return Verdict.EXACT, "exact on clean; observed spread <= noise floor", bias

    # Soft-structural: a *modest*, persistent, one-sided gap with a *tight* spread around
    # it and genuine name-semantic lift -- e.g. the ~1.9% I5/I6 deficit.  All three guards
    # are needed: persistence (> noise), tightness (small spread), and lift (not a fluke).
    if (abs(bias) > floor and abs(bias) <= cfg.eps_max
            and scale <= cfg.eps_max and lift >= cfg.lift_min):
        return Verdict.SOFT_STRUCTURAL, "persistent structural bias > noise floor", bias

    # Otherwise judge by the fitted band: tight + name-semantic => a real soft invariant.
    if eps <= cfg.eps_exact and lift >= cfg.lift_min:
        return Verdict.EXACT, "band collapses to noise; strong name lift", bias
    if eps <= cfg.eps_max and lift >= cfg.lift_min:
        return Verdict.SOFT, "tight band with name-semantic lift", bias
    if lift < cfg.lift_min:
        return Verdict.REJECT, "no name-semantic lift (near-tautology/false)", bias
    return Verdict.REJECT, "band too wide for the noise floor", bias
