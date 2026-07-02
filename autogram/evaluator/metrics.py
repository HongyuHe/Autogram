"""Metrics for v2: Wilson confidence and MDL tie-breaks."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from ..dsl import ast as A


def z_for_alpha(alpha: float) -> float:
    a = min(max(float(alpha), 1e-12), 1.0 - 1e-12)
    return float(NormalDist().inv_cdf(1.0 - a / 2.0))


def wilson(k: int, n: int, z: float | None = None):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    if z is None:
        z = z_for_alpha(0.05)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half), phat)


def mdl_gain(rule: A.Rule, eps: float, resid_over_s: np.ndarray) -> float:
    """Small description-length score used only for tie-breaking survivors."""
    eps_c = min(1.0, max(1e-9, float(eps)))
    bits_band = -math.log2(eps_c)
    disp = float(np.median(np.abs(resid_over_s))) if resid_over_s.size else eps_c
    disp_c = min(1.0, max(1e-9, disp))
    bits_resid = -math.log2(disp_c)
    form_cost = 0.15 * rule.complexity()
    return 0.5 * (bits_band + bits_resid) - form_cost
