"""Analytic tolerance (band) fitting -- split-conformal, never searched (design Sec. 5.4, 10.1).

For a candidate we observe a per-point residual ``rho`` and a typed scale ``s``.  We turn
this into a single dimensionless tolerance ``eps`` on ``|rho|/s`` that achieves a global
target coverage ``kappa*`` -- the *only* coverage knob, set once for the whole run, not per
rule.  Using split-conformal (fit ``eps`` on a calibration split, measure coverage on a
disjoint eval split) keeps the reported confidence honest rather than optimistic.

Why fit and not search: the residual distribution already determines the smallest band that
reaches a target coverage; searching over ``eps`` would re-discover that quantile while
multiplying the search space (Sec. 5.4).  So the inner loop is closed-form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def violation_magnitude(op: str, rho: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-point, dimensionless deviation that the band must cover.

    * ``~=`` / ``==`` -- two-sided: ``|rho| / s``.
    * ``<=``          -- only positive overshoot counts: ``max(0, rho) / s``.
    * ``>=``          -- only negative undershoot counts: ``max(0, -rho) / s``.

    ``!=`` (anti-invariant) is *not* a coverage band and is handled in the metric layer
    (it rewards *large* separation), so it is treated two-sided here for fitting purposes.
    """
    r = rho / s
    if op == "<=":
        return np.maximum(0.0, r)
    if op == ">=":
        return np.maximum(0.0, -r)
    return np.abs(r)


def _split(n: int, holdout_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = max(1, int(round((1.0 - holdout_frac) * n)))
    return idx[:n_cal], idx[n_cal:]


@dataclass
class BandFit:
    eps: float            # dimensionless tolerance on |rho|/s
    cov_cal: float        # coverage on the calibration split (~ target by construction)
    cov_eval: float       # honest coverage on the held-out split
    n_eval: int           # held-out point count (for Wilson CI)
    k_eval: int           # held-out points inside the band


def fit_band(op: str, rho: np.ndarray, s: np.ndarray, target_cov: float,
             holdout_frac: float, seed: int) -> BandFit:
    v = violation_magnitude(op, rho, s)
    n = v.size
    if n == 0:
        return BandFit(0.0, 0.0, 0.0, 0, 0)
    if n < 4:  # too few points to split honestly -- fit and report on the same data
        eps = float(np.quantile(v, target_cov))
        k = int(np.count_nonzero(v <= eps + 1e-15))
        return BandFit(eps, k / n, k / n, n, k)
    cal, ev = _split(n, holdout_frac, seed)
    # Conformal quantile with finite-sample correction: ceil((m+1) q) / m order statistic.
    m = cal.size
    q_level = min(1.0, np.ceil((m + 1) * target_cov) / m)
    eps = float(np.quantile(v[cal], q_level))
    cov_cal = float(np.mean(v[cal] <= eps + 1e-15))
    k_eval = int(np.count_nonzero(v[ev] <= eps + 1e-15))
    cov_eval = k_eval / ev.size if ev.size else cov_cal
    return BandFit(eps=eps, cov_cal=cov_cal, cov_eval=cov_eval,
                   n_eval=int(ev.size), k_eval=k_eval)
