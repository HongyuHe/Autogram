"""Analytic tolerance (band) fitting -- split-conformal, never searched (design Sec. 5.4, 10.1).

For a candidate we observe a per-point residual ``rho`` and a typed scale ``s``.  We turn
this into a single dimensionless tolerance ``eps`` on ``|rho|/s`` by finding the residual
knee: the largest observed gap between the tight core and the tail.  The achieved coverage
is therefore an output of the residual distribution, not a configured target coverage.

Why fit and not search: the residual distribution already determines the core edge;
searching over ``eps`` would re-discover that edge while multiplying the search space
(Sec. 5.4).  So the inner loop is closed-form.
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


def _knee_index(vs: np.ndarray) -> int:
    """Index of the core's upper edge: the lower side of the largest observed gap."""
    n = vs.size
    if n < 2:
        return n - 1
    gaps = np.diff(vs)
    if gaps.size == 0 or float(np.max(gaps)) <= 0.0:
        return n - 1
    return int(np.argmax(gaps))


def knee_coverage(v: np.ndarray) -> float:
    """Read an operating coverage off the residual distribution by knee detection.

    Sort the dimensionless violations ``v`` ascending and find the largest *observed gap*
    between consecutive values.  That gap is the boundary between the tight core (genuine
    structure) and the tail (violations); the fraction of points at or below it is the
    coverage the band should operate at.  Coverage is thus an OUTPUT of the data, not a
    configured target ``kappa*``.  If there is no gap, the residuals have no separable core
    and the operating point is the whole observed distribution.
    """
    n = v.size
    if n == 0:
        return 0.0
    vs = np.sort(v)
    i = _knee_index(vs)
    return (i + 1) / n


def fit_band_auto(op: str, rho: np.ndarray, s: np.ndarray,
                  holdout_frac: float, seed: int):
    """Self-calibrated band fit: set ``eps`` at the residual knee, coverage honest on held-out.

    The band edge is the *core's upper value* at the knee (split-conformal when there are
    enough points to hold out rows; tiny samples use the same observed knee rather than a
    hard-coded target).  A clean bimodal residual yields a tight ``eps`` rather than a
    quantile that rounds up into the tail.  Returns ``(BandFit, coverage)`` where
    ``coverage`` is the data-chosen operating point.  No ``kappa*`` and no noise constant
    ``eta`` are involved.
    """
    v = violation_magnitude(op, rho, s)
    n = v.size
    if n == 0:
        return BandFit(0.0, 0.0, 0.0, 0, 0), 0.0
    cal, ev = _split(n, holdout_frac, seed)
    vc = np.sort(v[cal])
    i = _knee_index(vc)
    eps = float(vc[i])
    cov_cal = float(np.mean(v[cal] <= eps + 1e-15))
    k_eval = int(np.count_nonzero(v[ev] <= eps + 1e-15))
    cov_eval = k_eval / ev.size if ev.size else cov_cal
    coverage = (i + 1) / cal.size
    return BandFit(eps=eps, cov_cal=cov_cal, cov_eval=cov_eval,
                   n_eval=int(ev.size), k_eval=k_eval), coverage
