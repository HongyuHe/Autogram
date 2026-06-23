"""Propagated noise model and the noise-vs-structure decomposition (design Sec. 5.2, 10.1).

The datasets carry injected, roughly zero-mean noise (std ~2%) on ~82% of ``low_*``
cells.  A faithful evaluator must not reward a candidate merely for *fitting that noise*.
The key idea (Sec. 10.1) is to use the clean/noisy oracle to *propagate* noise through
the candidate's (linear) arithmetic and compare the candidate's fitted tolerance against
what noise alone would force:

* ``rho_clean``  -- the *structural* residual (zero noise): reveals real bias such as the
  ~1.9% I5/I6 deficit, which survives noise removal.
* ``rho_noise = rho_obs - rho_clean`` -- the injected-noise contribution; ``sigma_prop``
  is a robust scale of it.

From these we classify a candidate (per point and in aggregate):

* **exact-masked-by-noise**: structural bias ~0 and observed band ~ sigma_prop
  (e.g. I4 two-end -- exact on clean, soft only because of noise).
* **soft-structural**: a persistent structural bias larger than the noise floor in the
  *median* sense, even if each point's deviation is sub-noise (e.g. I5/I6: ~1.9% < ~2%
  noise per-point, but a stable one-sided deficit).
* **noise-fit / reject**: no structural agreement; the only way to "hold" is a band far
  wider than sigma_prop -- the candidate is fitting noise or is simply false.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsl import ast as A
from ..dsl.evaluate import GroundedPair, ground_both


def robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD -- a Gaussian-consistent, outlier-robust standard deviation."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


@dataclass
class NoiseDecomposition:
    """Per-candidate noise/structure summary in dimensionless (rho/s) units."""
    structural_bias: float       # median(rho_clean / s)  -- signed
    structural_scale: float      # robust scale of rho_clean / s
    sigma_prop: float            # robust scale of rho_noise / s (propagated noise)
    structural_resid: np.ndarray  # rho_clean / s (for band fitting on clean structure)
    noise_resid: np.ndarray       # rho_noise / s
    n_bindings: int
    degenerate: bool


def decompose(rule: A.Rule, observed, clean, nm) -> NoiseDecomposition:
    gp: GroundedPair = ground_both(rule, observed, clean, nm)
    if gp.degenerate:
        e = np.empty(0)
        return NoiseDecomposition(0.0, 0.0, 0.0, e, e, 0, True)
    s = gp.scale
    struct = gp.rho_clean / s
    noise = gp.rho_noise / s
    return NoiseDecomposition(
        structural_bias=float(np.median(struct)),
        structural_scale=robust_scale(struct),
        sigma_prop=robust_scale(noise),
        structural_resid=struct,
        noise_resid=noise,
        n_bindings=gp.n_bindings,
        degenerate=False,
    )
