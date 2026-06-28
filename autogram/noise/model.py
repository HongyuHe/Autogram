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
from typing import List, Optional, Tuple

import numpy as np

from ..dsl import ast as A
from ..dsl import binders as B
from ..dsl.evaluate import GroundedPair, eval_term, ground_both


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


# --------------------------------------------------------------------------- deployed mode
# The oracle ``decompose`` above is only available *during development*, because it reads the
# hidden clean frame.  At *deployment* (design Sec. 5.5) the engine sees the noisy ``observed``
# frame only.  ``decompose_observed`` reconstructs the same NoiseDecomposition *without the
# clean frame*, using a single declared noise-model assumption -- a relative per-cell noise
# level ``eta`` (config ``eval.rel_noise``) -- propagated analytically through the candidate's
# linear arithmetic.  This is a *calibration constant*, NOT oracle access; the clean frame is
# never touched on this path (enforced by ``tests/test_deployed.py``).
#
# How the three summaries are recovered from the noisy residual ``z = rho_obs / s`` alone:
#   * ``structural_bias = median(z)``           -- zero-mean injected noise cancels under the
#                                                  median, so the persistent structural centre
#                                                  (e.g. the ~1.9% I5/I6 deficit) survives.
#   * ``sigma_prop``     = median of the *propagated* per-point noise std (in z-units), i.e.
#                          ``eta * sqrt(sum_i (c_i x_i)^2) / s`` over the residual's signed
#                          linear leaves restricted to noisy (``low_*``) cells -- the analytic
#                          analogue of the oracle's measured ``robust_scale(rho_noise / s)``.
#   * ``structural_scale``= ``sqrt(max(0, robust_scale(z)^2 - sigma_prop^2))`` -- the observed
#                          spread with the modelled noise variance deconvolved out.
#
# The crucial, honest consequence (validated empirically, both datasets): the *measured* oracle
# noise is far below ``eta`` (sigma_prop ~ 0.0015 vs eta ~ 0.02 for I5/I6), so the deployed
# floor ``gate_k * sigma_prop`` is ~0.12 and the sub-noise ~1.9% structural deficit no longer
# clears it -- I5/I6 fall back EXACT (the noise cannot be distinguished from structure without
# the clean oracle).  Exact (I4/I7/I8) and anti (I9) invariants are unaffected.

_LeafList = List[Tuple[float, np.ndarray, bool]]   # (coeff, column values, is_noisy)


def _linear_leaves(term: A.Term, sign: float, binder: str, binding: dict,
                   frame, nm) -> Optional[_LeafList]:
    """Collect signed linear leaves ``(coeff, x, noisy)`` of a term for one binding.

    Returns ``None`` if the term contains a non-linear aggregation (MIN/MAX), in which case
    the caller falls back to a conservative noise floor.  ``noisy`` marks ``low_*`` cells --
    the only columns the injected noise touches (the clean/observed frames agree elsewhere).
    """
    if isinstance(term, A.Const):
        return []                                    # exact literal carries no noise
    if isinstance(term, A.Ref):
        col = B.resolve_ref(term.role, binder, binding, nm)
        if col is None or not frame.has(col):
            return None
        return [(sign, frame.col(col), col in nm.low_cols)]
    if isinstance(term, A.Scale):
        return _linear_leaves(term.term, sign * term.coeff, binder, binding, frame, nm)
    if isinstance(term, A.Add):
        acc: _LeafList = []
        for t in term.terms:
            sub = _linear_leaves(t, sign, binder, binding, frame, nm)
            if sub is None:
                return None
            acc.extend(sub)
        return acc
    if isinstance(term, A.Agg):
        if term.kind in ("MIN", "MAX"):
            return None                              # non-linear -> caller falls back
        cols = B.resolve_family(term.family_role, binder, binding, nm)
        if not cols:
            return []
        if term.kind == "SUM":
            coeff = sign
        elif term.kind == "AVG":
            coeff = sign / len(cols)
        else:
            return None
        out: _LeafList = []
        for c in cols:
            if not frame.has(c):
                return None
            out.append((coeff, frame.col(c), c in nm.low_cols))
        return out
    return None


def decompose_observed(rule: A.Rule, observed, nm, eta: float = 0.02) -> NoiseDecomposition:
    """Observed-only noise/structure decomposition (no clean frame); see module note above."""
    bindings = B.enumerate_bindings(rule.binder, nm)
    z_parts: List[np.ndarray] = []
    sig_parts: List[np.ndarray] = []
    n_ok = 0
    for b in bindings:
        L = eval_term(rule.atom.left, rule.binder, b, observed, nm)
        R = eval_term(rule.atom.right, rule.binder, b, observed, nm)
        if L is None or R is None:
            continue
        n_ok += 1
        rho = L - R
        s = np.maximum(np.abs(L), np.abs(R))
        base = np.median(s[s > 0]) if np.any(s > 0) else 1.0
        s = np.maximum(s, 1e-6 * base)               # mirror evaluate.ground's scale floor
        z_parts.append(rho / s)
        ll = _linear_leaves(rule.atom.left, 1.0, rule.binder, b, observed, nm)
        lr = _linear_leaves(rule.atom.right, -1.0, rule.binder, b, observed, nm)
        if ll is None or lr is None:
            # non-linear term: conservative full-eta floor (over-estimates noise -> safe).
            sig_parts.append(np.full_like(s, float(eta)))
            continue
        var = np.zeros_like(s)
        for coeff, arr, noisy in (ll + lr):
            if noisy:
                var = var + (coeff * arr) ** 2
        sig_parts.append(float(eta) * np.sqrt(var) / s)
    if n_ok == 0:
        e = np.empty(0)
        return NoiseDecomposition(0.0, 0.0, 0.0, e, e, 0, True)
    z = np.concatenate(z_parts)
    sig = np.concatenate(sig_parts)
    mask = np.isfinite(z) & np.isfinite(sig)
    z, sig = z[mask], sig[mask]
    if z.size == 0:
        e = np.empty(0)
        return NoiseDecomposition(0.0, 0.0, 0.0, e, e, n_ok, True)
    sigma_prop = float(np.median(sig)) if sig.size else 0.0
    obs_scale = robust_scale(z)
    struct_scale = float(np.sqrt(max(0.0, obs_scale ** 2 - sigma_prop ** 2)))
    return NoiseDecomposition(
        structural_bias=float(np.median(z)),
        structural_scale=struct_scale,
        sigma_prop=sigma_prop,
        structural_resid=z,                          # observed z (no clean frame available)
        noise_resid=sig,                             # modelled per-point noise std (z-units)
        n_bindings=n_ok,
        degenerate=False,
    )
