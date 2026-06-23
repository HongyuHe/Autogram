"""Vectorized grounding and evaluation of DSL rules (design Sec. 5.2, 10.1).

Grounding a :class:`~autogram.dsl.ast.Rule` against a data :class:`~autogram.loader.loader.Frame`
produces, for every (snapshot t, binding b), a left value ``L`` and right value ``R``.
From these we derive:

* the *raw residual*   ``rho = L - R``,
* a *typed scale*      ``s = max(|L|, |R|)`` (the relative-error denominator; a small
  floor avoids division by zero),

and the per-operator *soft-satisfaction* test against a fitted band ``eps(s)``.

A genuine equality/approximate invariant satisfies ``|rho| <= eps(s)`` on (almost) all
points; an *anti-invariant* (``!=``) satisfies ``|rho| > eps(s)`` -- it asserts the two
sides reliably differ (e.g. I9 directionality).  The band itself is *fitted analytically*
by the evaluator (Sec. 10.4), never searched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..loader.loader import Frame
from ..loader.names import NameModel
from . import ast as A
from . import binders as B


@dataclass
class Band:
    """Soft-tolerance band ``eps(s) = slope * s + floor`` (doc Sec. 6.6 ``band``)."""
    slope: float = 0.0
    floor: float = 0.0

    def eps(self, s: np.ndarray) -> np.ndarray:
        return self.slope * s + self.floor


@dataclass
class Grounded:
    """A rule's residual population, computed once and reused across the evaluator."""
    rho: np.ndarray          # (n_points,) raw residuals  L - R
    scale: np.ndarray        # (n_points,) typed scale     max(|L|,|R|)
    n_bindings: int          # bindings that grounded successfully (in scope)
    n_candidates: int        # bindings attempted (for support denominator)
    degenerate: bool         # True if no in-scope bindings

    @property
    def n_points(self) -> int:
        return self.rho.shape[0]

    @property
    def support(self) -> float:
        """Fraction of attempted bindings that grounded in scope (Sec. 10.1)."""
        if self.n_candidates == 0:
            return 0.0
        return self.n_bindings / self.n_candidates


_TINY = 1e-9


def eval_term(term: A.Term, binder: str, binding: dict, frame: Frame,
              nm: NameModel):
    """Evaluate a term for one binding -> (N,) array, or ``None`` if out of scope."""
    if isinstance(term, A.Const):
        return np.full(frame.n_rows, float(term.value))
    if isinstance(term, A.Ref):
        col = B.resolve_ref(term.role, binder, binding, nm)
        if col is None or not frame.has(col):
            return None
        return frame.col(col)
    if isinstance(term, A.Scale):
        inner = eval_term(term.term, binder, binding, frame, nm)
        return None if inner is None else term.coeff * inner
    if isinstance(term, A.Add):
        acc = np.zeros(frame.n_rows)
        for t in term.terms:
            v = eval_term(t, binder, binding, frame, nm)
            if v is None:
                return None
            acc = acc + v
        return acc
    if isinstance(term, A.Agg):
        cols = B.resolve_family(term.family_role, binder, binding, nm)
        if not cols:
            # an empty family aggregates to the additive/extremal identity
            if term.kind == "SUM":
                return np.zeros(frame.n_rows)
            return None
        mat = np.stack([frame.col(c) for c in cols], axis=1)
        if term.kind == "SUM":
            return mat.sum(axis=1)
        if term.kind == "AVG":
            return mat.mean(axis=1)
        if term.kind == "MIN":
            return mat.min(axis=1)
        if term.kind == "MAX":
            return mat.max(axis=1)
    raise TypeError(f"unknown term {term!r}")


def ground(rule: A.Rule, frame: Frame, nm: NameModel,
           scale_floor_frac: float = 1e-6, subsample: int = 0, seed: int = 0) -> Grounded:
    """Ground a rule to its residual population over all snapshots x bindings.

    ``subsample`` (>0) caps the number of residual points kept, drawn reproducibly -- the
    coreset lever of Sec. 10.2 for scaling to millions of points.
    """
    bindings = B.enumerate_bindings(rule.binder, nm)
    rhos, scales = [], []
    n_ok = 0
    for b in bindings:
        L = eval_term(rule.atom.left, rule.binder, b, frame, nm)
        R = eval_term(rule.atom.right, rule.binder, b, frame, nm)
        if L is None or R is None:
            continue
        n_ok += 1
        rho = L - R
        s = np.maximum(np.abs(L), np.abs(R))
        rhos.append(rho)
        scales.append(s)
    if n_ok == 0:
        return Grounded(np.empty(0), np.empty(0), 0, len(bindings), True)
    rho = np.concatenate(rhos)
    scale = np.concatenate(scales)
    mask = np.isfinite(rho) & np.isfinite(scale)
    rho, scale = rho[mask], scale[mask]
    if subsample and rho.size > subsample:
        rng = np.random.default_rng(seed)
        keep = rng.choice(rho.size, size=subsample, replace=False)
        rho, scale = rho[keep], scale[keep]
    # global floor keeps near-zero-scale points from exploding the relative residual
    med = np.median(scale[scale > 0]) if np.any(scale > 0) else 1.0
    floor = scale_floor_frac * med
    scale = np.maximum(scale, floor)
    return Grounded(rho=rho, scale=scale, n_bindings=n_ok,
                    n_candidates=len(bindings), degenerate=False)


@dataclass
class GroundedPair:
    """Aligned observed/clean residuals for the same bindings (noise-gate input)."""
    rho_obs: np.ndarray
    rho_clean: np.ndarray
    scale: np.ndarray        # scale from the observed frame
    n_bindings: int
    degenerate: bool

    @property
    def rho_noise(self) -> np.ndarray:
        """Injected-noise contribution to the residual (linear arithmetic => exact)."""
        return self.rho_obs - self.rho_clean


def ground_both(rule: A.Rule, observed: Frame, clean: Frame, nm: NameModel,
                scale_floor_frac: float = 1e-6) -> GroundedPair:
    """Ground a rule on both frames with identical binding order.

    Because every term is linear, the noise-induced residual is exactly
    ``rho_obs - rho_clean``; the *structural* residual is ``rho_clean``.  This is
    the propagated-noise signal the evaluator gate uses (design Sec. 10.1) -- it
    never feeds the proposer or the search scorer.
    """
    bindings = B.enumerate_bindings(rule.binder, nm)
    ro, rc, sc = [], [], []
    n_ok = 0
    for b in bindings:
        Lo = eval_term(rule.atom.left, rule.binder, b, observed, nm)
        Ro = eval_term(rule.atom.right, rule.binder, b, observed, nm)
        if Lo is None or Ro is None:
            continue
        Lc = eval_term(rule.atom.left, rule.binder, b, clean, nm)
        Rc = eval_term(rule.atom.right, rule.binder, b, clean, nm)
        if Lc is None or Rc is None:
            continue
        n_ok += 1
        ro.append(Lo - Ro)
        rc.append(Lc - Rc)
        sc.append(np.maximum(np.abs(Lo), np.abs(Ro)))
    if n_ok == 0:
        e = np.empty(0)
        return GroundedPair(e, e, e, 0, True)
    rho_obs = np.concatenate(ro)
    rho_clean = np.concatenate(rc)
    scale = np.concatenate(sc)
    mask = np.isfinite(rho_obs) & np.isfinite(rho_clean) & np.isfinite(scale)
    rho_obs, rho_clean, scale = rho_obs[mask], rho_clean[mask], scale[mask]
    med = np.median(scale[scale > 0]) if np.any(scale > 0) else 1.0
    scale = np.maximum(scale, scale_floor_frac * med)
    return GroundedPair(rho_obs=rho_obs, rho_clean=rho_clean, scale=scale,
                        n_bindings=n_ok, degenerate=False)


def soft_sat(g: Grounded, op: str, band: Band) -> np.ndarray:
    """Boolean satisfaction array for the residual population under ``band``."""
    rho, s = g.rho, g.scale
    if op == "~=":
        return np.abs(rho) <= band.eps(s)
    if op == "==":
        return np.abs(rho) <= (_TINY + _TINY * s)
    if op == "<=":
        return rho <= (_TINY + _TINY * s)
    if op == ">=":
        return rho >= -(_TINY + _TINY * s)
    if op == "!=":
        return np.abs(rho) > band.eps(s)
    raise ValueError(f"unknown op {op!r}")


def rel_residual(g: Grounded) -> np.ndarray:
    """``|rho| / s`` -- the dimensionless residual used for band fitting."""
    return np.abs(g.rho) / g.scale
