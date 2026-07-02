"""Vectorized grounding of DSL rules.

Grounding a :class:`~autogram.dsl.ast.Rule` against a data :class:`~autogram.loader.loader.Frame`
produces, for every (snapshot t, binding b), a left value ``L`` and a right value ``R``.  From
these we derive the *raw residual* ``rho = L - R`` and a *typed scale* ``s = max(|L|, |R|)`` (the
relative-error denominator; a small floor avoids division by zero).

This module is deliberately thin: it only turns a rule + frame into a residual population.  The
data-only evaluator (:mod:`autogram.discovery.evaluate`) consumes that population -- it fits the
tolerance band, reads the operating coverage, and runs the acceptance tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..loader.loader import Frame
from ..loader.names import NameModel
from . import ast as A
from . import binders as B


@dataclass
class Grounded:
    """A rule's residual population, computed once and reused across the evaluator."""
    rho: np.ndarray          # (n_points,) raw residuals  L - R
    scale: np.ndarray        # (n_points,) typed scale     max(|L|,|R|)
    left: np.ndarray         # (n_points,) left values
    right: np.ndarray        # (n_points,) right values
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
    lefts, rights, rhos, scales = [], [], [], []
    n_ok = 0
    for b in bindings:
        L = eval_term(rule.atom.left, rule.binder, b, frame, nm)
        R = eval_term(rule.atom.right, rule.binder, b, frame, nm)
        if L is None or R is None:
            continue
        n_ok += 1
        rho = L - R
        s = np.maximum(np.abs(L), np.abs(R))
        lefts.append(L)
        rights.append(R)
        rhos.append(rho)
        scales.append(s)
    if n_ok == 0:
        empty = np.empty(0)
        return Grounded(empty, empty, empty, empty, 0, len(bindings), True)
    left = np.concatenate(lefts)
    right = np.concatenate(rights)
    rho = np.concatenate(rhos)
    scale = np.concatenate(scales)
    mask = np.isfinite(rho) & np.isfinite(scale)
    left, right, rho, scale = left[mask], right[mask], rho[mask], scale[mask]
    if subsample and rho.size > subsample:
        rng = np.random.default_rng(seed)
        keep = rng.choice(rho.size, size=subsample, replace=False)
        left, right, rho, scale = left[keep], right[keep], rho[keep], scale[keep]
    # global floor keeps near-zero-scale points from exploding the relative residual
    med = np.median(scale[scale > 0]) if np.any(scale > 0) else 1.0
    floor = scale_floor_frac * med
    scale = np.maximum(scale, floor)
    return Grounded(rho=rho, scale=scale, left=left, right=right, n_bindings=n_ok,
                    n_candidates=len(bindings), degenerate=False)


def rel_residual(g: Grounded) -> np.ndarray:
    """``|rho| / s`` -- the dimensionless residual used for band fitting."""
    return np.abs(g.rho) / g.scale
