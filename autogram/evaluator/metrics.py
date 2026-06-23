"""Quality metrics for a candidate invariant (design Sec. 10.1).

Four families, kept separate so the search can treat them as a Pareto vector and only
scalarize *within* a quality-diversity cell:

* **confidence / support** -- ``kappa_hat`` (held-out coverage) with a Wilson interval, and
  a separate ``support`` = fraction of in-scope bindings that actually grounded.
* **tightness / informativeness** -- ``w = 1/(1+eps)`` plus the name-semantic *lift*
  ``Lambda`` versus a permutation null that breaks the column pairing; a trivially-true form
  (``v ~= v``) has ``Lambda ~ 1`` and is penalised.
* **parsimony** -- an MDL description length ``DL = L_grammar + L(eps) + L_residual``.
* (noise handling lives in :mod:`autogram.evaluator.gate`.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..dsl import ast as A
from ..dsl import binders as B
from ..dsl.evaluate import eval_term
from .band import violation_magnitude


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion -- robust at extreme p / small n."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half), phat)


def tightness(eps: float) -> float:
    """Map a band to (0, 1]; eps=0 -> 1 (perfectly tight), large eps -> ~0."""
    return 1.0 / (1.0 + max(0.0, eps))


def name_blind_lift(rule: A.Rule, frame, nm, n_perm: int, seed: int) -> float:
    """Lift of the true name pairing over a null that shuffles columns within type.

    A real invariant relies on *which* columns are paired by name semantics; permuting the
    right-hand resolution to other same-type columns should inflate the residual.  Lift =
    (null robust-residual) / (true robust-residual).  ~1 means the tight fit is an artefact
    of the form (a tautology), not of the data's named structure.
    """
    bindings = B.enumerate_bindings(rule.binder, nm)
    true_res, scales = [], []
    for b in bindings:
        L = eval_term(rule.atom.left, rule.binder, b, frame, nm)
        R = eval_term(rule.atom.right, rule.binder, b, frame, nm)
        if L is None or R is None:
            continue
        true_res.append(L - R)
        scales.append(np.maximum(np.abs(L), np.abs(R)))
    if not true_res:
        return 1.0
    s = np.concatenate(scales)
    med = np.median(s[s > 0]) if np.any(s > 0) else 1.0
    s = np.maximum(s, 1e-6 * med)
    tr = np.abs(np.concatenate(true_res)) / s
    tr_scale = float(np.median(tr)) + 1e-12

    rng = np.random.default_rng(seed)
    cols = [c for c in frame.names if frame.col(c) is not None]
    if not cols:
        return 1.0
    by_type: dict[str, list[str]] = {}
    for c in cols:
        key = c.split("_", 1)[0]   # low_* vs high_*
        by_type.setdefault(key, []).append(c)
    # Replace the right operand by a random same-type column drawn per binding.
    null_scales = []
    for _ in range(max(1, n_perm)):
        per = []
        for b in bindings:
            L = eval_term(rule.atom.left, rule.binder, b, frame, nm)
            if L is None:
                continue
            key = "high" if "high" in _rough_type(rule.atom.right) else "low"
            pool = by_type.get(key) or cols
            rc = frame.col(pool[int(rng.integers(len(pool)))])
            if rc is None:
                continue
            sc = np.maximum(np.abs(L), np.abs(rc))
            mm = np.median(sc[sc > 0]) if np.any(sc > 0) else 1.0
            sc = np.maximum(sc, 1e-6 * mm)
            per.append(np.abs(L - rc) / sc)
        if per:
            null_scales.append(float(np.median(np.concatenate(per))))
    null_scale = float(np.median(null_scales)) if null_scales else tr_scale
    return null_scale / tr_scale


def _rough_type(term: A.Term) -> str:
    """Cheap guess of a term's column family ('high' for demand, else 'low')."""
    roles = _collect_roles(term)
    if any("demand" in r for r in roles):
        return "high"
    return "low"


def _collect_roles(term: A.Term) -> list[str]:
    if isinstance(term, A.Ref):
        return [term.role]
    if isinstance(term, A.Agg):
        return [term.family_role]
    if isinstance(term, A.Scale):
        return _collect_roles(term.term)
    if isinstance(term, A.Add):
        out: list[str] = []
        for t in term.terms:
            out += _collect_roles(t)
        return out
    return []


@dataclass
class MDL:
    grammar_bits: float
    band_bits: float
    residual_bits: float

    @property
    def total(self) -> float:
        return self.grammar_bits + self.band_bits + self.residual_bits


def mdl(rule: A.Rule, eps: float, resid_over_s: np.ndarray,
        w_complexity: float, w_band: float, w_residual: float) -> MDL:
    """A pragmatic MDL: form cost + tolerance cost + Gaussian-ish residual code length."""
    grammar_bits = w_complexity * rule.complexity() * math.log2(8.0)
    band_bits = w_band * (0.0 if eps <= 0 else max(0.0, -math.log2(min(1.0, eps) + 1e-9)))
    if resid_over_s.size:
        sigma = float(np.std(resid_over_s)) + 1e-9
        residual_bits = w_residual * 0.5 * math.log2(2 * math.pi * math.e * sigma * sigma + 1e-12)
        residual_bits = max(0.0, residual_bits)
    else:
        residual_bits = 0.0
    return MDL(grammar_bits, band_bits, residual_bits)
