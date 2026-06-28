"""Self-calibrated band: knee-based coverage as an OUTPUT, not a configured target (P1)."""

from __future__ import annotations

import numpy as np

from autogram.evaluator.band import fit_band_auto, knee_coverage, violation_magnitude


def test_knee_finds_tight_core():
    # 90 tight points near 0, 10 tail points near 0.9 -> knee coverage ~ 0.9
    v = np.concatenate([np.full(90, 0.01), np.full(10, 0.9)])
    cov = knee_coverage(v)
    assert 0.85 <= cov <= 0.95


def test_fit_band_auto_tight_for_structured_residual():
    rng = np.random.default_rng(0)
    core = rng.normal(0, 0.002, 400)          # tight core, |rho| < ~0.01
    tail = np.full(40, 0.9)                    # cleanly separated tail
    rho = np.concatenate([core, tail])
    s = np.ones_like(rho)
    band, cov = fit_band_auto("~=", rho, s, holdout_frac=0.5, seed=0)
    assert band.eps < 0.1            # a tight band is chosen
    assert cov > 0.8                 # coverage read off the data is high
    assert band.cov_eval > 0.8       # honest held-out coverage


def test_fit_band_auto_loose_for_featureless_residual():
    # constant relative residual (no tight core) -> band cannot be tight
    rho = np.full(300, 0.5)
    s = np.ones_like(rho)
    band, cov = fit_band_auto("~=", rho, s, holdout_frac=0.5, seed=0)
    assert band.eps >= 0.4


def test_fit_band_auto_small_sample_does_not_use_fixed_coverage_fallback():
    # A tiny sample still reports the operating point implied by the observed residuals.
    # The old implementation returned a hard-coded 0.9 target coverage here.
    rho = np.array([0.0, 0.25, 0.75])
    s = np.ones_like(rho)
    _band, cov = fit_band_auto("~=", rho, s, holdout_frac=0.5, seed=0)
    assert cov != 0.9


def test_violation_magnitude_one_sided():
    rho = np.array([-1.0, 1.0])
    s = np.ones_like(rho)
    assert np.allclose(violation_magnitude(">=", rho, s), [1.0, 0.0])
    assert np.allclose(violation_magnitude("<=", rho, s), [0.0, 1.0])
