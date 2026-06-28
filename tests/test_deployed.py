"""E5 -- deployed (observed-only) evaluator tests.

The default evaluator uses the clean/noisy *oracle* (``ds.clean``) to separate injected
noise from real structure (``noise.model.decompose``).  ``EvalConfig.deployed=True`` switches
to the observed-only estimator (``decompose_observed``) that never reads ``ds.clean`` and
instead *models* per-cell noise at a relative level ``rel_noise`` (eta).

These tests pin the three load-bearing E5 claims:

1. *Leakage-free by construction*: in deployed mode ``evaluate`` never calls the oracle
   ``decompose`` (so it cannot read ``ds.clean``); the default mode does.
2. *Conservative noise floor*: the modelled ``sigma_prop`` (eta=0.02) is >= the oracle's
   measured ``sigma_prop`` for a noisy-leaf rule -- the floor only ever rises when the clean
   oracle is removed, so the gate cannot become falsely permissive.
3. *The honest deployment cost*: sub-noise soft-structural laws (I5/I6, ~1.9% bias < ~2%
   modelled noise) relax EXACT under the observed-only gate, while genuinely exact (I4) and
   anti (I9) laws are unaffected -- form recall is preserved, strict recall is not.

All tests are offline (scripted proposer, no network) and read-only.
"""

from __future__ import annotations

from dataclasses import replace

import math

import pytest

from autogram.cli import _verdict_by_target
from autogram.config import EvalConfig, RunConfig
from autogram.evaluator import evaluator as ev
from autogram.evaluator.evaluator import evaluate
from autogram.evaluator.gate import Verdict
from autogram.noise.model import decompose, decompose_observed
from autogram.proposer import make_proposer
from autogram.search.loop import learn


def _scripted_oracle_portfolio(ds):
    """A small deterministic oracle-graded scripted run; returns its portfolio."""
    rc = RunConfig(dataset=ds.name)
    rc.search.iterations = 120
    rc.reseed()
    return learn(ds, rc, make_proposer("scripted")).portfolio, rc


def test_deployed_evaluate_never_calls_oracle_decompose(abilene, monkeypatch):
    portfolio, rc = _scripted_oracle_portfolio(abilene)
    rule = portfolio[0].rule

    def _boom(*_a, **_k):
        raise AssertionError("oracle decompose() must not run in deployed mode")

    monkeypatch.setattr(ev, "decompose", _boom)

    # Deployed: must succeed without ever touching the oracle path.
    out = evaluate(rule, abilene, replace(rc.eval, deployed=True))
    assert isinstance(out.verdict, Verdict)

    # Default (oracle) mode: must take the patched oracle path and raise -- proving the
    # branch really is the only consumer of the clean frame.
    with pytest.raises(AssertionError):
        evaluate(rule, abilene, replace(rc.eval, deployed=False))


def test_modelled_noise_floor_dominates_oracle_floor(abilene):
    """For a rule that actually touches noisy (low_*) cells, the modelled sigma_prop is a
    conservative over-estimate of the true (oracle-measured) sigma_prop."""
    portfolio, _ = _scripted_oracle_portfolio(abilene)
    nm, obs, clean = abilene.name_model, abilene.observed, abilene.clean

    checked = 0
    for r in portfolio:
        oracle = decompose(r.rule, obs, clean, nm)
        if oracle.degenerate or oracle.sigma_prop <= 0.0:
            continue  # high_*-only rule: no injected noise to propagate
        obsd = decompose_observed(r.rule, obs, nm, 0.02)
        assert not obsd.degenerate
        assert obsd.n_bindings > 0
        for v in (obsd.structural_bias, obsd.structural_scale, obsd.sigma_prop):
            assert math.isfinite(v)
        assert obsd.structural_scale >= 0.0
        assert obsd.sigma_prop > 0.0
        # The whole point of the deployed gate: the modelled floor never undercuts the
        # true floor, so removing the oracle can only make the gate stricter.
        assert obsd.sigma_prop >= oracle.sigma_prop - 1e-9
        checked += 1

    assert checked > 0, "expected at least one noisy-leaf rule in the scripted portfolio"


def test_deployed_relaxes_subnoise_soft_keeps_exact_and_anti(abilene):
    portfolio, rc = _scripted_oracle_portfolio(abilene)
    v_oracle = _verdict_by_target(portfolio)

    dep = replace(rc.eval, deployed=True)
    regraded = [evaluate(r.rule, abilene, dep) for r in portfolio]
    v_deployed = _verdict_by_target(regraded)

    # Genuinely exact two-end conservation (I4) and the directionality anti-law (I9) are
    # invariant to losing the clean oracle.
    assert v_oracle["I4"] == "EXACT"
    assert v_deployed["I4"] == "EXACT"
    assert v_oracle["I9"] == "ANTI"
    assert v_deployed["I9"] == "ANTI"

    # The two sub-noise soft-structural laws (~1.9% bias < ~2% modelled noise) are detectable
    # ONLY with the clean oracle; under the observed-only gate they relax to EXACT.
    for tid in ("I5", "I6"):
        assert v_oracle[tid] == "SOFT_STRUCTURAL", f"{tid} should be soft under the oracle"
        assert v_deployed[tid] == "EXACT", f"{tid} should relax to EXACT when deployed"
