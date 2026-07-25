"""Fast, offline tests for the gTIB byte-completeness emulator.

Run from the repository root::

    uv run python -m pytest generator/tests -q

or from inside ``generator/``::

    uv run python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from gtib_emulator.config import load_config
from gtib_emulator.generate import run
from gtib_emulator.invariants import check_all


def _small_config():
    """A tiny, fast configuration exercising every code path."""

    return load_config(overrides={
        "seed": 7,
        "time": {"duration_hours": 3.0},
        "scale": {"n_consumers": 4, "shards_min": 1, "shards_max": 3},
        "measurement": {"shard_churn_prob_per_hour": 0.2},  # exercise churn too
    })


@pytest.fixture(scope="module")
def result():
    return run(_small_config())


def test_hard_invariants_pass(result):
    cfg = result.config
    results = check_all(cfg, result.records, result.events)
    hard_failures = [r for r in results if r.tier == "hard" and not r.passed]
    assert not hard_failures, [f"{r.name}: {r.detail}" for r in hard_failures]


def test_byte_conservation_is_exact(result):
    for rec in result.records:
        phys = rec["phys"]
        resid = float(phys.conservation_residual().max())
        scale = float(phys.cum_input.max()) or 1.0
        assert resid / scale < 1e-9


def test_backlog_and_loss_are_physical(result):
    for rec in result.records:
        phys = rec["phys"]
        assert phys.backlog.min() >= -1e-9
        assert np.diff(phys.cum_true_loss, axis=1).min() >= -1e-6


def test_determinism_same_seed():
    a = run(_small_config()).derived
    b = run(_small_config()).derived
    # Same seed -> identical derived signals (ignoring NaN placement equivalence).
    assert a.shape == b.shape
    for col in ("completeness_ratio", "completeness_ratio_1h", "cum_lost_bytes"):
        x = a[col].to_numpy()
        y = b[col].to_numpy()
        assert np.allclose(x, y, equal_nan=True)


def test_derived_schema(result):
    expected = {
        "timestamp", "consumer_id", "minute_index",
        "input_rate_bytes_per_min", "output_rate_bytes_per_min",
        "completeness_ratio", "completeness_ratio_1h", "static_alert",
        "backlog_bytes", "cum_lost_bytes",
        "is_true_loss", "is_benign_burst", "is_artifact", "label", "oracle_alert",
        "traj_alert",
    }
    assert expected.issubset(set(result.derived.columns))


def test_events_have_expected_alert_semantics(result):
    ev = result.events
    if ev.empty:
        pytest.skip("no events generated in this tiny run")
    true_loss = ev[ev["type"].str.startswith("true_loss")]
    benign = ev[ev["type"].isin(["benign_burst", "artifact"])]
    assert (true_loss["expected_alert"] == True).all()      # noqa: E712
    assert (benign["expected_alert"] == False).all()        # noqa: E712


def test_healthy_ratio_above_alert_threshold_for_steady(result):
    cfg = result.config
    steady = [r["consumer"].consumer_id for r in result.records
              if r["consumer"].archetype == "steady"]
    d = result.derived
    calm = d[(d["consumer_id"].isin(steady)) & (d["label"] == "normal")]
    if len(calm) < 20:
        pytest.skip("not enough steady-normal minutes in this tiny run")
    frac_below = float((calm["completeness_ratio"] < cfg.alerting.alert_threshold).mean())
    assert frac_below < 0.10


def test_unknown_config_key_raises():
    with pytest.raises(KeyError):
        load_config(overrides={"scale": {"not_a_real_knob": 1}})
