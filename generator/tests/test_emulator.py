"""Fast, offline tests for the gTIB byte-completeness emulator.

Run from the repository root::

    uv run python -m pytest generator/tests -q

or from inside ``generator/``::

    uv run python -m pytest tests -q
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize(
    ("corruption", "detail"),
    (
        ("row_count", "rows="),
        ("minute_order", "minute_index"),
        ("timestamp_cadence", "timestamps"),
        ("timestamp_malformed", "timestamps"),
        ("consumer_identity", "consumer_id"),
    ),
)
def test_hard_checks_reject_malformed_derived_identity_grid(
    result,
    corruption,
    detail,
):
    records = [
        {
            **record,
            "frame": record["frame"].copy(),
        }
        for record in result.records
    ]
    frame = records[0]["frame"]
    if corruption == "row_count":
        records[0]["frame"] = frame.iloc[:-1].copy()
    elif corruption == "minute_order":
        minute = frame.columns.get_loc("minute_index")
        frame.iloc[[0, 1], minute] = [1, 0]
    elif corruption == "timestamp_cadence":
        frame.loc[frame.index[1], "timestamp"] += np.timedelta64(1, "s")
    elif corruption == "timestamp_malformed":
        frame["timestamp"] = frame["timestamp"].astype(object)
        frame.loc[frame.index[1], "timestamp"] = "not-a-timestamp"
    else:
        frame["consumer_id"] = frame["consumer_id"].astype(object)
        frame.loc[frame.index[1], "consumer_id"] = None

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert detail in grid.detail


def test_hard_checks_reject_duplicate_generated_consumer(result):
    records = list(result.records)
    records[1] = copy.deepcopy(records[0])

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert "duplicate consumer_id" in grid.detail
    assert "duplicate global (consumer_id, minute_index)" in grid.detail


def test_hard_checks_reject_missing_generated_consumer(result):
    records = list(result.records[:-1])

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert "unique consumer identities=" in grid.detail


@pytest.mark.parametrize(
    "missing_identity",
    (
        None,
        pd.NA,
        ("region", ("consumer", pd.NA)),
    ),
)
def test_hard_checks_reject_missing_consumer_identity(
    result,
    missing_identity,
):
    records = list(result.records)
    records[0] = copy.deepcopy(records[0])
    records[0]["consumer"].consumer_id = missing_identity

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert "missing or invalid consumer_id" in grid.detail


def test_derived_identity_grid_preserves_nested_typed_consumers(result):
    records = copy.deepcopy(result.records)
    identities = (
        ("region", ("consumer", True)),
        ("region", ("consumer", 1)),
        ("region", ("consumer", "1")),
        ("region", ("consumer", 1.0)),
    )
    for record, identity in zip(records, identities):
        record["consumer"].consumer_id = identity
        frame = record["frame"]
        frame["consumer_id"] = pd.Series(
            [identity] * len(frame),
            index=frame.index,
            dtype=object,
        )

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert grid.passed, grid.detail


def test_hard_checks_detect_corrupted_static_alert(result):
    records = [
        {
            **record,
            "frame": record["frame"].copy(),
        }
        for record in result.records
    ]
    records[0]["frame"]["static_alert"] = ~records[0][
        "frame"
    ]["static_alert"]

    checks = check_all(result.config, records, result.events)
    static = next(
        check
        for check in checks
        if check.name == "static_alert_definition"
    )

    assert not static.passed


def test_hard_checks_detect_missing_event_catalogue(result):
    checks = check_all(
        result.config,
        result.records,
        result.events.iloc[:0].copy(),
    )
    label = next(
        check
        for check in checks
        if check.name == "label_priority_definition"
    )

    assert not label.passed


def test_hard_checks_reject_missing_boolean_truth(result):
    records = [
        {
            **record,
            "frame": record["frame"].copy(),
        }
        for record in result.records
    ]
    active = records[0]["frame"]["is_true_loss"]
    index = active[active].index[0]
    records[0]["frame"]["is_true_loss"] = records[0][
        "frame"
    ]["is_true_loss"].astype(object)
    records[0]["frame"].loc[index, "is_true_loss"] = np.nan

    checks = check_all(result.config, records, result.events)
    label = next(
        check
        for check in checks
        if check.name == "label_priority_definition"
    )

    assert not label.passed


def test_hard_checks_reject_unflagged_infinite_counter(result):
    records = copy.deepcopy(result.records)
    records[0]["obs"].input_counted[0, 0] = np.inf
    records[0]["obs"].missing_flag[0, 0] = False
    records[0]["obs"].active_flag[0, 0] = True

    checks = check_all(result.config, records, result.events)
    finite = next(
        check
        for check in checks
        if check.name == "reported_telemetry_is_finite"
    )

    assert not finite.passed


def test_hard_checks_reject_infinite_derived_value(result):
    records = [
        {
            **record,
            "frame": record["frame"].copy(),
        }
        for record in result.records
    ]
    records[0]["frame"].loc[
        records[0]["frame"].index[1],
        "input_rate_bytes_per_min",
    ] = np.inf

    checks = check_all(result.config, records, result.events)
    finite = next(
        check
        for check in checks
        if check.name == "reported_telemetry_is_finite"
    )

    assert not finite.passed


@pytest.mark.parametrize(
    ("missing", "active", "value"),
    (
        (True, True, 0.0),
        (False, False, 0.0),
        (True, True, np.inf),
        (False, True, np.nan),
    ),
)
def test_hard_checks_enforce_counter_missingness_and_finiteness(
    result,
    missing,
    active,
    value,
):
    records = copy.deepcopy(result.records)
    obs = records[0]["obs"]
    obs.missing_flag[0, 0] = missing
    obs.active_flag[0, 0] = active
    obs.input_counted[0, 0] = value

    checks = check_all(result.config, records, result.events)
    finite = next(
        check
        for check in checks
        if check.name == "reported_telemetry_is_finite"
    )

    assert not finite.passed


@pytest.mark.parametrize(
    "field",
    (
        "cum_input",
        "cum_output_physical",
        "cum_true_loss",
        "backlog",
    ),
)
def test_hard_checks_reject_nan_physical_state(result, field):
    records = copy.deepcopy(result.records)
    getattr(records[0]["phys"], field)[0, 0] = np.nan

    checks = check_all(result.config, records, result.events)
    finite = next(
        check
        for check in checks
        if check.name == "physical_state_is_finite"
    )
    conservation = next(
        check
        for check in checks
        if check.name == "physical_byte_conservation"
    )

    assert not finite.passed
    assert field in finite.detail
    assert not conservation.passed


def test_hard_checks_reject_opposing_corrupt_physical_states(result):
    records = copy.deepcopy(result.records)
    phys = records[0]["phys"]
    phys.cum_output_physical = (
        phys.cum_output_physical
        + 1e18
        + 1e9
    )
    phys.cum_true_loss = phys.cum_true_loss - 1e18

    checks = check_all(result.config, records, result.events)
    conservation = next(
        check
        for check in checks
        if check.name == "physical_byte_conservation"
    )
    non_negative = next(
        check
        for check in checks
        if check.name == "physical_state_is_non_negative"
    )

    assert not conservation.passed
    assert not non_negative.passed


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


def test_generator_rejects_out_of_range_timestamps():
    with pytest.raises(ValueError, match="datetime64\\[ns\\]"):
        load_config(overrides={
            "time": {
                "start_timestamp": "3000-01-01T00:00:00Z",
                "duration_hours": 1.0,
            },
        })


def test_generator_rejects_nat_start_timestamp():
    with pytest.raises(ValueError, match="finite timestamp, not NaT"):
        load_config(overrides={
            "time": {"start_timestamp": "NaT"},
        })


def test_generator_rejects_partial_rate_bin():
    with pytest.raises(
        ValueError,
        match="361 raw samples.*complete 60-second rate bins",
    ):
        load_config(overrides={
            "time": {"duration_hours": 3610 / 3600},
        })


@pytest.mark.parametrize(
    ("time_overrides", "message"),
    (
        (
            {
                "raw_scrape_seconds": 90,
                "rate_window_seconds": 180,
            },
            "raw_scrape_seconds.*divide 60",
        ),
        (
            {
                "raw_scrape_seconds": 30,
                "rate_window_seconds": 120,
            },
            "rate_window_seconds must be exactly 60",
        ),
    ),
)
def test_generator_rejects_incompatible_minute_cadence(
    time_overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        load_config(overrides={"time": time_overrides})


def test_derived_schema(result):
    expected = {
        "timestamp", "consumer_id", "minute_index",
        "input_rate_bytes_per_min", "output_rate_bytes_per_min",
        "completeness_ratio", "completeness_ratio_1h", "static_alert",
        "backlog_bytes", "cum_lost_bytes",
        "is_true_loss", "is_benign_burst", "is_artifact", "label", "oracle_alert",
        "traj_alert",
        "archetype",
    }
    assert expected.issubset(set(result.derived.columns))


def test_raw_schema_includes_per_shard_hidden_state_when_enabled(result):
    assert result.config.output.include_hidden_state is True
    assert {"backlog_bytes", "cum_lost_bytes"}.issubset(result.raw.columns)


def test_events_have_expected_alert_semantics(result):
    ev = result.events
    if ev.empty:
        pytest.skip("no events generated in this tiny run")
    true_loss = ev[ev["type"].str.startswith("true_loss")]
    benign = ev[ev["type"].isin(["benign_burst", "artifact"])]
    assert (true_loss["expected_alert"] == True).all()      # noqa: E712
    assert (benign["expected_alert"] == False).all()        # noqa: E712


def test_priority_masks_exercise_cross_family_precedence(result):
    event_types = set(result.events["type"]) if not result.events.empty else set()
    has_true = any(value.startswith("true_loss") for value in event_types)
    has_benign = bool(event_types & {"benign_burst", "artifact"})
    if not (has_true and has_benign):
        pytest.skip("tiny schedule did not draw both priority families")
    flags = result.derived[[
        "is_true_loss",
        "is_benign_burst",
        "is_artifact",
    ]]

    assert (flags["is_true_loss"] & flags["is_benign_burst"]).any()
    assert (flags["is_true_loss"] & flags["is_artifact"]).any()
    assert (
        flags["is_benign_burst"]
        & flags["is_artifact"]
        & ~flags["is_true_loss"]
    ).any()


def test_benign_invariant_covers_artifact_subtype(result):
    check = next(
        item
        for item in check_all(
            result.config,
            result.records,
            result.events,
        )
        if item.name == "benign_events_do_not_lose_bytes"
    )

    assert check.passed
    assert "artifact:" in check.detail
    assert "artifact:0" not in check.detail


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
