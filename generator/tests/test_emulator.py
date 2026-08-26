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

from gtib_emulator import cli
from gtib_emulator import deriver as deriver_module
from gtib_emulator import invariants as invariants_module
from gtib_emulator.config import load_config
from gtib_emulator.deriver import (
    _minute_counters,
    _sustained_below,
    derive_consumer,
    trajectory_alert,
)
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


def _fill_object_column(
    frame: pd.DataFrame,
    mask: np.ndarray,
    column: str,
    value,
) -> None:
    frame[column] = frame[column].astype(object)
    index = frame.index[np.asarray(mask, dtype=bool)]
    frame.loc[index, column] = pd.Series(
        [value] * len(index),
        index=index,
        dtype=object,
    )


def test_hard_invariants_pass(result):
    cfg = result.config
    results = check_all(
        cfg,
        result.records,
        result.events,
        result.raw,
    )
    hard_failures = [r for r in results if r.tier == "hard" and not r.passed]
    assert not hard_failures, [f"{r.name}: {r.detail}" for r in hard_failures]


@pytest.mark.parametrize("command", ("generate", "validate"))
def test_cli_validation_rejects_corrupted_raw(
    result,
    command,
    monkeypatch,
    capsys,
):
    corrupted = copy.copy(result)
    corrupted.raw = result.raw.iloc[1:].reset_index(drop=True)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda *_args, **_kwargs: result.config,
    )
    monkeypatch.setattr(cli, "run", lambda _cfg: corrupted)
    monkeypatch.setattr(cli, "write_outputs", lambda _result: {})

    exit_code = cli.main([command])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "raw_identity_grid" in captured.out
    assert "raw rows=" in captured.out


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


def test_hard_checks_require_derived_archetype_column(result):
    records = [
        {
            **record,
            "frame": record["frame"].copy(),
        }
        for record in result.records
    ]
    records[0]["frame"] = records[0]["frame"].drop(
        columns=["archetype"]
    )

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert "missing required archetype column" in grid.detail


@pytest.mark.parametrize("corruption", ("wrong", None, pd.NA))
def test_hard_checks_reject_corrupted_derived_archetype(
    result,
    corruption,
):
    records = [
        {
            **record,
            "frame": record["frame"].copy(),
        }
        for record in result.records
    ]
    frame = records[0]["frame"]
    frame["archetype"] = frame["archetype"].astype(object)
    frame.at[frame.index[0], "archetype"] = corruption

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert "archetype rows that do not match" in grid.detail


def test_derived_archetype_validation_is_typed_and_length_safe(result):
    records = copy.deepcopy(result.records)
    frame = records[0]["frame"].iloc[:-1].copy()
    expected = True
    frame["archetype"] = pd.Series(
        [expected] * len(frame),
        index=frame.index,
        dtype=object,
    )
    frame.at[frame.index[0], "archetype"] = 1
    records[0]["consumer"].archetype = expected
    records[0]["frame"] = frame

    checks = check_all(result.config, records, result.events)
    grid = next(
        check
        for check in checks
        if check.name == "derived_identity_grid"
    )

    assert not grid.passed
    assert "rows=" in grid.detail
    assert "archetype rows that do not match" in grid.detail


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


def test_identity_grids_preserve_nested_typed_consumers_and_shards(
    result,
):
    records = copy.deepcopy(result.records)
    raw = result.raw.copy()
    events = result.events.copy()
    identities = (
        ("region", ("consumer", True)),
        ("region", ("consumer", 1)),
        ("region", ("consumer", "1")),
        ("region", ("consumer", 1.0)),
    )
    for record, identity in zip(records, identities):
        old_consumer = record["consumer"].consumer_id
        raw_consumer = (
            raw["consumer_id"].to_numpy(dtype=object) == old_consumer
        )
        event_consumer = (
            events["consumer_id"].to_numpy(dtype=object) == old_consumer
            if not events.empty
            else np.zeros(0, dtype=bool)
        )
        old_shards = list(record["consumer"].shard_ids)
        typed_components = (True, 1, "1", 1.0)
        new_shards = [
            ("shard", typed_components[index])
            for index in range(len(old_shards))
        ]
        for old_shard, new_shard in zip(old_shards, new_shards):
            shard_mask = (
                raw_consumer
                & (
                    raw["shard_id"].to_numpy(dtype=object)
                    == old_shard
                )
            )
            _fill_object_column(
                raw,
                shard_mask,
                "shard_id",
                new_shard,
            )
        _fill_object_column(
            raw,
            raw_consumer,
            "consumer_id",
            identity,
        )
        if not events.empty:
            _fill_object_column(
                events,
                event_consumer,
                "consumer_id",
                identity,
            )
        record["consumer"].consumer_id = identity
        record["consumer"].shard_ids = new_shards
        frame = record["frame"]
        frame["consumer_id"] = pd.Series(
            [identity] * len(frame),
            index=frame.index,
            dtype=object,
        )

    checks = check_all(result.config, records, events, raw)
    grids = {
        check.name: check
        for check in checks
        if check.name in {"raw_identity_grid", "derived_identity_grid"}
    }

    assert grids["raw_identity_grid"].passed, grids[
        "raw_identity_grid"
    ].detail
    assert grids["derived_identity_grid"].passed, grids[
        "derived_identity_grid"
    ].detail


def test_hard_checks_reject_duplicate_typed_shard_ids(result):
    records = copy.deepcopy(result.records)
    raw = result.raw.copy()
    consumer = records[0]["consumer"]
    first_shard, duplicate_shard = consumer.shard_ids[:2]
    consumer.shard_ids[1] = first_shard
    duplicate_mask = (
        (raw["consumer_id"] == consumer.consumer_id)
        & (raw["shard_id"] == duplicate_shard)
    )
    raw.loc[duplicate_mask, "shard_id"] = first_shard

    assert raw.duplicated(
        ["timestamp", "consumer_id", "shard_id"]
    ).any()

    checks = check_all(result.config, records, result.events, raw)
    grid = next(
        check
        for check in checks
        if check.name == "raw_identity_grid"
    )

    assert not grid.passed
    assert "duplicate typed shard_id" in grid.detail
    assert "duplicate global (timestamp, consumer_id, shard_id)" in (
        grid.detail
    )

    source_checks = check_all(result.config, records, result.events)
    source_grid = next(
        check
        for check in source_checks
        if check.name == "raw_identity_grid"
    )
    assert not source_grid.passed
    assert "duplicate typed shard_id" in source_grid.detail


@pytest.mark.parametrize(
    ("corruption", "detail"),
    (
        ("missing_shard", "missing=1"),
        ("extra_shard", "extra=1"),
        ("missing_row", "raw rows="),
        ("extra_row", "duplicate global"),
        ("timestamp_cadence", "raw timestamps"),
        ("timestamp_order", "raw timestamps"),
        ("shard_order", "row order or identity alignment"),
    ),
)
def test_hard_checks_reject_malformed_raw_identity_grid(
    result,
    corruption,
    detail,
):
    raw = result.raw.copy()
    consumer = result.records[0]["consumer"]
    first_shard, second_shard = consumer.shard_ids[:2]
    first_shard_mask = (
        (raw["consumer_id"] == consumer.consumer_id)
        & (raw["shard_id"] == first_shard)
    )

    if corruption == "missing_shard":
        raw = raw.loc[~first_shard_mask].reset_index(drop=True)
    elif corruption == "extra_shard":
        extra = raw.loc[first_shard_mask].copy()
        extra["shard_id"] = "unexpected_shard"
        raw = pd.concat([raw, extra], ignore_index=True)
    elif corruption == "missing_row":
        raw = raw.iloc[1:].reset_index(drop=True)
    elif corruption == "extra_row":
        raw = pd.concat(
            [raw, raw.iloc[[0]].copy()],
            ignore_index=True,
        )
    elif corruption == "timestamp_cadence":
        raw.loc[raw.index[1], "timestamp"] += np.timedelta64(1, "s")
    elif corruption == "timestamp_order":
        timestamps = raw.loc[
            raw.index[:2],
            "timestamp",
        ].to_numpy(copy=True)
        raw.loc[raw.index[:2], "timestamp"] = timestamps[::-1]
    else:
        first = raw.index[
            first_shard_mask
            & (raw["timestamp"] == raw["timestamp"].iloc[0])
        ][0]
        second = raw.index[
            (raw["consumer_id"] == consumer.consumer_id)
            & (raw["shard_id"] == second_shard)
            & (raw["timestamp"] == raw["timestamp"].iloc[0])
        ][0]
        raw.loc[[first, second], "shard_id"] = [
            second_shard,
            first_shard,
        ]

    checks = check_all(
        result.config,
        result.records,
        result.events,
        raw,
    )
    grid = next(
        check
        for check in checks
        if check.name == "raw_identity_grid"
    )

    assert not grid.passed
    assert detail in grid.detail


@pytest.mark.parametrize(
    ("corruption", "detail"),
    (
        ("not_dataframe", "expected DataFrame"),
        ("missing_column", "missing identity columns"),
        ("missing_consumer", "missing or invalid typed"),
        ("malformed_timestamp", "missing or invalid typed"),
    ),
)
def test_malformed_raw_tables_fail_hard_without_raising(
    result,
    corruption,
    detail,
):
    raw = result.raw.copy()
    if corruption == "not_dataframe":
        malformed = {"timestamp": []}
    elif corruption == "missing_column":
        malformed = raw.drop(columns=["shard_id"])
    elif corruption == "missing_consumer":
        malformed = raw
        _fill_object_column(
            malformed,
            np.arange(len(malformed)) == 0,
            "consumer_id",
            ("region", ("consumer", pd.NA)),
        )
    else:
        malformed = raw
        malformed["timestamp"] = malformed["timestamp"].astype(object)
        malformed.loc[malformed.index[0], "timestamp"] = "not-a-timestamp"

    checks = check_all(
        result.config,
        result.records,
        result.events,
        malformed,
    )
    grid = next(
        check
        for check in checks
        if check.name == "raw_identity_grid"
    )

    assert grid.tier == "hard"
    assert not grid.passed
    assert detail in grid.detail


@pytest.mark.parametrize(
    "column",
    (
        "collector_input_counted",
        "presenter_output_counted",
        "missing_flag",
        "reset_flag",
        "backlog_bytes",
        "cum_lost_bytes",
    ),
)
def test_raw_schema_requires_every_emitted_payload_column(
    result,
    column,
):
    raw = result.raw.drop(columns=[column])

    checks = check_all(
        result.config,
        result.records,
        result.events,
        raw,
    )
    grid = next(
        check
        for check in checks
        if check.name == "raw_identity_grid"
    )

    assert not grid.passed
    assert "missing payload columns" in grid.detail
    assert column in grid.detail


@pytest.mark.parametrize(
    ("column", "corruption"),
    (
        ("collector_input_counted", "increment"),
        ("presenter_output_counted", "unexpected_nan"),
        ("collector_input_counted", "nan_as_none"),
        ("missing_flag", "toggle"),
        ("reset_flag", "toggle"),
        ("backlog_bytes", "increment"),
        ("cum_lost_bytes", "increment"),
    ),
)
def test_raw_payload_must_exactly_match_generator_records(
    result,
    column,
    corruption,
):
    raw = result.raw.copy()
    if corruption == "nan_as_none":
        index = raw.index[raw[column].isna()][0]
        raw[column] = raw[column].astype(object)
        raw.at[index, column] = None
    else:
        index = raw.index[raw[column].notna()][0]
        if corruption == "increment":
            raw.at[index, column] += 1.0
        elif corruption == "unexpected_nan":
            raw.at[index, column] = np.nan
        else:
            raw.at[index, column] = not bool(raw.at[index, column])

    checks = check_all(
        result.config,
        result.records,
        result.events,
        raw,
    )
    grid = next(
        check
        for check in checks
        if check.name == "raw_identity_grid"
    )

    assert not grid.passed
    assert f"raw payload column {column!r}" in grid.detail


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


@pytest.mark.parametrize(
    "duration",
    (0, -1, 1.5, True, "10"),
)
def test_generator_rejects_invalid_alert_duration(duration):
    with pytest.raises(
        ValueError,
        match="alerting.alert_duration_minutes must be a positive integer",
    ):
        load_config(overrides={
            "alerting": {"alert_duration_minutes": duration},
        })


@pytest.mark.parametrize("duration", (0, -1, 1.5, True))
def test_sustained_below_defensively_rejects_invalid_duration(
    duration,
):
    with pytest.raises(
        ValueError,
        match="duration_minutes must be a positive integer",
    ):
        _sustained_below(
            np.array([np.nan, 0.5, 0.5]),
            threshold=0.99,
            duration_minutes=duration,
        )


def test_missing_reset_scrape_clears_prior_counter_state():
    increments, valid = _minute_counters(
        np.array([[100.0, np.nan, 5.0]]),
        np.array([[False, True, False]]),
        spm=1,
    )

    assert not valid[0, 1]
    assert valid[0, 2]
    assert increments[0, 2] == 5.0


def test_missing_minute_requires_adjacent_trustworthy_boundaries():
    increments, valid = _minute_counters(
        np.array([[
            5.0,
            10.0,
            np.nan,
            np.nan,
            15.0,
            20.0,
            25.0,
            30.0,
        ]]),
        np.zeros((1, 8), dtype=bool),
        spm=2,
    )
    rates = np.where(valid, increments, np.nan)

    assert not valid[0, 1]
    assert np.isnan(rates[0, 1])
    assert not valid[0, 2]
    assert np.isnan(rates[0, 2])
    assert valid[0, 3]
    assert rates[0, 3] == 10.0


def test_derived_identity_oracle_rejects_legacy_gap_rates(
    result,
    monkeypatch,
):
    records = copy.deepcopy(result.records)
    record = records[0]
    obs = record["obs"]
    cfg = result.config
    spm = cfg.raw_steps_per_minute
    end = 4 * spm
    counter_pattern = np.concatenate((
        np.linspace(10.0, 60.0, spm),
        np.full(spm, np.nan),
        np.linspace(70.0, 120.0, spm),
        np.linspace(130.0, 180.0, spm),
    ))
    for counter in (obs.input_counted, obs.output_counted):
        counter[:, :end] = counter_pattern
    obs.missing_flag[:, :end] = False
    obs.missing_flag[:, spm:2 * spm] = True
    obs.reset_flag[:, :end] = False
    obs.active_flag[:, :end] = True

    def legacy_minute_counters(counter, reset_flag, steps_per_minute):
        n_shards, n_steps = counter.shape
        n_minutes = n_steps // steps_per_minute
        counter = counter[:, :n_minutes * steps_per_minute]
        reset_flag = reset_flag[:, :n_minutes * steps_per_minute]
        filled = counter.copy()
        for shard in range(n_shards):
            last = np.nan
            for step in range(filled.shape[1]):
                if reset_flag[shard, step]:
                    last = 0.0
                if np.isnan(filled[shard, step]):
                    filled[shard, step] = last
                else:
                    last = filled[shard, step]
        boundary = filled.reshape(
            n_shards,
            n_minutes,
            steps_per_minute,
        )[:, :, -1]
        increments = np.diff(
            boundary,
            axis=1,
            prepend=boundary[:, :1],
        )
        reset_in_minute = reset_flag.reshape(
            n_shards,
            n_minutes,
            steps_per_minute,
        ).any(axis=2)
        valid = (
            np.isfinite(increments)
            & (increments >= 0.0)
            & ~reset_in_minute
        )
        valid[:, 0] = False
        return increments, valid

    monkeypatch.setattr(
        deriver_module,
        "_minute_counters",
        legacy_minute_counters,
    )
    monkeypatch.setattr(
        invariants_module,
        "_minute_counters",
        legacy_minute_counters,
        raising=False,
    )
    labels = record["frame"][[
        "is_true_loss",
        "is_benign_burst",
        "is_artifact",
        "label",
        "oracle_alert",
    ]].reset_index(drop=True)
    frame = derive_consumer(
        cfg,
        record["consumer"],
        obs,
        record["phys"],
    )
    frame = pd.concat([frame, labels], axis=1)
    frame["traj_alert"] = trajectory_alert(cfg, frame)
    record["frame"] = frame

    assert frame.loc[1, "input_rate_bytes_per_min"] == 0.0
    assert np.isfinite(frame.loc[2, "input_rate_bytes_per_min"])
    checks = check_all(cfg, records, result.events)
    identities = next(
        check
        for check in checks
        if check.name == "derived_signal_identities"
    )

    assert not identities.passed


def test_derived_signal_identities_share_missing_reset_semantics(result):
    records = copy.deepcopy(result.records)
    record = records[0]
    obs = record["obs"]
    cfg = result.config
    spm = cfg.raw_steps_per_minute
    reset_index = 2 * spm - 1
    end = 3 * spm

    old_lifetime = np.linspace(100.0 / spm, 100.0, spm)
    reset_minute = np.linspace(110.0, 100.0 + 10.0 * spm, spm)
    reset_minute[-1] = np.nan
    new_lifetime = np.linspace(0.0, 5.0, spm)
    counter_pattern = np.concatenate((
        old_lifetime,
        reset_minute,
        new_lifetime,
    ))
    for counter in (obs.input_counted, obs.output_counted):
        counter[0, :end] = counter_pattern
    obs.missing_flag[0, :end] = False
    obs.missing_flag[0, reset_index] = True
    obs.reset_flag[0, :end] = False
    obs.reset_flag[0, reset_index] = True
    obs.active_flag[0, :end] = True

    labels = record["frame"][[
        "is_true_loss",
        "is_benign_burst",
        "is_artifact",
        "label",
        "oracle_alert",
    ]].reset_index(drop=True)
    frame = derive_consumer(
        cfg,
        record["consumer"],
        obs,
        record["phys"],
    )
    frame = pd.concat([frame, labels], axis=1)
    frame["traj_alert"] = trajectory_alert(cfg, frame)
    record["frame"] = frame

    checks = check_all(cfg, records, result.events)
    identities = next(
        check
        for check in checks
        if check.name == "derived_signal_identities"
    )
    assert identities.passed, identities.detail

    record["frame"].loc[
        2,
        "input_rate_bytes_per_min",
    ] += 1.0
    corrupted_checks = check_all(cfg, records, result.events)
    corrupted_identities = next(
        check
        for check in corrupted_checks
        if check.name == "derived_signal_identities"
    )
    assert not corrupted_identities.passed


def test_seed_zero_missing_reset_does_not_report_false_decrease():
    cfg = load_config(overrides={"seed": 0})
    generated = run(cfg)

    checks = check_all(cfg, generated.records, generated.events)
    monotone = next(
        check
        for check in checks
        if check.name == "observed_counters_monotone_between_resets"
    )

    assert monotone.passed, monotone.detail


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
