"""First-class related-grain aggregation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from autogram.discovery.loop import build_dataframe_grammar
from autogram.discovery.known import KnownInvariant, _signature, shapes_for_invariant
from autogram.discovery.validate import (
    _runtime_relation_null,
    score_recovery,
)
from autogram.discovery import synth
from autogram.discovery.propose import EnumerationProposer, normalize_rule
from autogram.dsl import ast as A
from autogram.dsl.evaluate import _span_any, eval_term, typed_group_key
from autogram.dsl.parser import rule_from_dict, rule_to_dict
from autogram.dsl.typecheck import is_admissible
from autogram.loader.gtib import AUTOGRAM_PROFILE_ATTR, prepare_gtib, profile_dataframe
from autogram.loader.loader import Frame
from autogram.logic.solver import atom_expr
from autogram.schema.spec import (
    CellCodec,
    ColumnPattern,
    GrammarSpec,
    RelatedTemplate,
    RoleOntology,
)


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="relational",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("network",),
            ref_roles={"network": ()},
            fam_roles={"network": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"network": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )


def test_runtime_relation_null_preserves_declared_reset_domain():
    relation = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=8,
            freq="10s",
        ),
        "shard_id": ["s0"] * 8,
        "counter": np.arange(8, dtype=float) * 10.0,
        "reset_flag": [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
        ],
    })
    template = RelatedTemplate(
        binder="record",
        role="counter_delta",
        relation="raw",
        column="counter",
        mode="sum_delta",
        parent_keys=(),
        child_keys=(),
        partition_keys=("shard_id",),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
        reset_column="reset_flag",
        validity_columns=("counter",),
    )

    output = _runtime_relation_null(
        relation,
        [template],
        {},
        np.random.default_rng(0),
        definition_targets=False,
    )

    assert set(output["reset_flag"].tolist()) == {0.0, 1.0}
    assert len(set(output["counter"].tolist())) > 2


def test_runtime_relation_null_preserves_all_false_reset_support():
    relation = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=18, freq="10s"),
        "shard_id": ["s0"] * 18,
        "counter": np.arange(18, dtype=float) * 10.0,
        "reset_flag": [False] * 18,
    })
    template = RelatedTemplate(
        binder="record",
        role="counter_delta",
        relation="raw",
        column="counter",
        mode="sum_delta",
        parent_keys=(),
        child_keys=(),
        partition_keys=("shard_id",),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
        reset_column="reset_flag",
        validity_columns=("counter",),
    )

    output = _runtime_relation_null(
        relation,
        [template],
        {},
        np.random.default_rng(0),
        definition_targets=False,
    )

    assert not output["reset_flag"].astype(bool).any()
    counter = output["counter"].to_numpy(dtype=float)
    assert np.all(np.isfinite(counter))
    assert np.all(np.diff(counter) >= 0.0)


def test_mixed_span_and_delta_null_retains_both_relation_families():
    relation = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=12, freq="10s"),
        "consumer_id": ["c0"] * 12,
        "shard_id": ["s0"] * 12,
        "counter": np.arange(12, dtype=float),
        "reset_flag": [False] * 12,
        "type": ["event"] * 12,
        "span_start": pd.date_range("2026-01-01", periods=12, freq="10s"),
        "span_end": pd.date_range("2026-01-01 00:00:05", periods=12, freq="10s"),
    })
    span = RelatedTemplate(
        binder="record",
        role="event",
        relation="mixed",
        column="type",
        mode="span_any",
        parent_keys=("consumer_id",),
        child_keys=("consumer_id",),
        partition_keys=(),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
        span_start="span_start",
        span_end="span_end",
        filter_column="type",
        filter_values=("event",),
    )
    delta = RelatedTemplate(
        binder="record",
        role="counter_delta",
        relation="mixed",
        column="counter",
        mode="sum_delta",
        parent_keys=("consumer_id",),
        child_keys=("consumer_id",),
        partition_keys=("shard_id",),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
        reset_column="reset_flag",
        validity_columns=("counter",),
    )
    parent_context = {
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=3,
            freq="1min",
        ).to_numpy(),
        "consumer_id": np.array(["c0"] * 3, dtype=object),
    }

    output = _runtime_relation_null(
        relation,
        [span, delta],
        parent_context,
        np.random.default_rng(0),
        definition_targets=False,
    )

    assert int(np.count_nonzero(np.isfinite(
        pd.to_numeric(output["counter"], errors="coerce")
    ))) >= len(relation)
    assert output["span_start"].notna().any()
    assert len(output) > len(relation)


def _prepared() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=18, freq="10s")
    rows = []
    for shard, scale in (("shard_000_0", 1.0), ("shard_000_1", 2.0)):
        counter = np.cumsum(np.full(18, 10.0 * scale))
        output = np.cumsum(np.full(18, 8.0 * scale))
        for index, timestamp in enumerate(timestamps):
            rows.append({
                "timestamp": timestamp,
                "consumer_id": "consumer_000",
                "shard_id": shard,
                "collector_input_counted": counter[index],
                "presenter_output_counted": output[index],
                "missing_flag": False,
                "reset_flag": False,
                "backlog_bytes": float(index + (100 if shard.endswith("_1") else 0)),
                "cum_lost_bytes": float(2 * index + (200 if shard.endswith("_1") else 0)),
            })
    raw = pd.DataFrame(rows)
    derived = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
        "consumer_id": "consumer_000",
        "minute_index": [0, 1, 2],
        "input_rate_bytes_per_min": [np.nan, 180.0, 180.0],
        "output_rate_bytes_per_min": [np.nan, 144.0, 144.0],
        "backlog_bytes": [110.0, 122.0, 134.0],
        "cum_lost_bytes": [220.0, 244.0, 268.0],
    })
    return prepare_gtib(derived, raw)


def test_gtib_profile_declares_general_related_aggregates():
    profile = _prepared().attrs[AUTOGRAM_PROFILE_ATTR]
    related = profile["related_aggregates"]

    assert set(related) >= {
        "raw_input_rate",
        "raw_output_rate",
        "raw_backlog",
        "raw_cum_lost",
    }
    assert related["raw_input_rate"]["mode"] == "sum_delta"
    assert related["raw_backlog"]["mode"] == "sum_last"


def test_related_aggregate_round_trips_typechecks_and_is_enumerated():
    dataset, grammar = build_dataframe_grammar(_prepared(), _base_spec(), name="related")
    term = A.RelatedAgg("raw_input_rate")
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("input_rate_bytes_per_min"), "==", term),
    )

    assert rule_from_dict(rule_to_dict(rule)) == rule
    assert atom_expr(rule.atom, {}) is not None
    assert is_admissible(rule, grammar)[0] is True
    rendered = {candidate.unparse() for candidate in EnumerationProposer(grammar).propose()}
    assert normalize_rule(rule).unparse() in rendered


def test_related_aggregate_evaluates_raw_delta_and_boundary_joins():
    prepared = _prepared()
    materialized = prepared.drop(columns=[
        column
        for column in prepared.columns
        if column.endswith(("_input_increment", "_output_increment", "_backlog_bytes", "_cum_lost_bytes"))
        and column.startswith("shard_")
    ])
    materialized.attrs = prepared.attrs
    dataset, _grammar = build_dataframe_grammar(materialized, _base_spec(), name="related")
    binding = {}

    input_rate = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        binding,
        dataset.observed,
        dataset.name_model,
    )
    output_rate = eval_term(
        A.RelatedAgg("raw_output_rate"),
        "record",
        binding,
        dataset.observed,
        dataset.name_model,
    )
    backlog = eval_term(
        A.RelatedAgg("raw_backlog"),
        "record",
        binding,
        dataset.observed,
        dataset.name_model,
    )
    loss = eval_term(
        A.RelatedAgg("raw_cum_lost"),
        "record",
        binding,
        dataset.observed,
        dataset.name_model,
    )

    assert np.isnan(input_rate[0]) and np.isnan(output_rate[0])
    assert np.allclose(input_rate[1:], [180.0, 180.0])
    assert np.allclose(output_rate[1:], [144.0, 144.0])
    assert np.allclose(backlog, [110.0, 122.0, 134.0])
    assert np.allclose(loss, [220.0, 244.0, 268.0])


def test_related_delta_requires_adjacent_complete_partitions():
    prepared = _prepared()
    profile = prepared.attrs[AUTOGRAM_PROFILE_ATTR]
    raw = profile["related_frames"]["raw"].copy()
    missing = (
        (raw["shard_id"] == "shard_000_1")
        & (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )
    profile["related_frames"]["raw"] = raw.loc[~missing].copy()
    prepared.attrs[AUTOGRAM_PROFILE_ATTR] = profile
    dataset, _grammar = build_dataframe_grammar(
        prepared,
        _base_spec(),
        name="related_gap",
    )

    values = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert np.isnan(values).all()


def test_related_delta_recognizes_numeric_reset_flags():
    # A reset marker must be recognized regardless of dtype: a CSV round-trip, a parquet load, or
    # the runtime null control can present reset_flag as float {0.0, 1.0} rather than a native bool.
    # The delta aggregate must break at the numeric reset exactly as it does for a bool.
    from autogram.dsl.evaluate import _is_reset

    assert _is_reset(True) and _is_reset(1) and _is_reset(1.0) and _is_reset("true")
    assert not _is_reset(False) and not _is_reset(0) and not _is_reset(0.0)
    assert not _is_reset(float("nan"))

    prepared = _prepared()
    profile = prepared.attrs[AUTOGRAM_PROFILE_ATTR]
    raw = profile["related_frames"]["raw"].copy()
    reset = (
        (raw["shard_id"] == "shard_000_1")
        & (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )
    raw["reset_flag"] = 0.0
    raw.loc[reset, "reset_flag"] = 1.0
    profile["related_frames"]["raw"] = raw
    prepared.attrs[AUTOGRAM_PROFILE_ATTR] = profile
    dataset, _grammar = build_dataframe_grammar(
        prepared,
        _base_spec(),
        name="related_numeric_reset",
    )

    values = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert np.isnan(values[0])
    assert values[1] == 60.0
    assert values[2] == 180.0


def test_related_delta_sums_valid_shards_across_reset():
    prepared = _prepared()
    profile = prepared.attrs[AUTOGRAM_PROFILE_ATTR]
    raw = profile["related_frames"]["raw"].copy()
    reset = (
        (raw["shard_id"] == "shard_000_1")
        & (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )
    raw.loc[reset, "reset_flag"] = True
    profile["related_frames"]["raw"] = raw
    prepared.attrs[AUTOGRAM_PROFILE_ATTR] = profile
    dataset, _grammar = build_dataframe_grammar(
        prepared,
        _base_spec(),
        name="related_reset",
    )

    values = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert np.isnan(values[0])
    assert values[1] == 60.0
    assert values[2] == 180.0


def test_empty_event_filter_matches_no_spans():
    derived = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=2, freq="1min"),
        "consumer_id": ["consumer_000", "consumer_000"],
        "minute_index": [0, 1],
        "is_true_loss": [False, False],
    })
    events = pd.DataFrame({
        "consumer_id": ["consumer_000"],
        "type": ["benign_burst"],
        "span_start": [pd.Timestamp("2026-01-01 00:00:00")],
        "span_end": [pd.Timestamp("2026-01-01 00:02:00")],
    })
    prepared = prepare_gtib(derived, events=events)
    dataset, _grammar = build_dataframe_grammar(
        prepared,
        _base_spec(),
        name="empty_event_filter",
    )

    values = eval_term(
        A.RelatedAgg("event_true_loss"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert values.tolist() == [0.0, 0.0]


def test_span_filters_and_cache_keys_preserve_typed_identity():
    parent_times = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-01-01 00:02:00",
    ])
    frame = Frame(
        np.empty((2, 0), dtype=float),
        [],
        row_context={
            "timestamp": parent_times.to_numpy(),
            "consumer_id": np.array(["c", "c"], dtype=object),
        },
    )
    filter_values = np.empty(2, dtype=object)
    filter_values[0] = True
    filter_values[1] = 1
    child = pd.DataFrame({
        "consumer_id": ["c", "c"],
        "type": pd.Series(filter_values, dtype=object),
        "span_start": pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:02:00",
        ]),
        "span_end": pd.to_datetime([
            "2026-01-01 00:01:00",
            "2026-01-01 00:03:00",
        ]),
    })

    def template(value):
        return RelatedTemplate(
            binder="record",
            role=f"event_{type(value).__name__}",
            relation="events",
            column="",
            mode="span_any",
            parent_keys=("consumer_id",),
            child_keys=("consumer_id",),
            partition_keys=(),
            parent_time="timestamp",
            child_time="",
            window_seconds=60,
            span_start="span_start",
            span_end="span_end",
            filter_column="type",
            filter_values=(value,),
        )

    true_values = _span_any(template(True), frame, child)
    one_values = _span_any(template(1), frame, child)

    assert true_values is not None and one_values is not None
    assert true_values.tolist() == [1.0, 0.0]
    assert one_values.tolist() == [0.0, 1.0]


def test_span_join_preserves_datetime_child_key_identity():
    parent_time = pd.Timestamp("1970-01-01")
    frame = Frame(
        np.empty((1, 0), dtype=float),
        [],
        row_context={
            "timestamp": np.array(
                [parent_time.to_datetime64()],
            ),
            "consumer_id": np.array([1], dtype=object),
        },
    )
    child_key = np.empty(1, dtype=object)
    child_key[0] = np.datetime64(
        "1970-01-01T00:00:00.000000001",
        "ns",
    )
    child = pd.DataFrame({
        "consumer_id": child_key,
        "span_start": [parent_time],
        "span_end": [parent_time + pd.Timedelta("1s")],
    })
    template = RelatedTemplate(
        binder="record",
        role="event",
        relation="events",
        column="",
        mode="span_any",
        parent_keys=("consumer_id",),
        child_keys=("consumer_id",),
        partition_keys=(),
        parent_time="timestamp",
        child_time="",
        window_seconds=60,
        span_start="span_start",
        span_end="span_end",
    )

    values = _span_any(template, frame, child)

    assert values is not None
    assert values.tolist() == [0.0]


def test_span_join_rejects_timestamp_outside_nanosecond_range():
    frame = Frame(
        np.empty((1, 0), dtype=float),
        [],
        row_context={
            "timestamp": np.array(
                [np.datetime64("1970-01-01", "ns")],
            ),
        },
    )
    child = pd.DataFrame({
        "span_start": np.array(
            [np.datetime64("2554-01-01", "D")],
        ),
        "span_end": np.array(
            [np.datetime64("2554-01-02", "D")],
        ),
    })
    template = RelatedTemplate(
        binder="record",
        role="event",
        relation="events",
        column="",
        mode="span_any",
        parent_keys=(),
        child_keys=(),
        partition_keys=(),
        parent_time="timestamp",
        child_time="",
        window_seconds=60,
        span_start="span_start",
        span_end="span_end",
    )

    with pytest.raises(ValueError, match="datetime64\\[ns\\] range"):
        _span_any(template, frame, child)


def test_span_runtime_null_preserves_typed_consumers():
    consumers = np.empty(4, dtype=object)
    consumers[:2] = True
    consumers[2:] = 1
    times = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-01-01 00:01:00",
        "2026-01-01 00:00:00",
        "2026-01-01 00:01:00",
    ]).to_numpy()
    relation = pd.DataFrame({
        "consumer_id": pd.Series(dtype=object),
        "type": pd.Series(dtype=object),
        "span_start": pd.Series(dtype="datetime64[ns]"),
        "span_end": pd.Series(dtype="datetime64[ns]"),
    })
    template = RelatedTemplate(
        binder="record",
        role="event",
        relation="events",
        column="",
        mode="span_any",
        parent_keys=("consumer_id",),
        child_keys=("consumer_id",),
        partition_keys=(),
        parent_time="timestamp",
        child_time="",
        window_seconds=60,
        span_start="span_start",
        span_end="span_end",
        filter_column="type",
        filter_values=("event",),
    )

    generated = _runtime_relation_null(
        relation,
        [template],
        {
            "timestamp": times,
            "consumer_id": consumers,
        },
        np.random.default_rng(0),
        definition_targets=False,
    )

    identities = {
        typed_group_key(value)
        for value in generated["consumer_id"].tolist()
    }
    assert identities == {
        typed_group_key(True),
        typed_group_key(1),
    }
    assert len(generated) == 2


def test_related_aggregates_scale_to_multi_day_child_history():
    n_minutes = 2_160
    parent_times = pd.date_range("2026-01-01", periods=n_minutes, freq="1min")
    raw_rows = []
    for shard_index in range(3):
        input_counter = 0.0
        output_counter = 0.0
        for step, timestamp in enumerate(
            pd.date_range(
                parent_times[0],
                periods=n_minutes * 6,
                freq="10s",
            )
        ):
            input_counter += 10.0 * (shard_index + 1)
            output_counter += 8.0 * (shard_index + 1)
            raw_rows.append({
                "timestamp": timestamp,
                "consumer_id": "consumer_000",
                "shard_id": f"s{shard_index}",
                "collector_input_counted": input_counter,
                "presenter_output_counted": output_counter,
                "missing_flag": False,
                "reset_flag": False,
                "backlog_bytes": float(step + shard_index),
                "cum_lost_bytes": float(step * 2 + shard_index),
            })
    derived = pd.DataFrame({
        "timestamp": parent_times,
        "consumer_id": "consumer_000",
        "minute_index": np.arange(n_minutes),
        "input_rate_bytes_per_min": np.nan,
        "output_rate_bytes_per_min": np.nan,
        "backlog_bytes": np.nan,
        "cum_lost_bytes": np.nan,
    })
    prepared = prepare_gtib(derived, pd.DataFrame(raw_rows))
    materialized = prepared.drop(columns=[
        column
        for column in prepared.columns
        if column.startswith("s")
        and column.endswith((
            "_input_increment",
            "_output_increment",
            "_backlog_bytes",
            "_cum_lost_bytes",
        ))
    ])
    materialized.attrs = prepared.attrs
    dataset, _grammar = build_dataframe_grammar(
        materialized,
        _base_spec(),
        name="related_scale",
    )

    started = time.perf_counter()
    values = {
        role: eval_term(
            A.RelatedAgg(role),
            "record",
            {},
            dataset.observed,
            dataset.name_model,
        )
        for role in (
            "raw_input_rate",
            "raw_output_rate",
            "raw_backlog",
            "raw_cum_lost",
        )
    }
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert np.isnan(values["raw_input_rate"][0])
    assert np.allclose(values["raw_input_rate"][1:], 360.0)
    assert np.allclose(values["raw_output_rate"][1:], 288.0)


def test_related_aggregate_known_signature_and_recovery_field():
    known = KnownInvariant(
        "rate_from_raw",
        "==",
        "input_rate_bytes_per_min",
        {"related": "raw_input_rate"},
    )
    assert _signature(known) == (
        "equality",
        "exact",
        (
            "related_aggregate",
            ("input_rate_bytes_per_min", "raw_input_rate"),
        ),
    )
    assert shapes_for_invariant(known) == ["cross_grain"]

    from autogram.discovery import validate as validation
    result = type("Result", (), {"portfolio": []})()
    original = validation.portfolio_relations
    try:
        validation.portfolio_relations = lambda _result: {_signature(known)}
        recovery = score_recovery(
            result,
            {"cross_grain": {_signature(known)[2][1]}},
        )
    finally:
        validation.portfolio_relations = original
    assert recovery.cross_grain == 1.0


def test_cross_grain_proxy_generator_carries_related_frame_contract():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=40,
        noise=0.0,
        seed=2,
        families=("cross_grain",),
    )
    assert set(data.planted) == {"cross_grain"}
    assert data.planted["cross_grain"]
    assert "raw" in data.relations
    assert data.related_aggregates["proxy_raw_sum"]["mode"] == "sum_delta"


def test_related_level_sum_is_ungradeable_when_a_shard_reading_is_missing():
    """Round-25: a cross-grain *level* sum is all-or-nothing.

    Levels are not increments. Every shard's backlog exists whether or not the shard reported it,
    so a total that quietly omits one shard under-counts, and one that carries the shard's previous
    reading forward invents a measurement the shard never emitted. Both make the aggregate look
    healthy while being wrong. The only honest answer when a required shard has no reading at the
    boundary is that the row cannot be graded.

    (Increments behave differently and deliberately so -- see
    `test_related_delta_sums_valid_shards_across_reset`: a shard that reset has no measurable
    increment and the emitted per-minute value excludes it too.)
    """
    prepared = _prepared()
    profile = prepared.attrs[AUTOGRAM_PROFILE_ATTR]
    raw = profile["related_frames"]["raw"].copy()

    # Baseline: with every reading present the level sum is the sum of the two shards.
    dataset, _grammar = build_dataframe_grammar(prepared, _base_spec(), name="level_complete")
    baseline = eval_term(
        A.RelatedAgg("raw_backlog"), "record", {}, dataset.observed, dataset.name_model
    )
    assert baseline[1] == 122.0

    # Drop one shard's reading at the minute-1 boundary. The previous reading (116) must NOT be
    # carried forward, and the other shard's 6 must not be presented as the total either.
    boundary = (
        (raw["shard_id"] == "shard_000_1")
        & (raw["timestamp"] == pd.Timestamp("2026-01-01 00:01:50"))
    )
    assert bool(boundary.any())
    raw.loc[boundary, "backlog_bytes"] = np.nan
    profile["related_frames"]["raw"] = raw
    prepared.attrs[AUTOGRAM_PROFILE_ATTR] = profile
    dataset, _grammar = build_dataframe_grammar(prepared, _base_spec(), name="level_missing")

    values = eval_term(
        A.RelatedAgg("raw_backlog"), "record", {}, dataset.observed, dataset.name_model
    )

    assert np.isnan(values[1]), values[1]
    # Neighbouring minutes still have complete readings and are unaffected.
    assert values[2] == 134.0


def test_materialized_and_streaming_paths_agree_when_a_shard_minute_is_absent():
    """Round-26: the two implementations of a cross-grain sum must agree on missing coverage.

    A shard with no rows at all for one parent minute is a *coverage gap*, not a zero contribution.
    The streaming join already refuses to grade such a row. The materialised fast path used to leave
    a structural zero there, so the same law read as a family sum silently under-counted while
    `RELATED(...)` reported nothing -- two answers to one question.

    A minute the shard DID report but whose increment is unusable (a reset) stays a deliberate zero
    in both paths, because the emitted per-minute value excludes that shard too. This test pins both
    halves of that distinction.
    """
    prepared = _prepared()
    raw = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["related_frames"]["raw"].copy()
    derived = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
        "consumer_id": "consumer_000",
        "minute_index": [0, 1, 2],
        "input_rate_bytes_per_min": [np.nan, 180.0, 180.0],
        "output_rate_bytes_per_min": [np.nan, 144.0, 144.0],
        "backlog_bytes": [110.0, 122.0, 134.0],
        "cum_lost_bytes": [220.0, 244.0, 268.0],
    })

    # Delete every reading one shard has inside parent minute 1 -- a genuine coverage gap.
    gap = (
        (raw["shard_id"] == "shard_000_1")
        & (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )
    assert int(gap.sum()) > 0
    gapped = prepare_gtib(derived, raw.loc[~gap].reset_index(drop=True))

    # Fast path: the shard's own materialised columns report the gap rather than a zero.
    for column in ("shard_000_1_input_increment", "shard_000_1_output_increment"):
        values = pd.to_numeric(gapped[column], errors="coerce").to_numpy(dtype=float)
        assert np.isnan(values[1]), (column, values)

    # Streaming path: the same row is ungradeable.
    dataset, _grammar = build_dataframe_grammar(gapped, _base_spec(), name="coverage_gap")
    joined = eval_term(
        A.RelatedAgg("raw_input_rate"), "record", {}, dataset.observed, dataset.name_model
    )
    assert np.isnan(joined[1]), joined

    # A reported-but-reset minute is the other case: both paths treat it as a deliberate zero
    # contribution, so the parent row is still graded.
    reset = (
        (raw["shard_id"] == "shard_000_1")
        & (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )
    with_reset = raw.copy()
    with_reset.loc[reset, "reset_flag"] = True
    reset_frame = prepare_gtib(derived, with_reset)
    values = pd.to_numeric(
        reset_frame["shard_000_1_input_increment"], errors="coerce"
    ).to_numpy(dtype=float)
    assert values[1] == 0.0, values
    dataset, _grammar = build_dataframe_grammar(reset_frame, _base_spec(), name="reset_gap")
    joined = eval_term(
        A.RelatedAgg("raw_input_rate"), "record", {}, dataset.observed, dataset.name_model
    )
    assert joined[1] == 60.0, joined


def _family_sum(frame, columns):
    """The fast path's answer: sum the materialised shard columns, NaN if any is ungradeable."""
    total = np.zeros(len(frame))
    unknown = np.zeros(len(frame), dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        unknown |= np.isnan(values)
        total += np.nan_to_num(values)
    return np.where(unknown, np.nan, total)


def test_materialized_and_streaming_increments_agree_on_every_awkward_minute():
    """Round-27: the fast path must reproduce the join's coverage AND its any-valid rule.

    Three distinct situations used to be collapsed into a structural zero by the materialised path
    while the streaming join answered differently:

    * a minute with no adjacent prior boundary (the first minute, and the minute after a gap) has no
      measurable increment at all -- ungradeable;
    * a minute where one shard reset is a deliberate zero contribution from that shard, because the
      emitted per-minute value excludes it too -- the row is still graded;
    * a minute where *every* shard reset has no usable contributor at all, so the total carries no
      information -- ungradeable.
    """
    derived = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
        "consumer_id": "consumer_000",
        "minute_index": [0, 1, 2],
        "input_rate_bytes_per_min": [np.nan, 180.0, 180.0],
        "output_rate_bytes_per_min": [np.nan, 144.0, 144.0],
        "backlog_bytes": [110.0, 122.0, 134.0],
        "cum_lost_bytes": [220.0, 244.0, 268.0],
    })
    raw = _prepared().attrs[AUTOGRAM_PROFILE_ATTR]["related_frames"]["raw"].copy()
    minute_one = (
        (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )
    columns = ["shard_000_0_input_increment", "shard_000_1_input_increment"]

    def _both(mutated):
        frame = prepare_gtib(derived, mutated)
        dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="agreement")
        joined = eval_term(
            A.RelatedAgg("raw_input_rate"), "record", {}, dataset.observed, dataset.name_model
        )
        return np.asarray(joined, dtype=float), _family_sum(frame, columns)

    def _same(left, right):
        return all(
            (np.isnan(a) and np.isnan(b)) or a == b
            for a, b in zip(left, right)
        )

    joined, fast = _both(raw)
    assert _same(joined, fast), (joined, fast)
    assert np.isnan(joined[0]) and joined[1] == 180.0

    one_reset = raw.copy()
    one_reset.loc[minute_one & (one_reset["shard_id"] == "shard_000_1"), "reset_flag"] = True
    joined, fast = _both(one_reset)
    assert _same(joined, fast), (joined, fast)
    assert joined[1] == 60.0, joined

    all_reset = raw.copy()
    all_reset.loc[minute_one, "reset_flag"] = True
    joined, fast = _both(all_reset)
    assert _same(joined, fast), (joined, fast)
    assert np.isnan(joined[1]), joined

    # One shard loses a whole minute while the other keeps reporting. The surviving shard makes the
    # row non-barren, so the "at least one usable contributor" pass does not rescue it -- only the
    # per-shard coverage rule does. Both the gap minute and the minute *after* it (which now has no
    # adjacent prior boundary for the affected shard) must be ungradeable in both paths.
    dropped = raw.loc[
        ~(minute_one & (raw["shard_id"] == "shard_000_0"))
    ].reset_index(drop=True)
    joined, fast = _both(dropped)
    assert _same(joined, fast), (joined, fast)
    assert np.isnan(joined[1]) and np.isnan(joined[2]), joined



def test_related_delta_overflow_is_reported_not_absorbed_as_invalidity():
    """Round-30 review: a counter difference that blows up must not silently contribute zero.

    ``valid &= np.isfinite(delta)`` treats an overflowed increment exactly like a missing reading,
    so the shard drops out of the sum and the total looks complete while under-counting -- and a
    false cross-grain law is then accepted at hold rate and support 1.0. The overflow has to be
    tracked inside the aggregation, because it never reaches the output to be inferred from.
    """
    from autogram.dsl.evaluate import _related_aggregate

    timestamps = pd.date_range("2026-01-01", periods=18, freq="10s")
    frames = []
    for shard in ("s0", "s1"):
        counter = np.cumsum(np.full(18, 10.0))
        if shard == "s1":
            # Two finite readings whose difference exceeds float64 across the window.
            counter = np.where(np.arange(18) < 7, -1.5e308, 1.5e308)
        frames.append(pd.DataFrame({
            "timestamp": timestamps,
            "shard_id": shard,
            "counter": counter,
            "reset_flag": False,
        }))
    raw = pd.concat(frames, ignore_index=True)

    parent = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:01:00", periods=2, freq="1min"),
        "consumer_id": ["c0", "c0"],
        "total": [1.0, 1.0],
    })
    frame = profile_dataframe(
        parent,
        time_index="timestamp",
        group_keys=("consumer_id",),
        related_frames={"raw": raw},
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="overflowing_related")
    template = RelatedTemplate(
        binder="record",
        role="raw_counter_delta",
        relation="raw",
        column="counter",
        mode="sum_delta",
        parent_keys=(),
        child_keys=(),
        partition_keys=("shard_id",),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
        reset_column="reset_flag",
        validity_columns=(),
    )

    values, overflow = _related_aggregate(template, dataset.observed)

    assert overflow is not None
    assert bool(np.any(overflow)), "the blown-up shard was absorbed as mere invalidity"
    # The overflow is invisible in the output: the shard was dropped, so the total stayed finite.
    assert np.all(np.isfinite(values[~np.isnan(values)]))


def test_reset_interval_is_excluded_before_streaming_delta_arithmetic():
    from autogram.dsl.evaluate import _related_aggregate

    parent_time = pd.Timestamp("2026-01-01 00:01:00")
    raw = pd.DataFrame({
        "timestamp": [
            parent_time - pd.Timedelta("10s"),
            parent_time + pd.Timedelta("10s"),
        ] * 2,
        "shard_id": ["bad", "bad", "good", "good"],
        "counter": [-1.5e308, 1.5e308, 0.0, 10.0],
        "reset_flag": [False, True, False, False],
    })
    parent = pd.DataFrame({
        "timestamp": [parent_time],
        "consumer_id": ["c0"],
        "total": [10.0],
    })
    frame = profile_dataframe(
        parent,
        time_index="timestamp",
        group_keys=("consumer_id",),
        related_frames={"raw": raw},
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="reset_before_delta",
    )
    template = RelatedTemplate(
        binder="record",
        role="raw_counter_delta",
        relation="raw",
        column="counter",
        mode="sum_delta",
        parent_keys=(),
        child_keys=(),
        partition_keys=("shard_id",),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
        reset_column="reset_flag",
        validity_columns=(),
    )

    values, overflow = _related_aggregate(
        template,
        dataset.observed,
    )

    assert np.allclose(values, [10.0])
    assert overflow is None
