"""GTIB long-table ingestion and materialization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogram.cli import _load_dataframe
from autogram.config import DiscoveryConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.loop import build_dataframe_grammar
from autogram.dsl import ast as A
from autogram.dsl.evaluate import eval_term
from autogram.loader.gtib import AUTOGRAM_PROFILE_ATTR, infer_tabular_profile, prepare_gtib
from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="flat",
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


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2026-01-01", periods=18, freq="10s")
    rows = []
    for shard, scale in (("shard_000_0", 1.0), ("shard_000_1", 2.0)):
        counter = np.cumsum(np.full(18, 10.0 * scale))
        output = np.cumsum(np.full(18, 8.0 * scale))
        for i, ts in enumerate(timestamps):
            rows.append({
                "timestamp": ts,
                "consumer_id": "consumer_000",
                "shard_id": shard,
                "collector_input_counted": counter[i],
                "presenter_output_counted": output[i],
                "missing_flag": False,
                "reset_flag": False,
                "backlog_bytes": float(i + (100 if shard.endswith("_1") else 0)),
                "cum_lost_bytes": float(2 * i + (200 if shard.endswith("_1") else 0)),
            })
    raw = pd.DataFrame(rows)
    derived = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="1min"),
        "consumer_id": "consumer_000",
        "minute_index": [0, 1, 2],
        "input_rate_bytes_per_min": [np.nan, 180.0, 180.0],
        "output_rate_bytes_per_min": [np.nan, 144.0, 144.0],
        "completeness_ratio": [np.nan, 0.8, 0.8],
        "completeness_ratio_1h": [np.nan, np.nan, np.nan],
        "static_alert": [False, False, False],
        "backlog_bytes": [105.0, 117.0, 129.0],
        "cum_lost_bytes": [210.0, 234.0, 258.0],
        "is_true_loss": [False, True, True],
        "is_benign_burst": [False, False, False],
        "is_artifact": [False, False, False],
        "label": ["normal", "true_loss", "true_loss"],
        "oracle_alert": [False, True, True],
        "traj_alert": [False, False, True],
    })
    return derived, raw



def _increments(series):
    """Materialised increments with NaN rendered as ``None`` so lists compare readably."""
    return [None if pd.isna(value) else float(value) for value in series.tolist()]


def test_string_backed_reset_flags_do_not_spuriously_reset():
    # A reset-free frame whose reset_flag arrives as strings ("false") must materialize identical
    # increments to a Boolean frame: a naive .any() would treat the non-empty "false" as truthy and
    # zero out the boundary increment.
    from autogram.loader.gtib import _coerce_boolean_flag
    import pandas as pd

    coerced = _coerce_boolean_flag(pd.Series(["false", "true", "0", "1", ""]))
    assert coerced.tolist() == [False, True, False, True, pd.NA]
    # unrecognised / explicitly-missing tokens are NA, not silently False
    assert _coerce_boolean_flag(pd.Series(["nan", "none", "maybe"])).isna().all()
    # out-of-range numerics are NA rather than truthy
    assert _coerce_boolean_flag(pd.Series([0.0, 1.0, 2.0])).tolist() == [False, True, pd.NA]

    derived, raw = _tables()
    string_raw = raw.copy()
    string_raw["reset_flag"] = "false"
    string_raw["missing_flag"] = "false"

    frame = prepare_gtib(derived, string_raw)

    # The first minute has no adjacent prior boundary, so no increment is measurable there and the
    # cell is ungradeable -- the same answer the streaming join gives. What matters for this test is
    # that the later minutes are unaffected by the string-typed flags.
    assert _increments(frame["shard_000_0_input_increment"]) == [None, 60.0, 60.0]
    assert _increments(frame["shard_000_1_input_increment"]) == [None, 120.0, 120.0]


def test_derived_boolean_fields_are_coerced_from_strings():
    # Declared derived Boolean fields (is_true_loss, static_alert, ...) that arrive as strings must
    # still be recognized as Boolean condition/role columns after ingestion.
    import pandas as pd
    from autogram.loader.gtib import AUTOGRAM_PROFILE_ATTR

    derived, raw = _tables()
    derived = derived.copy()
    derived["is_true_loss"] = ["false", "true", "true"]
    derived["static_alert"] = ["0", "1", "0"]

    frame = prepare_gtib(derived, raw)

    assert pd.api.types.is_bool_dtype(frame["is_true_loss"]) or str(frame["is_true_loss"].dtype) == "boolean"
    assert frame["is_true_loss"].tolist() == [False, True, True]
    assert frame["static_alert"].tolist() == [False, True, False]


def test_prepare_gtib_materializes_exact_shard_increments_and_boundaries():
    derived, raw = _tables()

    frame = prepare_gtib(derived, raw)

    # A leading `None` is the first minute: with no adjacent prior boundary there is no measurable
    # increment, so the cell is ungradeable rather than a claim that the shard moved by zero. The
    # streaming join reports the same, and the two paths must agree.
    assert _increments(frame["shard_000_0_input_increment"]) == [None, 60.0, 60.0]
    assert _increments(frame["shard_000_1_input_increment"]) == [None, 120.0, 120.0]
    assert _increments(frame["shard_000_0_output_increment"]) == [None, 48.0, 48.0]
    assert _increments(frame["shard_000_1_output_increment"]) == [None, 96.0, 96.0]
    assert frame["shard_000_0_backlog_bytes"].tolist() == [5.0, 11.0, 17.0]
    assert frame["shard_000_1_backlog_bytes"].tolist() == [105.0, 111.0, 117.0]
    assert frame["shard_000_0_cum_lost_bytes"].tolist() == [10.0, 22.0, 34.0]
    assert frame["shard_000_1_cum_lost_bytes"].tolist() == [210.0, 222.0, 234.0]
    assert np.allclose(
        frame[["shard_000_0_input_increment", "shard_000_1_input_increment"]].sum(axis=1)[1:],
        frame["input_rate_bytes_per_min"].iloc[1:],
    )


def test_prepare_gtib_records_time_groups_conditions_families_and_raw_relation():
    derived, raw = _tables()

    frame = prepare_gtib(derived, raw)
    profile = frame.attrs[AUTOGRAM_PROFILE_ATTR]

    assert profile["time_index"] == "timestamp"
    assert profile["group_keys"] == ["consumer_id"]
    assert set(profile["condition_columns"]) >= {
        "label",
        "is_true_loss",
        "is_benign_burst",
        "is_artifact",
        "static_alert",
        "oracle_alert",
        "traj_alert",
    }
    assert set(profile["families"]["shard_input_increment"]) == {
        "shard_000_0_input_increment",
        "shard_000_1_input_increment",
    }
    assert "raw" in profile["related_frames"]
    assert len(profile["related_frames"]["raw"]) == len(raw)


def test_prepare_gtib_qualifies_shard_ids_reused_across_consumers():
    derived, raw = _tables()
    raw = raw.loc[raw["shard_id"] == "shard_000_0"].copy()
    raw.loc[:, "shard_id"] = "shared"
    other_raw = raw.copy()
    other_raw.loc[:, "consumer_id"] = "consumer_001"
    other_raw.loc[:, ["collector_input_counted", "presenter_output_counted"]] *= 2.0
    other_derived = derived.copy()
    other_derived.loc[:, "consumer_id"] = "consumer_001"
    combined = prepare_gtib(
        pd.concat([derived, other_derived], ignore_index=True),
        pd.concat([raw, other_raw], ignore_index=True),
    )

    first = "consumer_000__shared_input_increment"
    second = "consumer_001__shared_input_increment"
    family = combined.attrs[AUTOGRAM_PROFILE_ATTR]["families"]["shard_input_increment"]
    assert family == [first, second]
    # Within a shard's OWN consumer the first minute is ungradeable (no prior boundary); rows
    # belonging to the other consumer keep a structural 0.0, because the shard genuinely
    # contributes nothing there. Collapsing those two cases together is what made the fast path
    # disagree with the streaming join.
    assert _increments(combined[first]) == [None, 60.0, 60.0, 0.0, 0.0, 0.0]
    assert _increments(combined[second]) == [0.0, 0.0, 0.0, None, 120.0, 120.0]


@pytest.mark.parametrize(
    "first_shard,second_shard",
    [
        (True, 1),
        (1, "1"),
    ],
)
def test_materialized_and_streaming_joins_preserve_typed_shard_identity(
    first_shard,
    second_shard,
):
    """Typed-distinct shard IDs must not merge in materialization or in generated names.

    pandas ``groupby`` merges ``True`` with ``1`` before a key is exposed, while ``str``-keyed
    lookups and names merge ``1`` with ``"1"``. Either collapse under-counts one path or overwrites
    a materialized column, making the fast path disagree with the streaming related join.
    """
    derived, raw = _tables()
    shard_ids = np.empty(len(raw), dtype=object)
    first = raw["shard_id"] == "shard_000_0"
    shard_ids[first.to_numpy()] = first_shard
    shard_ids[~first.to_numpy()] = second_shard
    raw = raw.copy()
    raw["shard_id"] = pd.Series(shard_ids, dtype=object)
    assert {type(value) for value in raw["shard_id"].tolist()} == {
        type(first_shard),
        type(second_shard),
    }

    prepared = prepare_gtib(derived, raw)
    family = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["families"][
        "shard_input_increment"
    ]

    assert len(family) == len(set(family)) == 2
    materialized = prepared[family].sum(
        axis=1,
        min_count=len(family),
    ).to_numpy(dtype=float)

    streaming_frame = prepared.drop(columns=family)
    streaming_frame.attrs = prepared.attrs
    dataset, _grammar = build_dataframe_grammar(
        streaming_frame,
        _base_spec(),
        name="typed_shard_streaming",
    )
    streaming = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert streaming is not None
    assert np.array_equal(materialized, streaming, equal_nan=True)
    assert np.allclose(streaming[1:], [180.0, 180.0])


def test_materialized_names_reserve_literal_ids_and_existing_columns():
    """Qualified typed names may not steal a literal string ID or overwrite source data."""
    derived, raw = _tables()
    first = raw["shard_id"] == "shard_000_0"
    integer = raw.loc[first].copy()
    integer["shard_id"] = pd.Series(
        np.full(len(integer), 1, dtype=object),
        index=integer.index,
        dtype=object,
    )
    string_one = raw.loc[~first].copy()
    string_one["shard_id"] = pd.Series(
        np.full(len(string_one), "1", dtype=object),
        index=string_one.index,
        dtype=object,
    )
    literal = raw.loc[first].copy()
    literal["shard_id"] = "int:1"
    literal[["collector_input_counted", "presenter_output_counted"]] *= 3.0
    raw = pd.concat([integer, string_one, literal], ignore_index=True)
    derived = derived.copy()
    derived["int:1_input_increment"] = 777.0
    derived["input_rate_bytes_per_min"] = [np.nan, 360.0, 360.0]

    prepared = prepare_gtib(derived, raw)
    family = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["families"][
        "shard_input_increment"
    ]

    assert prepared["int:1_input_increment"].tolist() == [777.0] * 3
    assert "int:1_input_increment" not in family
    assert len(family) == len(set(family)) == 3
    assert any(name.startswith("int:1_input_increment#") for name in family)
    total = prepared[family].sum(axis=1, min_count=3).to_numpy(dtype=float)
    assert np.allclose(total[1:], [360.0, 360.0])


def test_secondary_name_suffix_does_not_rename_an_ordinary_shard():
    """Collision suffix allocation must reserve every unique preferred spelling up front."""
    derived, raw = _tables()
    base_raw = raw.loc[raw["shard_id"] == "shard_000_0"].copy()
    raw_parts = []
    for consumer, shard in (
        ("c1", "s"),
        ("c2", "s"),
        ("z0", "c1__s"),
        ("z0", "c1__s#2"),
    ):
        part = base_raw.copy()
        part["consumer_id"] = consumer
        part["shard_id"] = shard
        if shard == "c1__s#2":
            part[
                ["collector_input_counted", "presenter_output_counted"]
            ] *= 4.0
        raw_parts.append(part)
    derived_parts = []
    for consumer in ("c1", "c2", "z0"):
        part = derived.copy()
        part["consumer_id"] = consumer
        derived_parts.append(part)

    prepared = prepare_gtib(
        pd.concat(derived_parts, ignore_index=True),
        pd.concat(raw_parts, ignore_index=True),
    )
    family = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["families"][
        "shard_input_increment"
    ]

    assert "c1__s#2_input_increment" in family
    assert len(family) == len(set(family)) == 4
    assert _increments(prepared["c1__s#2_input_increment"]) == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        240.0,
        240.0,
    ]


def test_missing_consumer_identity_agrees_between_materialized_and_streaming():
    """``pd.NA`` and ``None`` denote the same missing consumer on both join paths."""
    derived, raw = _tables()
    derived = derived.copy()
    raw = raw.copy()
    derived["consumer_id"] = pd.Series(
        [pd.NA] * len(derived),
        dtype=object,
    )
    raw["consumer_id"] = pd.Series(
        [None] * len(raw),
        dtype=object,
    )

    prepared = prepare_gtib(derived, raw)
    family = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["families"][
        "shard_input_increment"
    ]
    materialized = prepared[family].sum(
        axis=1,
        min_count=len(family),
    ).to_numpy(dtype=float)
    streaming_frame = prepared.drop(columns=family)
    streaming_frame.attrs = prepared.attrs
    dataset, _grammar = build_dataframe_grammar(
        streaming_frame,
        _base_spec(),
        name="missing_consumer_streaming",
    )
    streaming = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert streaming is not None
    assert np.array_equal(materialized, streaming, equal_nan=True)


def test_derived_only_consumer_is_ungradeable_on_all_materialized_families():
    derived, raw = _tables()
    extra = derived.copy()
    extra["consumer_id"] = "derived_only"
    combined = prepare_gtib(
        pd.concat([derived, extra], ignore_index=True),
        raw,
    )
    profile = combined.attrs[AUTOGRAM_PROFILE_ATTR]

    for family in profile["families"].values():
        values = combined[family].iloc[len(derived):]
        assert values.isna().all().all(), family

    input_family = profile["families"]["shard_input_increment"]
    streaming_frame = combined.drop(columns=input_family)
    streaming_frame.attrs = combined.attrs
    dataset, _grammar = build_dataframe_grammar(
        streaming_frame,
        _base_spec(),
        name="derived_only_streaming",
    )
    streaming = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )
    materialized = combined[input_family].sum(
        axis=1,
        min_count=len(input_family),
    ).to_numpy(dtype=float)

    assert streaming is not None
    assert np.array_equal(materialized, streaming, equal_nan=True)


def test_materialization_sorts_scalar_and_composite_consumer_ids():
    """Mixed scalar/composite typed IDs must have mutually comparable deterministic sort keys."""
    derived, raw = _tables()
    raw = raw.loc[raw["shard_id"] == "shard_000_0"].copy()
    first_raw = raw.copy()
    first_raw["consumer_id"] = pd.Series(
        [1] * len(first_raw),
        dtype=object,
    )
    second_raw = raw.copy()
    second_raw["consumer_id"] = pd.Series(
        [(1, 2)] * len(second_raw),
        dtype=object,
    )
    first_derived = derived.copy()
    first_derived["consumer_id"] = pd.Series(
        [1] * len(first_derived),
        dtype=object,
    )
    second_derived = derived.copy()
    second_derived["consumer_id"] = pd.Series(
        [(1, 2)] * len(second_derived),
        dtype=object,
    )

    prepared = prepare_gtib(
        pd.concat([first_derived, second_derived], ignore_index=True),
        pd.concat([first_raw, second_raw], ignore_index=True),
    )

    family = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["families"][
        "shard_input_increment"
    ]
    assert len(family) == len(set(family)) == 2


def test_materialization_fails_loudly_on_finite_counter_subtraction_overflow():
    """A finite subtraction overflow cannot be hidden as an invalid zero contribution."""
    derived, raw = _tables()
    raw = raw.copy()
    shard = raw["shard_id"] == "shard_000_0"
    minute_zero = raw["timestamp"] < pd.Timestamp("2026-01-01 00:01:00")
    raw.loc[shard & minute_zero, "collector_input_counted"] = -1.5e308
    raw.loc[shard & ~minute_zero, "collector_input_counted"] = 1.5e308

    with pytest.raises(ValueError, match="counter subtraction overflowed"):
        prepare_gtib(derived, raw)


def test_materialization_preserves_original_minute_indices_after_slice():
    derived, raw = _tables()
    sliced = derived.loc[derived["minute_index"].isin([1, 2])].copy()

    frame = prepare_gtib(sliced, raw)

    assert frame["shard_000_0_input_increment"].tolist() == [60.0, 60.0]
    assert frame["shard_000_1_input_increment"].tolist() == [120.0, 120.0]


def test_materialized_windows_keep_a_non_aligned_derived_origin():
    """Materialized minute windows must be the same intervals the streaming join evaluates."""
    derived, raw = _tables()
    derived = derived.copy()
    derived["timestamp"] += pd.Timedelta(seconds=30)
    derived["input_rate_bytes_per_min"] = [180.0, 180.0, 180.0]

    prepared = prepare_gtib(derived, raw)
    family = prepared.attrs[AUTOGRAM_PROFILE_ATTR]["families"][
        "shard_input_increment"
    ]
    materialized = prepared[family].sum(
        axis=1,
        min_count=len(family),
    ).to_numpy(dtype=float)

    streaming_frame = prepared.drop(columns=family)
    streaming_frame.attrs = prepared.attrs
    dataset, _grammar = build_dataframe_grammar(
        streaming_frame,
        _base_spec(),
        name="non_aligned_streaming",
    )
    streaming = eval_term(
        A.RelatedAgg("raw_input_rate"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert streaming is not None
    assert np.array_equal(materialized, streaming, equal_nan=True)
    assert np.allclose(streaming, [180.0, 180.0, 90.0])


def test_materialization_does_not_bridge_missing_raw_minute():
    derived, raw = _tables()
    missing = (
        (raw["shard_id"] == "shard_000_0")
        & (raw["timestamp"] >= pd.Timestamp("2026-01-01 00:01:00"))
        & (raw["timestamp"] < pd.Timestamp("2026-01-01 00:02:00"))
    )

    frame = prepare_gtib(derived, raw.loc[~missing].copy())

    values = frame["shard_000_0_input_increment"].tolist()
    # The gap must not be bridged: no increment may span the absent minute. Round-26 strengthened
    # the answer from a structural zero to NaN, so the absent minute is *ungradeable* rather than a
    # claim that the shard contributed nothing. Round-27 extended that to every minute with no
    # adjacent prior boundary -- the first minute, and the minute right after the gap -- which is
    # exactly what the streaming join reports. A zero on any of these would silently under-count a
    # family sum while the join refuses to grade the same row.
    assert _increments(pd.Series(values)) == [None, None, None]


def test_load_dataframe_accepts_csv_and_auto_prepares_gtib(tmp_path):
    derived, raw = _tables()
    derived_path = tmp_path / "timeseries_derived.csv"
    raw_path = tmp_path / "timeseries_raw.csv"
    derived.to_csv(derived_path, index=False)
    raw.to_csv(raw_path, index=False)

    loaded = _load_dataframe(str(derived_path))

    assert "shard_000_0_input_increment" in loaded
    assert loaded.attrs[AUTOGRAM_PROFILE_ATTR]["time_index"] == "timestamp"


def test_generic_profile_does_not_group_by_unique_identifier():
    unique = infer_tabular_profile(pd.DataFrame({
        "request_id": [f"r{index}" for index in range(20)],
        "x": np.arange(20.0),
    }))
    repeated = infer_tabular_profile(pd.DataFrame({
        "tenant_id": np.repeat(["a", "b"], 10),
        "x": np.arange(20.0),
    }))

    assert unique.attrs[AUTOGRAM_PROFILE_ATTR]["group_keys"] == []
    assert repeated.attrs[AUTOGRAM_PROFILE_ATTR]["group_keys"] == ["tenant_id"]


def test_generic_profile_infers_typed_distinct_groups_and_enforces_their_gate():
    n = 400
    consumers = np.empty(n, dtype=object)
    consumers[: n // 2] = True
    consumers[n // 2 :] = 1
    x = np.ones(n)
    y = np.ones(n)
    # Pooled hold rate 0.90 (acceptable); integer-group hold rate 0.80 (not acceptable).
    failing = np.arange(n // 2, n // 2 + n // 10)
    y[failing] = 2.0
    frame = infer_tabular_profile(pd.DataFrame({
        "consumer_id": pd.Series(consumers, dtype=object),
        "x": x,
        "y": y,
    }))

    assert frame.attrs[AUTOGRAM_PROFILE_ATTR]["group_keys"] == ["consumer_id"]
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="typed_inferred_groups",
    )
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.85,
            band_mode="global",
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("x"), "~=", A.Ref("y")))
    )

    assert not result.accepted
    assert len(result.parameters["group_hold_rates"]) == 2


def test_profiled_gtib_builds_record_grammar_and_row_context():
    derived, raw = _tables()
    frame = prepare_gtib(derived, raw)
    base = GrammarSpec(
        name="flat",
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
        noisy_kind="tabular",
        demand_kind="tabular",
    )

    dataset, grammar = build_dataframe_grammar(frame, base, name="gtib")

    assert "record" in grammar.binders
    assert set(grammar.refs_for("record")) >= {
        "input_rate_bytes_per_min",
        "output_rate_bytes_per_min",
        "completeness_ratio",
        "static_alert",
    }
    assert set(grammar.fams_for("record")) >= {
        "shard_input_increment",
        "shard_output_increment",
    }
    assert dataset.time_index == "timestamp"
    assert dataset.group_keys == ("consumer_id",)
    assert dataset.row_context["label"].tolist() == ["normal", "true_loss", "true_loss"]
    assert "label" not in dataset.observed.names
    assert "minute_index" not in dataset.observed.names
    assert "minute_index" not in grammar.refs_for("record")
    assert not any(
        role.startswith("shard_")
        for role in grammar.refs_for("record")
    )


def test_high_cardinality_identifier_is_not_inferred_as_a_condition():
    """Round-27: an ordinary CSV carrying an identifier column must still load.

    Condition inference accepted any object-typed column, so a `request_id` with one distinct value
    per row was proposed as a condition and the schema compiler then rejected it for exceeding the
    64-value condition ceiling -- the file could not be ingested at all. A condition names a
    *regime*, so its domain has to be small and repeated.
    """
    from autogram.loader.gtib import infer_tabular_profile
    from autogram.schema.compiler import _MAX_CONDITION_DOMAIN

    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "request_id": [f"req-{index}" for index in range(n)],
        "region": np.where(np.arange(n) % 2 == 0, "eu", "us"),
        "latency_ms": np.linspace(1.0, 100.0, n),
    })

    prepared = infer_tabular_profile(df)
    conditions = list(prepared.attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"])

    assert "request_id" not in conditions
    # A genuine low-cardinality regime label is still picked up.
    assert "region" in conditions
    for column in conditions:
        assert int(df[column].nunique(dropna=True)) <= _MAX_CONDITION_DOMAIN

    # A domain that is bounded but still above the compiler's ceiling must also be refused, even
    # though it repeats often enough not to look like an identifier. This is the case the
    # near-unique heuristic alone would let through.
    wide = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=1000, freq="1min"),
        "bucket": [f"b-{index % 70}" for index in range(1000)],
        "latency_ms": np.linspace(1.0, 100.0, 1000),
    })
    assert int(wide["bucket"].nunique()) > _MAX_CONDITION_DOMAIN
    assert int(wide["bucket"].nunique()) <= len(wide) // 2
    wide_conditions = list(
        infer_tabular_profile(wide).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )
    assert "bucket" not in wide_conditions



def test_identifier_repeated_twice_is_not_inferred_as_a_condition():
    """Round-29 / TODO-4: a bounded, repeating domain is still not a regime.

    50 distinct values over 100 rows clears both round-27 guards -- 50 is under the compiler's
    64-value ceiling, and 50 is not strictly greater than ``len(frame) // 2`` -- yet it is an
    identifier that happens to repeat twice. Expanding it produced a quarter of a million
    conditions. What is actually wanted is meaningful per-value support.
    """
    from autogram.loader.gtib import _min_condition_value_rows, infer_tabular_profile
    from autogram.schema.compiler import _MAX_CONDITION_DOMAIN

    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "session_id2": [f"s-{index % 50}" for index in range(n)],
        "latency_ms": np.linspace(1.0, 100.0, n),
    })
    distinct = int(df["session_id2"].nunique())
    assert distinct == 50
    assert distinct <= _MAX_CONDITION_DOMAIN            # passes the domain-ceiling guard
    assert not distinct > max(1, len(df) // 2)          # passes the near-unique guard

    conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "session_id2" not in conditions
    assert _min_condition_value_rows(n) > 2


def test_genuine_regime_label_is_still_inferred_as_a_condition():
    from autogram.loader.gtib import infer_tabular_profile

    n = 600
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "mode": np.resize(np.array(["steady", "burst", "drain"], dtype=object), n),
        "latency_ms": np.linspace(1.0, 100.0, n),
    })

    conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "mode" in conditions


def test_condition_value_floor_tracks_the_evaluator_support_floor():
    from autogram.config import DiscoveryConfig
    from autogram.loader.gtib import _min_condition_value_rows

    floor = DiscoveryConfig()
    assert _min_condition_value_rows(100) == int(floor.min_condition_points)
    # Once the table is large enough, the fractional floor dominates the absolute one.
    assert _min_condition_value_rows(100_000) == int(
        floor.min_condition_fraction * 100_000
    )


def test_conditioned_search_space_is_counted_before_expansion():
    """The ceiling must fire on a pre-count, not after the blow-up has been materialised.

    The message names both factors -- the number of conditionable rules and the number of
    conditions -- which is only knowable before expansion; the old check could only report that a
    running tally had crossed the ceiling, after paying for every variant built up to that point.
    """
    import re

    import pytest

    from autogram.discovery.propose import EnumerationProposer, SearchSpaceTruncatedError
    from autogram.dsl.grammar import Grammar

    grammar = Grammar(
        binders=("record",),
        ops=("~=", "=="),
        ref_roles={"record": ("x", "y", "z")},
        fam_roles={"record": ()},
        max_complexity=10,
        conditional_enabled=True,
        condition_columns={"regime": ("a", "b", "c", "d")},
        max_conditioned_rules=2,
    )

    with pytest.raises(SearchSpaceTruncatedError) as excinfo:
        EnumerationProposer(grammar).propose()

    message = str(excinfo.value)
    match = re.search(
        r"expand at least (\d+) conditionable rules over (\d+) conditions = "
        r"(\d+) conditioned candidates",
        message,
    )
    assert match, message
    conditionable, n_conditions, projected = (int(group) for group in match.groups())
    assert conditionable > 0 and n_conditions > 0
    assert projected == conditionable * n_conditions
    assert projected > 2


def test_conditioned_precount_fails_fast_without_consuming_the_whole_stream():
    """The pre-count must stream, not materialise.

    Round-29 review: a pre-count that first builds ``list(self._candidate_rules())`` trades one
    memory blow-up for another and cannot fail fast -- it consumes the entire raw search before
    refusing. The count has to raise the moment the running product crosses the ceiling.
    """
    import pytest

    from autogram.discovery.propose import EnumerationProposer, SearchSpaceTruncatedError
    from autogram.dsl import ast as A
    from autogram.dsl.grammar import Grammar

    grammar = Grammar(
        binders=("record",),
        ops=("~=",),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        max_complexity=10,
        conditional_enabled=True,
        condition_columns={"regime": ("a", "b")},
        max_conditioned_rules=1,
    )
    proposer = EnumerationProposer(grammar)
    produced = {"n": 0}

    def _endless():
        while True:
            produced["n"] += 1
            yield A.Rule("record", A.Compare(A.Ref("x"), "~=", A.Ref("y")))

    proposer._candidate_rules = _endless

    with pytest.raises(SearchSpaceTruncatedError):
        proposer.propose()

    # A materialising pre-count would never return at all on an endless stream; a streaming one
    # refuses after a couple of candidates.
    assert produced["n"] <= 4


def test_one_rare_value_does_not_disable_a_valid_regime_column():
    """Round-29 review: `counts.min()` discarded a whole column over a single rare label.

    ``{normal: 80, alert: 39, unknown: 1}`` is a regime label with two well-populated strata. The
    thin stratum's own conditioned rules are rejected downstream by the evaluator's support floor,
    which is where that decision belongs -- discarding the column throws away the other two.
    """
    from autogram.loader.gtib import infer_tabular_profile

    labels = np.array(["normal"] * 80 + ["alert"] * 39 + ["unknown"], dtype=object)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=labels.size, freq="1min"),
        "label": labels,
        "latency_ms": np.linspace(1.0, 100.0, labels.size),
    })

    conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "label" in conditions


def test_regime_test_uses_the_supplied_discovery_config():
    """The floor must track the configuration a run will actually use, not a frozen default."""
    from autogram.config import DiscoveryConfig
    from autogram.loader.gtib import infer_tabular_profile

    n = 60
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "bucket": [f"b-{index % 10}" for index in range(n)],   # 6 rows per value
        "latency_ms": np.linspace(1.0, 100.0, n),
    })

    default_conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )
    relaxed_conditions = list(
        infer_tabular_profile(
            df, DiscoveryConfig(min_condition_points=5, min_condition_fraction=0.0),
        ).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "bucket" not in default_conditions      # 6 rows per value is below the default floor
    assert "bucket" in relaxed_conditions          # ... but clears an explicitly lowered one


def test_skewed_regime_with_several_small_strata_is_still_inferred():
    """Round-30 review: a dominant stratum beside several small ones is still a regime.

    ``{normal: 50, r0..r4: 10}`` over 100 rows was rejected by a domain-size shape test, although
    `normal` alone supplies fifty gradeable conditioned rows.
    """
    from autogram.loader.gtib import infer_tabular_profile

    labels = np.array(["normal"] * 50 + sum(([f"r{i}"] * 10 for i in range(5)), []), dtype=object)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=labels.size, freq="1min"),
        "mode": labels,
        "latency_ms": np.linspace(1.0, 100.0, labels.size),
    })

    conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "mode" in conditions


def test_a_single_fat_value_beside_many_thin_ones_is_not_a_regime():
    """A column that only *describes* a corner of the table is not a regime label."""
    from autogram.loader.gtib import infer_tabular_profile

    n = 600
    labels = np.array(
        ["hot"] * 200 + [f"id-{index}" for index in range(400)], dtype=object,
    )
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "tag": labels,
        "latency_ms": np.linspace(1.0, 100.0, n),
    })

    conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "tag" not in conditions


def test_identifier_tail_beside_a_fat_stratum_is_not_a_regime():
    """Round-31 review: one fat stratum does not license a long identifier tail.

    ``{common: 50, id-0..id-49: 1}`` clears both the "some value is gradeable" and the "eligible
    values cover most of the table" tests, yet its fifty singletons are an identifier and expanding
    them projects over a million conditioned variants.
    """
    from autogram.loader.gtib import infer_tabular_profile

    labels = np.array(["common"] * 50 + [f"id-{index}" for index in range(50)], dtype=object)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=labels.size, freq="1min"),
        "tag": labels,
        "latency_ms": np.linspace(1.0, 100.0, labels.size),
    })

    conditions = list(
        infer_tabular_profile(df).attrs[AUTOGRAM_PROFILE_ATTR]["condition_columns"]
    )

    assert "tag" not in conditions


def test_conditioned_band_definitions_are_counted_against_the_ceiling():
    """Round-31 review: conditioned band definitions bypassed `max_conditioned_rules` entirely."""
    import pytest

    from autogram.discovery.propose import EnumerationProposer, SearchSpaceTruncatedError
    from autogram.dsl.grammar import Grammar

    # A single ref role, so there is no conditionable *comparison* to trip the ceiling: the only
    # conditioned candidates are the band definitions themselves.
    grammar = Grammar(
        binders=("record",),
        ops=("~=",),
        ref_roles={"record": ("x",)},
        fam_roles={"record": ()},
        max_complexity=10,
        band_enabled=True,
        conditional_enabled=True,
        condition_columns={"regime": ("a", "b", "c")},
        max_conditioned_rules=1,
    )

    with pytest.raises(SearchSpaceTruncatedError, match="conditioned candidates"):
        EnumerationProposer(grammar).propose()


def test_condition_precount_includes_cross_column_conjunctions():
    """The pre-count must cover every condition shape it will later materialise."""
    import pytest

    from autogram.discovery.propose import EnumerationProposer, SearchSpaceTruncatedError
    from autogram.dsl.grammar import Grammar

    # Two categorical columns of 1000 values each: the per-column equalities alone are only 2000,
    # but their cross-column conjunctions are ~2,000,000 -- past the trusted ceiling.
    columns = {
        "left": tuple(f"l{index}" for index in range(1000)),
        "right": tuple(f"r{index}" for index in range(1000)),
    }
    grammar = Grammar(
        binders=("record",),
        ops=("~=",),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        max_complexity=10,
        conditional_enabled=True,
        condition_columns=columns,
        max_condition_values=1,
    )

    with pytest.raises(SearchSpaceTruncatedError, match="condition grammar would enumerate"):
        EnumerationProposer(grammar).propose()


def test_condition_precount_matches_what_is_actually_generated():
    """Round-32 review: the ceiling must not refuse a grammar it could actually enumerate.

    Only pairs from DIFFERENT columns become conjunctions, so counting `C(total_values, 2)`
    overcounted by every same-column pair -- 1,000,405 projected against 956,038 generated, which
    rejected a grammar that fits under the trusted ceiling.
    """
    from autogram.discovery.propose import EnumerationProposer
    from autogram.dsl.grammar import Grammar

    columns = {
        f"c{index}": tuple(f"v{value}" for value in range(64))
        for index in range(22)
    }
    columns["small"] = tuple(f"s{value}" for value in range(6))
    grammar = Grammar(
        binders=("record",),
        ops=("~=",),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        max_complexity=10,
        conditional_enabled=True,
        condition_columns=columns,
        max_condition_values=1,
    )

    # Does not raise, and the count it would have refused on is the count it really produces.
    generated = EnumerationProposer(grammar)._conditions()

    values = [len(v) for v in columns.values()]
    simple_pairs = sum(
        values[i] * values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    assert len(generated) == sum(values) + simple_pairs
