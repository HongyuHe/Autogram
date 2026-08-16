"""Grouped lag, difference, and rolling-window temporal relations."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.calibrate import _capability_tiers, _widen_spec
from autogram.discovery import synth
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.known import KnownInvariant, _signature, recover_known, shapes_for_invariant
from autogram.discovery.loop import (
    build_dataframe_grammar,
    prepare_dataframe,
)
from autogram.discovery.induce import SchemaInducer
from autogram.discovery.propose import EnumerationProposer, normalize_rule
from autogram.discovery.validate import score_recovery
from autogram.dsl import ast as A
from autogram.dsl.evaluate import (
    _consecutive_window_ends,
    _datetime_ns,
    _row_group_keys,
    eval_term,
    typed_group_key,
)
from autogram.dsl.grammar import Grammar
from autogram.dsl.parser import rule_from_dict, rule_to_dict
from autogram.dsl.typecheck import is_admissible
from autogram.loader.gtib import profile_dataframe
from autogram.logic.solver import atom_expr
from autogram.schema.spec import (
    CellCodec,
    ColumnPattern,
    GrammarSpec,
    RefTemplate,
    RoleOntology,
)


def _base_spec(max_degree: int = 1) -> GrammarSpec:
    return GrammarSpec(
        name="temporal",
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
        max_degree=max_degree,
    )


def test_lag_known_recovery_resolves_structured_binder_role_and_binding():
    n = 80
    frame = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=n,
            freq="1min",
        ),
        "series_id": ["s"] * n,
        "metric_n0": np.arange(1.0, n + 1.0),
        "metric_n1": np.arange(2.0, n + 2.0),
    })
    spec = GrammarSpec(
        name="structured-temporal",
        patterns=(
            ColumnPattern(
                name="metric",
                matcher="regex",
                kind="measurement",
                direction="value",
                regex=r"^metric_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("measurement",)},
            fam_roles={"node": ()},
        ),
        ref_templates=(
            RefTemplate("node", "measurement", "metric_{X}"),
        ),
        family_selectors=(),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
        temporal_enabled=True,
        max_lag=2,
        time_index="timestamp",
        group_keys=("series_id",),
        metadata_columns=("timestamp", "series_id"),
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        spec,
        name="structured_lag_recovery",
    )
    atomic = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(
        A.Rule(
            "node",
            A.Compare(A.Ref("measurement"), ">=", A.Const(0)),
        )
    )
    result = SimpleNamespace(dataset=dataset, portfolio=[atomic])

    recovered = recover_known(
        result,
        [
            KnownInvariant(
                "lag",
                ">=",
                {"lag": ["metric_n0", 1]},
                0,
            ),
        ],
    )

    assert recovered["recovered"] == 1


def _profile(
    frame: pd.DataFrame,
    windows=(3,),
    max_lag=3,
    max_degree=1,
) -> pd.DataFrame:
    return profile_dataframe(
        frame,
        time_index="timestamp",
        group_keys=("series_id",),
        temporal_windows=windows,
        max_lag=max_lag,
        max_degree=max_degree,
    )


def test_temporal_typed_identity_does_not_collide_with_integer_zero():
    integer = typed_group_key(0)
    numpy_integer = typed_group_key(np.int64(0))
    instant = typed_group_key(
        np.datetime64("1970-01-01", "ns")
    )
    duration = typed_group_key(np.timedelta64(0, "ns"))

    assert integer == numpy_integer
    assert len({integer, instant, duration}) == 3
    assert (
        typed_group_key(np.datetime64("NaT", "ns"))
        == typed_group_key(None)
    )
    assert (
        typed_group_key(np.timedelta64("NaT", "ns"))
        == typed_group_key(None)
    )


def test_temporal_group_array_preserves_temporal_scalars():
    instant = typed_group_key(
        np.datetime64("1970-01-01", "ns")
    )
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": np.array(
                [
                    "1970-01-01T00:00:00.000000000",
                    "1970-01-01T00:00:00.000000001",
                ],
                dtype="datetime64[ns]",
            ),
            "x": [1.0, 2.0],
        }),
        group_keys=("group_id",),
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="temporal_group_identity",
    )
    labels = _row_group_keys(
        dataset.observed,
        dataset.name_model,
        np.arange(2),
    )
    assert labels is not None
    assert typed_group_key(labels[0]) == instant


def test_nat_timestamp_cannot_complete_a_temporal_window():
    cadence = 60 * 1_000_000_000
    maximum = np.iinfo(np.int64).max
    timestamps = np.array(
        [
            maximum - 2 * cadence + 1,
            maximum - cadence + 1,
            np.iinfo(np.int64).min,
        ],
        dtype=np.int64,
    ).view("datetime64[ns]")
    frame = _profile(pd.DataFrame({
        "timestamp": timestamps,
        "series_id": "a",
        "x": [1.0, 2.0, 3.0],
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="nat_temporal_window",
    )

    consecutive = _consecutive_window_ends(
        dataset.observed,
        dataset.name_model,
        np.arange(3),
        2,
    )
    lagged = eval_term(
        A.Lag(A.Ref("x"), 1),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert consecutive.tolist() == [False, True, False]
    assert np.isnan(lagged[2])


def test_temporal_cadence_rejects_wide_unit_timestamp_wrap():
    timestamps = np.array(
        [
            "2262-04-10",
            "2262-04-11",
            "2262-04-12",
            "2262-04-13",
        ],
        dtype="datetime64[s]",
    )
    frame = _profile(pd.DataFrame({
        "timestamp": timestamps,
        "series_id": "a",
        "x": np.arange(4.0),
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="wide_temporal_cadence",
    )

    with pytest.raises(ValueError, match="datetime64\\[ns\\] range"):
        eval_term(
            A.Lag(A.Ref("x"), 1),
            "record",
            {},
            dataset.observed,
            dataset.name_model,
        )


def test_checked_datetime_parser_accepts_numpy_string_scalars():
    parsed = _datetime_ns(np.array(
        ["Jan 01 2026", "01/02/2026"],
        dtype=str,
    ))

    assert [
        pd.Timestamp(int(value))
        for value in parsed
    ] == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]


def test_mixed_timestamp_formats_use_one_chronological_order():
    groups = 100
    frame = _profile(pd.DataFrame({
        "timestamp": np.tile(
            ["Jan 01 2026", "Jan 02 2026", "01/03/2026"],
            groups,
        ),
        "series_id": np.repeat(
            [f"series-{index}" for index in range(groups)],
            3,
        ),
        "x": np.tile([10.0, 20.0, -100.0], groups),
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="mixed_timestamp_formats",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(
            A.Diff(A.Ref("x"), 1),
            ">=",
            A.Const(0),
        ),
    ))

    assert not result.accepted
    assert result.hold_rate == pytest.approx(0.5)


def test_composite_group_keys_survive_stratified_subsampling():
    # Round-22: a composite group key must be bucketed as a 1-D object array of TUPLES. Building it
    # as a 2-D array made ``tolist()`` yield unhashable lists and crashed group-stratified
    # subsampling with ``TypeError: unhashable type: 'list'``.
    n = 400
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "consumer_id": np.repeat(["c0", "c1"], n // 2),
        "shard_id": np.tile(np.repeat(["s0", "s1"], n // 4), 2),
        "x": np.arange(float(n)),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("consumer_id", "shard_id"),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="composite_groups")
    assert dataset.group_keys == ("consumer_id", "shard_id")
    rule = A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))

    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(subsample=100, seed=0, band_mode="global"),
    ).evaluate(rule)

    assert evaluation.n_points > 0
    assert evaluation.accepted


def test_subsampling_does_not_hide_a_failing_group():
    # Round-21 soundness: a grouped universal law that fails on one group must be rejected even under
    # aggressive subsampling. Pooled random subsampling could drop the small failing group entirely
    # and accept the law; group-stratified subsampling keeps every group represented.
    good = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=1000, freq="1min"),
        "series_id": ["good"] * 1000,
        "x": np.arange(1000.0),  # strictly increasing -> DELTA_1(x) >= 0 holds
    })
    bad = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="1min"),
        "series_id": ["bad"] * 6,
        "x": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],  # strictly decreasing -> DELTA_1(x) >= 0 fails
    })
    frame = _profile(pd.concat([good, bad], ignore_index=True), windows=(2,), max_lag=2)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="subsample_group")
    rule = A.Rule("record", A.Compare(A.Diff(A.Ref("x"), 1), ">=", A.Const(0)))

    full = DataOnlyEvaluator(
        dataset, DiscoveryConfig(hold_rate_threshold=0.9, band_mode="global"),
    ).evaluate(rule)
    sampled = DataOnlyEvaluator(
        dataset, DiscoveryConfig(hold_rate_threshold=0.9, band_mode="global", subsample=30, seed=1),
    ).evaluate(rule)
    # Rejected without sampling, and STILL rejected with a small subsample (the failing group is
    # never dropped).
    assert not full.accepted
    assert not sampled.accepted


def test_raw_exact_sign_is_computed_before_subsampling():
    # Round-21 soundness: a lone violation must keep raw_exact_sign False even if subsampling could
    # drop it. x is non-negative except one row, so ``x >= 0`` is NOT tolerance-free exact; the flag
    # (read off the full population) must stay False so the archive never suppresses a valid lag.
    values = np.ones(500)
    values[123] = -50.0
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=500, freq="1min"),
        "series_id": ["a"] * 500,
        "x": values,
    }), windows=(2,), max_lag=2)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="subsample_exact")
    rule = A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))
    for seed in range(6):
        evaluation = DataOnlyEvaluator(
            dataset,
            DiscoveryConfig(hold_rate_threshold=0.5, band_mode="global", subsample=20, seed=seed),
        ).evaluate(rule)
        assert evaluation.raw_exact_sign is False


def test_temporal_terms_round_trip_and_are_solver_screenable():
    terms = (
        A.Lag(A.Ref("x"), 3),
        A.Diff(A.Ref("x"), 2),
        A.Rolling(A.Ref("x"), 5, "SUM"),
    )
    for term in terms:
        rule = A.Rule("record", A.Compare(term, ">=", A.Const(0)))
        assert rule_from_dict(rule_to_dict(rule)) == rule
        assert atom_expr(rule.atom, {}) is not None


@pytest.mark.parametrize(
    "term",
    [
        {
            "k": "Lag",
            "steps": -1,
            "term": {"k": "Ref", "role": "x"},
        },
        {
            "k": "Diff",
            "steps": 0,
            "term": {"k": "Ref", "role": "x"},
        },
        {
            "k": "Rolling",
            "window": 0,
            "kind": "SUM",
            "term": {"k": "Ref", "role": "x"},
        },
        {
            "k": "Rolling",
            "window": 2,
            "kind": "BOGUS",
            "term": {"k": "Ref", "role": "x"},
        },
    ],
)
def test_rule_deserialization_rejects_invalid_temporal_terms(term):
    payload = {
        "binder": "record",
        "op": "==",
        "left": term,
        "right": {"k": "Ref", "role": "y"},
    }

    with pytest.raises(ValueError):
        rule_from_dict(payload)


def test_rule_deserialization_can_enforce_full_grammar_admissibility():
    grammar = Grammar(
        binders=("record",),
        ops=("==",),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
    )
    payload = {
        "binder": "record",
        "op": "==",
        "left": {"k": "Ref", "role": "unknown"},
        "right": {"k": "Ref", "role": "y"},
    }

    with pytest.raises(ValueError, match="inadmissible"):
        rule_from_dict(payload, grammar=grammar)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "binder": "record",
            "op": "==",
            "left": {"k": "Const", "value": float("nan")},
            "right": {"k": "Ref", "role": "y"},
        },
        {
            "binder": "record",
            "atom_kind": "CategoryDefinition",
            "target_column": "label",
            "cases": [],
            "default": "normal",
        },
    ],
)
def test_rule_deserialization_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        rule_from_dict(payload)


def test_single_row_rolling_terms_normalize_to_their_input():
    rule = A.Rule(
        "record",
        A.Compare(
            A.Rolling(A.Ref("x"), 1, "SUM"),
            "==",
            A.Ref("x"),
        ),
    )

    normalized = normalize_rule(rule)

    assert normalized.atom.left == A.Ref("x")
    assert normalized.atom.right == A.Ref("x")


def test_search_config_overrides_temporal_grammar_bounds():
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=8, freq="1min"),
        "series_id": ["a"] * 8,
        "x": np.arange(8, dtype=float),
    }))
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        search_cfg=SearchConfig(max_lag=2, windows=(2,)),
        name="search_bounds",
    )

    assert grammar.max_lag == 2
    assert grammar.windows == (2,)


def test_proposer_enumerates_every_lag_within_the_grammar_bound():
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=12, freq="1min"),
        "series_id": ["a"] * 12,
        "x": np.arange(12, dtype=float),
    }), windows=(3,), max_lag=3)
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="all_lags",
    )

    rendered = {
        rule.unparse()
        for rule in EnumerationProposer(grammar).propose()
    }

    for steps in range(1, grammar.max_lag + 1):
        target = normalize_rule(A.Rule(
            "record",
            A.Compare(
                A.Lag(A.Ref("x"), steps),
                ">=",
                A.Const(0),
            ),
        ))
        assert target.unparse() in rendered


def test_proposer_enumerates_rolling_equalities_and_strict_lag_bounds():
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=12,
            freq="1min",
        ),
        "series_id": ["a"] * 12,
        "x": np.arange(12, dtype=float),
        "y": np.arange(12, dtype=float),
    }), windows=(3,), max_lag=3)
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="temporal_forms",
    )
    expected = (
        A.Rule(
            "record",
            A.Compare(
                A.Rolling(A.Ref("x"), 3, "SUM"),
                "==",
                A.Ref("y"),
            ),
        ),
        A.Rule(
            "record",
            A.Compare(
                A.Lag(A.Ref("x"), 2),
                "<",
                A.Const(0),
            ),
        ),
    )

    rendered = {
        rule.unparse()
        for rule in EnumerationProposer(grammar).propose()
    }

    assert all(
        is_admissible(rule, grammar)[0]
        and normalize_rule(rule).unparse() in rendered
        for rule in expected
    )


def test_profile_capabilities_union_with_widened_spec():
    frame = profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="1min"),
            "series_id": ["a"] * 8,
            "x": np.arange(8, dtype=float),
        }),
        time_index="timestamp",
        group_keys=("series_id",),
        temporal_windows=(5,),
        run_lengths=(5,),
        advanced=True,
        agg_kinds=("SUM",),
    )
    spec = replace(
        _base_spec(),
        ontology=replace(
            _base_spec().ontology,
            agg_kinds=("AVG", "MAX"),
        ),
        temporal_enabled=True,
        max_lag=10,
        windows=(2, 10),
        advanced_enabled=True,
        run_lengths=(2,),
    )

    _dataset, base_grammar = build_dataframe_grammar(
        frame,
        spec,
        name="profile_union",
    )
    _dataset, widened_grammar = build_dataframe_grammar(
        frame,
        _widen_spec(
            spec,
            all_aggs=True,
            temporal=True,
            max_lag=10,
            windows=(2, 10),
            advanced=True,
            run_lengths=(2,),
        ),
        name="profile_union_widened",
    )

    assert base_grammar.agg_kinds == ("SUM",)
    assert base_grammar.windows == (5,)
    assert base_grammar.run_lengths == (5,)
    assert set(widened_grammar.agg_kinds) == {
        "SUM",
        "AVG",
        "MIN",
        "MAX",
    }
    assert widened_grammar.windows == (2, 5, 10)
    assert widened_grammar.run_lengths == (2, 5)


def test_profile_capability_limits_override_incidental_induction():
    frame = profile_dataframe(
        pd.DataFrame({
            "x": np.arange(8, dtype=float),
            "flag": np.resize([False, True], 8),
        }),
        condition_columns=("flag",),
        advanced=False,
        max_conjunction_terms=2,
        band_enabled=False,
        max_degree=1,
        proportional=False,
    )
    base = _base_spec(max_degree=2)
    induced = replace(
        base,
        ontology=replace(
            base.ontology,
            ops=(*base.ontology.ops, "~\u221d"),
        ),
        advanced_enabled=True,
        max_conjunction_terms=4,
        band_enabled=True,
    )

    _dataset, grammar = build_dataframe_grammar(
        frame,
        induced,
        name="profile_authority",
    )
    widened = _widen_spec(
        induced,
        max_degree=2,
        proportional=True,
        advanced=True,
        max_conjunction_terms=4,
    )
    _dataset, widened_grammar = build_dataframe_grammar(
        frame,
        widened,
        name="profile_authority_widened",
    )

    assert not grammar.advanced_enabled
    assert not grammar.band_enabled
    assert grammar.max_degree == 1
    assert grammar.max_conjunction_terms == 2
    assert "~\u221d" not in grammar.ops
    assert widened_grammar.advanced_enabled
    assert widened_grammar.max_degree == 2
    assert widened_grammar.max_conjunction_terms == 4
    assert "~\u221d" in widened_grammar.ops


def test_prepare_dataframe_returns_runtime_spec_and_profile_metadata_wins():
    frame = profile_dataframe(
        pd.DataFrame({
            "x": np.arange(8, dtype=float),
            "identifier": np.arange(8, dtype=int),
        }),
        metadata_columns=("identifier",),
    )
    induced = replace(
        _base_spec(),
        metadata_columns=("x",),
    )

    class FixedInducer(SchemaInducer):
        def induce(self, columns, sample_rows=None):
            return induced

    dataset, grammar, runtime_spec = prepare_dataframe(
        frame,
        inducer=FixedInducer(),
        name="runtime_spec",
    )

    assert runtime_spec.ontology.binders == ("record",)
    assert runtime_spec.metadata_columns == ("identifier",)
    assert "x" in dataset.observed.names
    assert "identifier" not in dataset.observed.names
    assert grammar.binders == ("record",)


def test_grouped_lag_and_difference_never_cross_group_boundaries():
    frame = _profile(pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:00",
            "2026-01-01 00:01:00",
            "2026-01-01 00:01:00",
            "2026-01-01 00:02:00",
            "2026-01-01 00:02:00",
        ]),
        "series_id": ["a", "b", "a", "b", "a", "b"],
        "x": [1.0, 100.0, 3.0, 90.0, 6.0, 70.0],
    }), max_degree=2)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="grouped")
    binding = {}

    lag = eval_term(A.Lag(A.Ref("x"), 1), "record", binding, dataset.observed, dataset.name_model)
    delta = eval_term(A.Diff(A.Ref("x"), 1), "record", binding, dataset.observed, dataset.name_model)

    assert np.isnan(lag[:2]).all()
    assert np.allclose(lag[2:], [1.0, 100.0, 3.0, 90.0])
    assert np.allclose(delta[2:], [2.0, -10.0, 3.0, -20.0])


def test_temporal_terms_are_invalid_across_timestamp_gaps():
    frame = _profile(pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:01:00",
            "2026-01-01 00:10:00",
        ]),
        "series_id": ["a", "a", "a"],
        "x": [1.0, 2.0, 3.0],
    }), windows=(3,), max_lag=3)
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="temporal_gap",
    )

    lag = eval_term(
        A.Lag(A.Ref("x"), 1),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )
    rolling = eval_term(
        A.Rolling(A.Ref("x"), 3, "SUM"),
        "record",
        {},
        dataset.observed,
        dataset.name_model,
    )

    assert np.isnan(lag[2])
    assert np.isnan(rolling[2])


def test_rolling_ratio_and_monotonicity_are_accepted():
    n = 80
    numerator = np.arange(1.0, n + 1.0)
    denominator = np.arange(11.0, n + 11.0)
    rolling_num = pd.Series(numerator).rolling(3, min_periods=3).sum()
    rolling_den = pd.Series(denominator).rolling(3, min_periods=3).sum()
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "numerator": numerator,
        "denominator": denominator,
        "windowed_ratio": rolling_num / rolling_den,
        "monotone": np.cumsum(np.arange(1.0, n + 1.0)),
    }), max_degree=2)
    dataset, grammar = build_dataframe_grammar(frame, _base_spec(max_degree=2), name="temporal")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=1e-10, hold_rate_threshold=0.9, band_mode="global"),
    )
    ratio_rule = A.Rule(
        "record",
        A.Compare(
            A.Ref("windowed_ratio"),
            "==",
            A.Div(
                A.Rolling(A.Ref("numerator"), 3, "SUM"),
                A.Rolling(A.Ref("denominator"), 3, "SUM"),
            ),
        ),
    )
    monotone_rule = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("monotone"), 1), ">=", A.Const(0)),
    )

    assert is_admissible(ratio_rule, grammar)[0] is True
    assert is_admissible(monotone_rule, grammar)[0] is True
    assert evaluator.evaluate(ratio_rule).accepted
    assert evaluator.evaluate(monotone_rule).accepted
    rendered = {rule.unparse() for rule in EnumerationProposer(grammar).propose()}
    assert normalize_rule(ratio_rule).unparse() in rendered
    assert normalize_rule(monotone_rule).unparse() in rendered


def test_time_shuffled_monotone_control_is_rejected():
    rng = np.random.default_rng(4)
    n = 240
    values = np.cumsum(rng.uniform(1.0, 3.0, size=n))
    base = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "monotone": values,
    })
    original, _ = build_dataframe_grammar(_profile(base), _base_spec(), name="ordered")
    shuffled_frame = base.copy()
    shuffled_frame["timestamp"] = rng.permutation(shuffled_frame["timestamp"].to_numpy())
    shuffled, _ = build_dataframe_grammar(_profile(shuffled_frame), _base_spec(), name="shuffled")
    rule = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("monotone"), 1), ">=", A.Const(0)),
    )
    config = DiscoveryConfig(tolerance=1e-12, hold_rate_threshold=0.8, band_mode="global")

    assert DataOnlyEvaluator(original, config).evaluate(rule).accepted
    assert not DataOnlyEvaluator(shuffled, config).evaluate(rule).accepted


def test_temporal_known_signatures_and_proxy_recovery():
    known = [
        KnownInvariant("monotone", ">=", {"delta": "loss"}, 0),
        KnownInvariant(
            "windowed",
            "==",
            "ratio_1h",
            {
                "ratio": [
                    {"roll_sum": ["output", 60]},
                    {"roll_sum": ["input", 60]},
                ],
            },
        ),
    ]

    assert _signature(known[0])[0] == "delta_bound"
    assert _signature(known[1])[:2] == ("equality", "exact")
    assert _signature(known[1])[2][0] == "windowed_ratio"
    assert shapes_for_invariant(known[0]) == ["monotone"]
    assert shapes_for_invariant(known[1]) == ["windowed_ratio"]

    result = SimpleNamespace(portfolio=[])
    planted = {
        "monotone": {("loss", 1, ">=")},
        "windowed_ratio": {("ratio_1h", "output", "input", 60)},
    }
    from autogram.discovery import validate as validation
    original_relations = validation.portfolio_relations
    original_one_sided = validation._portfolio_one_sided_columns
    try:
        validation.portfolio_relations = lambda _result: {
            ("delta_bound", ("loss", 1, ">=")),
            ("windowed_ratio", ("ratio_1h", "output", "input", 60)),
        }
        validation._portfolio_one_sided_columns = lambda *_args: set()
        recovery = score_recovery(result, planted)
    finally:
        validation.portfolio_relations = original_relations
        validation._portfolio_one_sided_columns = original_one_sided

    assert recovery.monotone == 1.0
    assert recovery.windowed_ratio == 1.0


def test_documented_lag_bound_signature_recovers():
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=120, freq="1min"),
        "series_id": ["a"] * 120,
        "x": np.arange(1.0, 121.0),
    }), windows=(2,), max_lag=2)
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="known_lag",
    )
    rule = A.Rule(
        "record",
        A.Compare(A.Lag(A.Ref("x"), 2), ">=", A.Const(0)),
    )
    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-12,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(rule)
    result = SimpleNamespace(dataset=dataset, portfolio=[evaluation])
    known = KnownInvariant(
        "lag_nonnegative",
        ">=",
        {"lag": ["x", 2]},
        0,
    )

    assert _signature(known) == ("lag_bound", ("x", 2, ">="))
    assert recover_known(result, [known])["recall"] == 1.0


def test_lag_bound_retained_and_recovered_end_to_end_when_atomic_not_exact():
    # Round-19 end-to-end: a genuinely independent lag law must survive the real archive, not be
    # dropped as a bloated shadow. x is non-negative except its final row, so ``LAG_1(x) >= 0`` holds
    # (the shift never inspects the last row) while the atomic ``x >= 0`` is NOT exact. Running the
    # rules through the production archive must keep the lag in the non-redundant portfolio and
    # recover the lag known -- otherwise the lag is a false-negative miss.
    from autogram.discovery.archive import ParetoArchive

    values = np.array([1.0] * 119 + [-100.0])
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=120, freq="1min"),
        "series_id": ["a"] * 120,
        "x": values,
    }), windows=(2,), max_lag=2)
    dataset_, _grammar = build_dataframe_grammar(frame, _base_spec(), name="known_lag_e2e")
    cfg = DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.9, band_mode="global")
    ev = DataOnlyEvaluator(dataset_, cfg)
    archive = ParetoArchive()
    archive.add(ev.evaluate(A.Rule("record", A.Compare(A.Lag(A.Ref("x"), 1), ">=", A.Const(0)))))
    archive.add(ev.evaluate(A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))))
    portfolio = archive.portfolio(non_redundant=True)
    # The lag survives (its atomic is not exact), and it recovers the lag known.
    assert any(
        isinstance(e.rule.atom, A.Compare) and isinstance(e.rule.atom.left, A.Lag)
        for e in portfolio
    )
    result = SimpleNamespace(dataset=dataset_, portfolio=portfolio)
    known = KnownInvariant("lag_nonnegative", ">=", {"lag": ["x", 1]}, 0)
    assert recover_known(result, [known])["recall"] == 1.0


def test_lag_bound_retained_when_atomic_is_tolerance_absorbed_end_to_end():
    # Round-20 soundness end-to-end: the atomic ``x >= 0`` reaches hold-rate 1.0 only because the
    # acceptance tolerance absorbs the single ``-10`` against the huge ``1e9`` scale, yet
    # ``LAG_100(x) >= 0`` is a genuinely accepted, independent lag law. Suppression must be gated on
    # tolerance-free exactness, so the lag survives the archive and is recovered (not a false miss).
    from autogram.discovery.archive import ParetoArchive

    values = np.array([-10.0] + [1.0] * 99 + [1e9] * 100)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=200, freq="1min"),
        "series_id": ["a"] * 200,
        "x": values,
    }), windows=(2,), max_lag=100)
    dataset_, _grammar = build_dataframe_grammar(frame, _base_spec(), name="known_lag_absorbed")
    cfg = DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.9, band_mode="global")
    ev = DataOnlyEvaluator(dataset_, cfg)
    atomic_eval = ev.evaluate(A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0))))
    lag_eval = ev.evaluate(A.Rule("record", A.Compare(A.Lag(A.Ref("x"), 100), ">=", A.Const(0))))
    # Atomic hold-rate is 1.0 (tolerance-absorbed) but NOT raw-exact; the lag is accepted.
    assert atomic_eval.hold_rate == 1.0 and atomic_eval.raw_exact_sign is False
    assert lag_eval.accepted
    archive = ParetoArchive()
    archive.add(lag_eval)
    archive.add(atomic_eval)
    portfolio = archive.portfolio(non_redundant=True)
    assert any(
        isinstance(e.rule.atom, A.Compare) and isinstance(e.rule.atom.left, A.Lag)
        for e in portfolio
    )
    result = SimpleNamespace(dataset=dataset_, portfolio=portfolio)
    known = KnownInvariant("lag_nonnegative", ">=", {"lag": ["x", 100]}, 0)
    assert recover_known(result, [known])["recall"] == 1.0


def test_lag_bound_not_recovered_when_tolerance_absorbs_atomic_violation():
    # Round-18 soundness: one-sided *evaluation* accepts a bound within a relative tolerance against a
    # population scale floor, so an atomic ``x >= 0`` can report hold-rate 1.0 even though a raw value
    # is negative (a large-scale column absorbs the violation). That does NOT imply the shifted law.
    # Reviewer counterexample: x=[-10,1,1,1e9,1e9,1e9], lag 3 -> atomic hold-rate 1.0 but LAG_3(x)>=0
    # genuinely fails. The lag known must NOT be recovered (recovery is verified tolerance-free).
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="1min"),
        "series_id": ["a"] * 6,
        "x": [-10.0, 1.0, 1.0, 1e9, 1e9, 1e9],
    }), windows=(2,), max_lag=3)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="known_lag_tol")
    cfg = DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.6, band_mode="global")
    ev = DataOnlyEvaluator(dataset, cfg)
    atomic = ev.evaluate(A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0))))
    lag = ev.evaluate(A.Rule("record", A.Compare(A.Lag(A.Ref("x"), 3), ">=", A.Const(0))))
    # The tolerance absorbs -10 for the atomic (hold-rate 1.0) but the genuine lag law fails.
    assert atomic.hold_rate == 1.0 and not lag.accepted
    result = SimpleNamespace(dataset=dataset, portfolio=[atomic])
    known = KnownInvariant("lag_nonnegative", ">=", {"lag": ["x", 3]}, 0)

    report = recover_known(result, [known])
    assert report["recall"] == 0.0
    assert report["invariants"][0]["recovered"] is False


def test_lag_bound_recovered_from_exact_atomic_sign_law():
    # Soundness (round-17): a lag sign law ``LAG_k(x) >= 0`` is a *guaranteed consequence* of an
    # EXACT atomic sign law ``x >= 0`` (hold-rate 1.0) -- x non-negative on every row forces the
    # shift non-negative on every valid lagged row. The lag form itself is not retained (it would
    # pollute the null-temporal control), so recovery maps to the exact atomic. Portfolio holds only
    # the exact atomic (no lag rule); the lag known must be recovered.
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=120, freq="1min"),
        "series_id": ["a"] * 120,
        "x": np.arange(1.0, 121.0),
    }), windows=(2,), max_lag=2)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="known_lag_exact_atomic")
    atomic = A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))
    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=1e-12, hold_rate_threshold=0.9, band_mode="global"),
    ).evaluate(atomic)
    assert evaluation.accepted and evaluation.hold_rate == 1.0
    result = SimpleNamespace(dataset=dataset, portfolio=[evaluation])
    known = KnownInvariant("lag_nonnegative", ">=", {"lag": ["x", 2]}, 0)

    report = recover_known(result, [known])
    assert report["recall"] == 1.0
    assert report["invariants"][0]["recovered"] is True

    # The opposite direction is not implied by an exact ``x >= 0`` and must not be recovered.
    other = KnownInvariant("lag_nonpositive", "<=", {"lag": ["x", 2]}, 0)
    assert recover_known(result, [other])["recall"] == 0.0


def test_lag_bound_recovered_structurally_from_retained_lag_rule():
    # A genuinely temporal lag rule that survives in the portfolio matches the lag known
    # structurally (no atomic present at all).
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=120, freq="1min"),
        "series_id": ["a"] * 120,
        "x": np.arange(1.0, 121.0),
    }), windows=(2,), max_lag=2)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="known_lag_retained")
    lag_rule = A.Rule("record", A.Compare(A.Lag(A.Ref("x"), 2), ">=", A.Const(0)))
    lag_eval = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=1e-12, hold_rate_threshold=0.9, band_mode="global"),
    ).evaluate(lag_rule)
    assert lag_eval.accepted
    result = SimpleNamespace(dataset=dataset, portfolio=[lag_eval])
    known = KnownInvariant("lag_nonnegative", ">=", {"lag": ["x", 2]}, 0)

    report = recover_known(result, [known])
    assert report["recall"] == 1.0
    assert report["invariants"][0]["recovered"] is True


def test_lag_bound_not_recovered_from_nonexact_atomic():
    # No false positive (round-17): an atomic ``x >= 0`` that is merely ACCEPTED but NOT exact
    # (hold-rate < 1.0 because a few rows are negative) does NOT imply the shifted law over its
    # distinct valid-row population, so a lag known must remain unrecovered when no lag rule exists.
    values = np.arange(1.0, 121.0)
    values[10] = -1.0
    values[50] = -2.0
    values[90] = -3.0
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=120, freq="1min"),
        "series_id": ["a"] * 120,
        "x": values,
    }), windows=(2,), max_lag=2)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="known_lag_nonexact")
    atomic = A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))
    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=1e-12, hold_rate_threshold=0.9, band_mode="global"),
    ).evaluate(atomic)
    # Accepted (few violations) but NOT exact.
    assert evaluation.accepted and evaluation.hold_rate < 1.0
    result = SimpleNamespace(dataset=dataset, portfolio=[evaluation])
    known = KnownInvariant("lag_nonnegative", ">=", {"lag": ["x", 2]}, 0)

    report = recover_known(result, [known])
    assert report["recall"] == 0.0
    assert report["invariants"][0]["recovered"] is False


def test_temporal_proxy_generators_plant_monotone_and_windowed_ratio():
    monotone = synth.make_synthetic(
        n_entities=3,
        n_snapshots=40,
        noise=0.0,
        seed=3,
        families=("monotone",),
    )
    assert "monotone" in monotone.planted
    for column in monotone.planted["monotone"]:
        index = monotone.columns.index(column[0])
        assert np.all(np.diff(monotone.matrix[:, index]) >= 0.0)

    windowed = synth.make_synthetic(
        n_entities=3,
        n_snapshots=40,
        noise=0.0,
        seed=3,
        families=("windowed_ratio",),
        temporal_window=5,
    )
    target = next(iter(windowed.planted["windowed_ratio"]))
    lhs, numerator, denominator, width = target
    actual = windowed.matrix[:, windowed.columns.index(lhs)]
    expected = (
        pd.Series(windowed.matrix[:, windowed.columns.index(numerator)])
        .rolling(width, min_periods=width)
        .sum()
        / pd.Series(windowed.matrix[:, windowed.columns.index(denominator)])
        .rolling(width, min_periods=width)
        .sum()
    ).to_numpy()
    assert np.allclose(actual, expected, equal_nan=True)


def test_temporal_capability_is_a_separate_widening_tier():
    tiers = _capability_tiers()
    assert tiers[-1]["temporal"] is True
    widened = _widen_spec(
        _base_spec(max_degree=2),
        temporal=True,
        max_lag=60,
        windows=(45, 60),
    )
    assert widened.temporal_enabled is True
    assert widened.max_lag == 60
    assert widened.windows == (45, 60)


def test_zero_support_lag_law_is_not_credited_by_implication():
    """Round-27: implication transfers a law the data witnesses; it does not invent one.

    An exact atomic sign law `x >= 0` does imply `LAG_k(x) >= 0` on every row where the lag is
    defined -- but if the lag is longer than the series, there is no such row. Crediting it would
    report recall for a law the dataset never exhibits, which is exactly the kind of over-claim the
    held-out recall figure is supposed to rule out.
    """
    import numpy as np
    import pandas as pd

    from autogram.config import DiscoveryConfig, SearchConfig
    from autogram.discovery.known import KnownInvariant, recover_known
    from autogram.discovery.loop import build_dataframe_grammar, run_prepared
    from autogram.loader.gtib import profile_dataframe
    from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology

    spec = GrammarSpec(
        name="lag_support",
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
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "queue_bytes": np.arange(1.0, n + 1.0),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=(),
        temporal_windows=(2,),
        max_lag=3,
    )
    search = SearchConfig(max_complexity=8, max_lag=3, windows=(2,), seed=0)
    dataset, grammar = build_dataframe_grammar(frame, spec, search_cfg=search, name="lag_support")
    result = run_prepared(
        dataset,
        grammar,
        discovery_cfg=DiscoveryConfig(band_mode="global", seed=0),
        search_cfg=search,
    )

    reachable = KnownInvariant("lag_2", ">=", {"lag": ["queue_bytes", 2]}, 0)
    unreachable = KnownInvariant("lag_100", ">=", {"lag": ["queue_bytes", 100]}, 0)

    assert recover_known(result, [reachable])["recall"] == 1.0
    # The 100-step lag grounds no row on a 50-row series, so it must NOT be credited.
    assert recover_known(result, [unreachable])["recall"] == 0.0
