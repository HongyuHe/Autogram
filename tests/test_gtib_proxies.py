"""Focused proxy generation, capability wiring, and recovery for GTIB shapes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from autogram.config import DiscoveryConfig
from autogram.config import SearchConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.induce import SchemaInducer
from autogram.discovery.regime import ProxyEntry, RegimeSpec
from autogram.discovery.validate import (
    evaluate_grid_candidate,
    null_temporal_at,
    null_definitions_at,
    prepare_proxy_suite,
    score_recovery,
    tune_joint,
)
from autogram.dsl import ast as A
from autogram.schema.spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    GrammarSpec,
    RefTemplate,
    RoleOntology,
)


class _SyntheticInducer(SchemaInducer):
    def induce(self, columns, sample_rows=None) -> GrammarSpec:
        return GrammarSpec(
            name="synthetic",
            patterns=(
                ColumnPattern(
                    "flow",
                    "regex",
                    "flow",
                    "demand",
                    regex=r"^flow_(?P<source>n\d+)_(?P<destination>n\d+)$",
                    node_groups=("source", "destination"),
                    source_group="source",
                    destination_group="destination",
                    token_groups=("source", "destination"),
                ),
                ColumnPattern(
                    "source",
                    "regex",
                    "measurement",
                    "source",
                    regex=r"^measurement_(?P<source>n\d+)_source$",
                    node_groups=("source",),
                    source_group="source",
                    token_groups=("source",),
                ),
                ColumnPattern(
                    "destination",
                    "regex",
                    "measurement",
                    "destination",
                    regex=r"^measurement_(?P<source>n\d+)_destination$",
                    node_groups=("source",),
                    source_group="source",
                    token_groups=("source",),
                ),
                ColumnPattern(
                    "to",
                    "regex",
                    "measurement",
                    "to",
                    regex=r"^measurement_(?P<source>n\d+)_to_(?P<peer>n\d+)$",
                    node_groups=("source", "peer"),
                    source_group="source",
                    peer_group="peer",
                    token_groups=("source", "peer"),
                ),
                ColumnPattern(
                    "from",
                    "regex",
                    "measurement",
                    "from",
                    regex=r"^measurement_(?P<source>n\d+)_from_(?P<peer>n\d+)$",
                    node_groups=("source", "peer"),
                    source_group="source",
                    peer_group="peer",
                    token_groups=("source", "peer"),
                ),
            ),
            ontology=RoleOntology(
                binders=("cell", "node", "network", "link"),
                ref_roles={
                    "cell": ("self",),
                    "node": (
                        "measurement_source",
                        "measurement_destination",
                        "demand_self",
                    ),
                    "network": (),
                    "link": (),
                },
                fam_roles={
                    "cell": (),
                    "node": (),
                    "network": (),
                    "link": (),
                },
            ),
            ref_templates=(
                RefTemplate("cell", "self", "{col}"),
                RefTemplate("node", "measurement_source", "measurement_{X}_source"),
                RefTemplate("node", "measurement_destination", "measurement_{X}_destination"),
                RefTemplate("node", "demand_self", "flow_{X}_{X}"),
            ),
            family_selectors=(),
            binder_enumerate={
                "cell": "per_measured_col",
                "node": "per_node",
                "network": "singleton",
                "link": "per_directed_link",
            },
            cell_codec=CellCodec(kind="scalar"),
            noisy_kind="measurement",
            demand_kind="flow",
            link_marker_direction="to",
        )


class _WideSyntheticInducer(_SyntheticInducer):
    def induce(self, columns, sample_rows=None) -> GrammarSpec:
        spec = super().induce(columns, sample_rows)
        fam_roles = dict(spec.ontology.fam_roles)
        fam_roles["node"] = (
            "demand_row",
            "demand_col",
            "fam_from",
            "fam_to",
        )
        return replace(
            spec,
            ontology=replace(
                spec.ontology,
                fam_roles=fam_roles,
                agg_kinds=("SUM", "AVG", "MIN", "MAX"),
            ),
            family_selectors=(
                FamilySelector(
                    "node",
                    "demand_row",
                    "flow",
                    "demand",
                    (
                        ("source", "==", "X"),
                        ("destination", "!=", "X"),
                    ),
                ),
                FamilySelector(
                    "node",
                    "demand_col",
                    "flow",
                    "demand",
                    (
                        ("destination", "==", "X"),
                        ("source", "!=", "X"),
                    ),
                ),
            ),
        )


class _CountingSyntheticInducer(_SyntheticInducer):
    def __init__(self):
        self.calls = 0
        self.column_schemas = []

    def induce(self, columns, sample_rows=None) -> GrammarSpec:
        self.calls += 1
        self.column_schemas.append(tuple(columns))
        return super().induce(columns, sample_rows)


def _rule(shape: str) -> A.Rule:
    source = A.Ref("measurement_source")
    destination = A.Ref("measurement_destination")
    demand = A.Ref("demand_self")
    if shape == "ratio":
        atom = A.Compare(source, "==", A.Div(destination, demand))
    elif shape == "proportional":
        atom = A.Compare(source, "~∝", destination)
    elif shape == "monotone":
        atom = A.Compare(A.Diff(source, 1), ">=", A.Const(0))
    elif shape == "windowed_ratio":
        atom = A.Compare(
            source,
            "==",
            A.Div(
                A.Rolling(destination, 5, "SUM"),
                A.Rolling(demand, 5, "SUM"),
            ),
        )
    elif shape == "conditional_positive":
        return A.Rule(
            "node",
            A.Compare(A.Diff(source, 1), ">=", A.Const(0)),
            condition=A.Condition("regime", "==", ("positive",)),
        )
    elif shape == "conditional_zero":
        return A.Rule(
            "node",
            A.Compare(A.Diff(source, 1), "==", A.Const(0)),
            condition=A.Condition("regime", "==", ("zero",)),
        )
    elif shape == "sustained":
        atom = A.BooleanDefinition(
            source,
            A.Sustained(A.Bound(destination, "<", None), 5),
        )
    elif shape == "conjunction":
        atom = A.BooleanDefinition(
            source,
            A.Conjunction((
                A.Bound(destination, "<", None),
                A.Bound(
                    A.Rolling(
                        A.Add((destination, A.Scale(-1.0, demand))),
                        5,
                        "SUM",
                    ),
                    ">",
                    0.0,
                ),
                A.Bound(A.Diff(destination, 5), "<=", 0.0),
            )),
        )
    elif shape == "categorical":
        atom = A.CategoryDefinition(
            "category",
            (
                ("flag_a", "class_a"),
                ("flag_b", "class_b"),
                ("flag_c", "class_c"),
            ),
            "baseline",
        )
    elif shape == "cross_grain":
        atom = A.Compare(source, "==", A.RelatedAgg("proxy_raw_sum"))
    elif shape == "healthy_band":
        return A.Rule(
            "node",
            A.BandDefinition(source, None),
            condition=A.Condition(
                "",
                "all",
                (
                    A.Condition("segment", "==", ("stable",)),
                    A.Condition("state", "==", ("baseline",)),
                ),
            ),
        )
    else:
        raise AssertionError(shape)
    return A.Rule("node", atom)


@pytest.mark.parametrize(
    "shape,noise",
    [
        ("ratio", 0.0),
        ("proportional", 0.02),
        ("monotone", 0.0),
        ("windowed_ratio", 0.0),
        ("conditional_positive", 0.0),
        ("conditional_zero", 0.0),
        ("sustained", 0.0),
        ("conjunction", 0.0),
        ("categorical", 0.0),
        ("cross_grain", 0.0),
        ("healthy_band", 0.0),
    ],
)
def test_new_proxy_shape_recovers_with_zero_temporal_null(shape, noise):
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                shape,
                noise=noise,
                n_entities=3,
                n_snapshots=120,
                temporal_window=5,
            ),
        ]),
        seed=0,
        inducer=_WideSyntheticInducer(),
    )
    proxy = suite.positives[0]
    evaluation = DataOnlyEvaluator(
        proxy.ds,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.75,
            band_mode="global",
        ),
    ).evaluate(_rule(shape))
    result = type(
        "Result",
        (),
        {"portfolio": [evaluation], "dataset": proxy.ds},
    )()
    recovery = score_recovery(result, proxy.planted)

    assert evaluation.accepted, (shape, evaluation.reason, evaluation.hold_rate)
    assert getattr(recovery, shape) == 1.0


def test_categorical_proxy_recovers_transitively_identified_order():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                "categorical",
                noise=0.0,
                n_entities=3,
                n_snapshots=120,
            ),
        ]),
        seed=0,
        inducer=_WideSyntheticInducer(),
    )
    proxy = suite.positives[0]
    rule = _rule("categorical")
    n = proxy.ds.observed.n_rows
    phase = np.arange(n) % 6
    first = np.isin(phase, (0, 3))
    second = np.isin(phase, (1, 3, 4))
    third = np.isin(phase, (2, 4))
    category = np.full(n, "baseline", dtype=object)
    category[third] = "class_c"
    category[second] = "class_b"
    category[first] = "class_a"
    for name, values in {
        "flag_a": first,
        "flag_b": second,
        "flag_c": third,
        "category": category,
    }.items():
        proxy.ds.row_context[name] = values
        proxy.ds.observed.row_context[name] = values

    evaluation = DataOnlyEvaluator(
        proxy.ds,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.75,
            band_mode="global",
        ),
    ).evaluate(rule)
    recovery = score_recovery(
        type(
            "Result",
            (),
            {"portfolio": [evaluation], "dataset": proxy.ds},
        )(),
        proxy.planted,
    )

    assert not np.any(first & third)
    assert evaluation.accepted
    assert recovery.categorical == 1.0


def test_time_shuffled_proxy_accepts_no_temporal_rules():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry("monotone", noise=0.0, n_entities=3, n_snapshots=240),
        ]),
        seed=0,
        inducer=_WideSyntheticInducer(),
    )
    assert suite.temporal_null is not None
    assert null_temporal_at(
        suite.temporal_null,
        DiscoveryConfig(
            tolerance=1e-12,
            hold_rate_threshold=0.8,
            band_mode="global",
        ),
        seed=0,
    ) == 0


def test_definition_null_accepts_no_advanced_definitions():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry("sustained", noise=0.0, n_entities=3, n_snapshots=240),
            ProxyEntry("categorical", noise=0.0, n_entities=3, n_snapshots=240),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )
    assert suite.definition_null is not None
    case_columns = {
        name
        for name, values in suite.definition_null.ds.row_context.items()
        if np.asarray(values).dtype.kind == "b"
    }
    assert set(
        suite.definition_null.G.category_cases_for("node")
    ) == case_columns
    assert case_columns.isdisjoint(
        suite.definition_null.G.condition_columns
    )
    assert any(
        isinstance(rule.atom, A.CategoryDefinition)
        for rule in suite.definition_null.proposer.propose()
    )
    assert null_definitions_at(
        suite.definition_null,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.62,
            band_mode="global",
        ),
        seed=0,
    ) == 0


def test_null_controls_compile_conditional_and_related_hypotheses():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry("conditional_positive", noise=0.0),
            ProxyEntry("healthy_band", noise=0.0),
            ProxyEntry("cross_grain", noise=0.0),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )

    assert suite.null.G.band_enabled
    assert set(suite.null.G.condition_columns) >= {"segment", "state"}
    assert suite.null.G.related_for("node") == ("proxy_raw_sum",)
    assert "raw" in suite.null.ds.relations
    assert suite.temporal_null is not None
    assert suite.temporal_null.G.conditional_enabled
    assert "regime" in suite.temporal_null.G.condition_columns

    candidate = evaluate_grid_candidate(
        suite,
        tolerance=0.05,
        hold_rate_threshold=0.75,
        seed=0,
        band_mode="global",
    )
    assert candidate.null_equalities == 0
    assert candidate.null_temporal == 0


def test_cross_grain_only_suite_still_covers_full_null_capabilities():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry("cross_grain", noise=0.0),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )

    assert suite.temporal_null is not None
    assert suite.definition_null is not None
    assert suite.null.G.degree_cap("node") >= 2
    assert "~\u221d" in suite.null.G.ops
    assert suite.null.G.band_enabled
    assert suite.null.G.related_for("node") == ("proxy_raw_sum",)
    assert suite.temporal_null.G.conditional_enabled
    assert suite.temporal_null.G.degree_cap("node") >= 2
    assert suite.definition_null.G.advanced_enabled


def test_null_grammars_use_runtime_search_envelope():
    search = SearchConfig(
        max_complexity=16,
        max_add_arity=3,
        max_rules=1234,
        max_nonlinear_leaves=11,
        max_linear_leaves=7,
        max_conditioned_rules=321,
    )
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[ProxyEntry("cross_grain", noise=0.0)]),
        seed=0,
        inducer=_SyntheticInducer(),
        null_search_cfg=search,
    )

    for proxy in (
        suite.null,
        suite.temporal_null,
        suite.definition_null,
    ):
        assert proxy is not None
        assert proxy.G.max_complexity == 16
        assert proxy.G.max_add_arity == 3
        assert proxy.G.max_rules == 1234
        assert proxy.G.max_nonlinear_leaves == 11
        assert proxy.G.max_linear_leaves == 7
        assert proxy.G.max_conditioned_rules == 321
        assert proxy.search_cfg is search


def test_windowed_ratio_null_enumerates_rolling_ratio_hypotheses():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry("windowed_ratio", noise=0.0),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )

    assert suite.temporal_null is not None
    assert any(
        isinstance(rule.atom, A.Compare)
        and any(
            isinstance(term, A.Div)
            and isinstance(term.num, A.Rolling)
            and isinstance(term.den, A.Rolling)
            for term in (rule.atom.left, rule.atom.right)
        )
        for rule in suite.temporal_null.proposer.propose()
    )


def test_proxy_temporal_window_controls_positive_and_null_grammars():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                "windowed_ratio",
                noise=0.0,
                temporal_window=7,
            ),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )

    assert 7 in suite.positives[0].G.windows
    assert suite.temporal_null is not None
    assert 7 in suite.temporal_null.G.windows
    candidate = evaluate_grid_candidate(
        suite,
        tolerance=0.05,
        hold_rate_threshold=0.75,
        seed=0,
        band_mode="global",
    )
    assert candidate.proxies[0].recovery == 1.0


@pytest.mark.parametrize("shape", [
    "ratio", "proportional", "monotone", "lag_bound", "sum_balance",
    "conditional_proportional", "conditional_pair",
    "windowed_ratio", "conditional_positive",
    "conditional_zero", "sustained", "conjunction", "categorical",
    "cross_grain", "healthy_band",
])
def test_joint_grid_scores_new_proxy_compactly_and_safely(shape):
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                shape,
                noise=0.0,
                n_entities=3,
                n_snapshots=120,
                temporal_window=5,
            ),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )
    if shape == "categorical":
        proxy = suite.positives[0]
        target, cases, _default = next(iter(proxy.planted["categorical"]))
        case_columns = {column for column, _value in cases}
        assert set(proxy.G.category_cases_for("node")) == case_columns
        assert target in proxy.G.condition_columns
        assert case_columns.isdisjoint(proxy.G.condition_columns)
        assert any(
            rule.signature() == _rule("categorical").signature()
            for rule in proxy.proposer.propose()
        )
    candidate = evaluate_grid_candidate(
        suite,
        tolerance=0.05,
        hold_rate_threshold=0.75,
        seed=0,
        band_mode="global",
    )
    outcome = candidate.proxies[0]

    assert outcome.recovery >= 0.8, candidate.evidence()
    if shape == "categorical":
        assert outcome.recovery == 1.0
    assert outcome.compact, candidate.evidence()
    assert candidate.null_equalities == 0
    assert candidate.null_temporal == 0
    if shape == "proportional":
        assert "~∝" in suite.null.G.ops


def test_all_new_shapes_jointly_tune_under_all_null_guards():
    shapes = [
        "ratio",
        "proportional",
        "monotone",
        "lag_bound",
        "sum_balance",
        "conditional_proportional",
        "conditional_pair",
        "windowed_ratio",
        "conditional_positive",
        "conditional_zero",
        "sustained",
        "conjunction",
        "categorical",
        "cross_grain",
        "healthy_band",
    ]
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                shape,
                noise=0.0,
                n_entities=3,
                n_snapshots=120,
                temporal_window=5,
            )
            for shape in shapes
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )
    tuned = tune_joint(
        suite,
        seed=0,
        band_mode="global",
        max_expansions=0,
        thresholds=[0.75],
        tolerances=[0.05],
    )

    assert tuned["proxy_shapes"] == shapes
    assert tuned["selected_null_equalities"] == 0
    assert tuned["selected_null_temporal"] == 0
    assert tuned["selected_null_definitions"] == 0
    assert all(proxy["recovery"] == 1.0 for proxy in tuned["per_proxy"])


def test_sum_balance_null_grammar_contains_sum_vs_sum_candidates():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                "sum_balance",
                noise=0.0,
                n_entities=3,
                n_snapshots=80,
            ),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )

    assert "demand_row" in suite.null.G.fams_for("node")
    assert "demand_col" in suite.null.G.fams_for("node")
    assert any(
        isinstance(rule.atom, A.Compare)
        and isinstance(rule.atom.left, A.Agg)
        and isinstance(rule.atom.right, A.Agg)
        and rule.atom.left.kind == rule.atom.right.kind == "SUM"
        for rule in suite.null.proposer.propose()
    )


def test_presence_proxy_null_has_balanced_independent_absence_masks():
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry(
                "presence_pair",
                noise=0.0,
                n_entities=3,
                n_snapshots=80,
            ),
        ]),
        seed=0,
        inducer=_SyntheticInducer(),
    )
    assert suite.presence_null is not None
    matrix = suite.presence_null.ds.observed.matrix
    absent = np.abs(matrix) <= 1e-12

    assert np.all(absent.sum(axis=0) == matrix.shape[0] // 2)
    assert any(
        not np.array_equal(absent[:, left], absent[:, right])
        for left in range(matrix.shape[1])
        for right in range(left + 1, matrix.shape[1])
    )


@pytest.mark.parametrize("shape", ["ratio", "windowed_ratio"])
def test_wide_aggregate_vocabulary_does_not_starve_ratio_proxy(shape):
    suite = prepare_proxy_suite(
        RegimeSpec(entries=[ProxyEntry(shape, noise=0.0)]),
        seed=0,
        inducer=_WideSyntheticInducer(),
    )

    candidate = evaluate_grid_candidate(
        suite,
        tolerance=0.02,
        hold_rate_threshold=0.5,
        seed=0,
        band_mode="global",
    )

    assert candidate.proxies[0].recovery == 1.0, candidate.evidence()


def test_proxy_suite_induces_each_distinct_column_schema_once():
    inducer = _CountingSyntheticInducer()

    prepare_proxy_suite(
        RegimeSpec(entries=[
            ProxyEntry("ratio"),
            ProxyEntry("monotone"),
            ProxyEntry("categorical"),
        ]),
        seed=0,
        inducer=inducer,
    )

    assert inducer.calls == len(set(inducer.column_schemas)) == 2
