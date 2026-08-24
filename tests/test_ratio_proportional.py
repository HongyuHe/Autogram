"""Ratio declarations and robust proportional-equality evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogram.config import DiscoveryConfig
from autogram.cli import _portfolio_payload
from autogram.discovery.archive import ParetoArchive
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.known import KnownInvariant, _signature, recover_known, shapes_for_invariant
from autogram.discovery.loop import build_dataframe_grammar
from autogram.discovery.propose import EnumerationProposer
from autogram.discovery.validate import score_recovery
from autogram.dsl import ast as A
from autogram.dsl.grammar import Grammar
from autogram.dsl.typecheck import is_admissible
from autogram.loader.gtib import profile_dataframe
from autogram.logic.solver import atom_expr, equivalent
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


def _grammar() -> Grammar:
    return Grammar(
        binders=("record",),
        ops=("~=", "==", "~∝"),
        ref_roles={"record": ("x", "y", "ratio")},
        fam_roles={"record": ()},
        max_complexity=10,
        max_degree=2,
    )


def test_proportional_operator_is_typed_enumerated_and_solver_screenable():
    grammar = _grammar()
    rule = A.Rule("record", A.Compare(A.Ref("y"), "~∝", A.Ref("x")))

    assert is_admissible(rule, grammar)[0] is True
    assert atom_expr(rule.atom, {}) is not None
    assert any(r.atom.op == "~∝" for r in EnumerationProposer(grammar).propose())


def test_proposer_enumerates_conditioned_proportionality():
    frame = profile_dataframe(
        pd.DataFrame({
            "x": np.arange(1.0, 41.0),
            "y": np.arange(2.0, 82.0, 2.0),
            "label": np.resize(
                np.array(["normal", "alert"], dtype=object),
                40,
            ),
        }),
        condition_columns=("label",),
        proportional=True,
    )
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="conditioned_proportional",
    )
    target = A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
        condition=A.Condition("label", "==", ("normal",)),
    )

    assert target.signature() in {
        rule.signature()
        for rule in EnumerationProposer(grammar).propose()
    }


def test_proportional_evaluator_fits_robust_coefficient_per_group():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.uniform(10.0, 100.0, size=2 * n)
    group = np.repeat(["a", "b"], n)
    expected = np.where(group == "a", 1.7, 0.6)
    y = expected * x
    y[[5, 19, 131, 177]] *= 8.0
    frame = profile_dataframe(
        pd.DataFrame({"group_id": group, "x": x, "y": y}),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(frame, _base_spec(), name="proportional")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.02,
            hold_rate_threshold=0.85,
            band_mode="global",
        ),
    )

    result = evaluator.evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~∝", A.Ref("x")))
    )

    assert result.accepted
    assert result.strictness == "proportional"
    assert abs(result.parameters["coefficients"]["a"] - 1.7) < 1e-9
    assert abs(result.parameters["coefficients"]["b"] - 0.6) < 1e-9
    assert result.hold_rate > 0.97
    payload = _portfolio_payload(SimpleNamespace(
        dataset=dataset,
        rounds_run=1,
        progress_history=[],
        diagnostics=[],
        portfolio=[result],
    ))
    assert payload["portfolio"][0]["parameters"]["coefficients"] == {
        "a": 1.7,
        "b": 0.6,
    }


def test_proportional_evaluator_checks_zero_predictor_rows():
    x = np.ones(100, dtype=float)
    x[:30] = 0.0
    y = 2.0 * x
    y[:30] = 5.0
    frame = profile_dataframe(pd.DataFrame({"x": x, "y": y}))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="proportional_zero",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.9,
            band_mode="global",
            seed=0,
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~∝", A.Ref("x")))
    )

    assert not result.accepted
    assert result.hold_rate < 0.7
    assert result.n_points > 30


@pytest.mark.parametrize(
    ("band_mode", "expected_n_points"),
    (("global", 440), ("adaptive", 272)),
)
def test_proportional_zero_rows_have_coefficient_independent_semantics(
    band_mode,
    expected_n_points,
):
    x = np.concatenate((
        np.arange(1.0, 801.0),
        np.zeros(600),
        np.zeros(200),
    ))
    y = np.concatenate((
        2.0 * x[:800],
        np.zeros(600),
        np.ones(200),
    ))
    frame = profile_dataframe(pd.DataFrame({"x": x, "y": y}))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name=f"proportional_zero_semantics_{band_mode}",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=2.0,
            hold_rate_threshold=0.8,
            band_mode=band_mode,
            seed=0,
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
    ))

    assert not result.accepted
    assert result.n_points == expected_n_points
    assert result.support == pytest.approx(0.625)
    assert result.hold_rate < 0.55


def test_conditioned_proportional_reapplies_support_floor_after_neutral_rows():
    n_rows = 1000
    selected = np.arange(n_rows) < 20
    x = np.ones(n_rows)
    y = 2.0 * x
    x[19] = 0.0
    y[19] = 0.0
    frame = profile_dataframe(
        pd.DataFrame({
            "x": x,
            "y": y,
            "label": np.where(selected, "selected", "other"),
        }),
        condition_columns=("label",),
        proportional=True,
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="conditioned_proportional_neutral_support",
    )
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
        condition=A.Condition("label", "==", ("selected",)),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.1,
            band_mode="global",
            seed=0,
        ),
    ).evaluate(rule)

    assert not result.accepted
    assert "condition support below minimum after proportional exclusions" in result.reason
    assert "19 source rows" in result.reason


def test_exact_equality_at_float64_max_does_not_accept_zero():
    maximum = np.finfo(float).max
    frame = profile_dataframe(pd.DataFrame({
        "x": np.full(400, maximum),
        "zero": np.zeros(400),
    }))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="max_exact_equality",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("x"), "==", A.Ref("zero")),
    ))

    assert np.isfinite(result.eps)
    assert not result.accepted
    assert result.hold_rate == 0.0


def test_exact_equality_never_reports_infinite_normalized_epsilon():
    minimum = np.nextafter(0.0, 1.0)
    frame = profile_dataframe(pd.DataFrame({
        "x": np.full(400, minimum),
        "zero": np.zeros(400),
    }))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="subnormal_exact_equality",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("x"), "==", A.Ref("zero")),
    ))

    assert np.isfinite(result.eps)


def test_exact_equality_rejects_materially_different_tiny_values():
    frame = profile_dataframe(pd.DataFrame({
        "x": np.full(400, 1e-100),
        "zero": np.zeros(400),
    }))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="tiny_exact_equality",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("x"), "==", A.Ref("zero")),
    ))

    assert not result.accepted
    assert result.hold_rate == 0.0
    assert np.isfinite(result.eps)


def test_exact_equality_accepts_identical_zero_and_subnormal_rows():
    minimum = np.nextafter(0.0, 1.0)
    values = np.concatenate((
        np.zeros(200),
        np.full(200, minimum),
    ))
    frame = profile_dataframe(pd.DataFrame({
        "x": values,
        "y": values.copy(),
    }))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="mixed_subnormal_exact_equality",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("x"), "==", A.Ref("y")),
    ))

    assert result.accepted
    assert result.hold_rate == 1.0
    assert np.isfinite(result.eps)


@pytest.mark.parametrize(
    ("band_mode", "expected_n_points"),
    (("global", 30), ("adaptive", 9)),
)
def test_proportional_accepts_subnormal_rows_but_treats_identical_zero_as_neutral(
    band_mode,
    expected_n_points,
):
    minimum = np.nextafter(0.0, 1.0)
    values = np.concatenate((
        np.zeros(400),
        np.full(600, minimum),
    ))
    frame = profile_dataframe(pd.DataFrame({
        "x": values,
        "y": values.copy(),
    }), proportional=True)
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="mixed_subnormal_proportional",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.1,
            band_mode=band_mode,
            seed=3,
            subsample=101,
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("x"), "~\u221d", A.Ref("y")),
    ))

    assert result.accepted
    assert result.hold_rate == 1.0
    assert result.n_points == expected_n_points
    assert result.support == pytest.approx(0.6)
    payload = _portfolio_payload(SimpleNamespace(
        dataset=dataset,
        rounds_run=1,
        progress_history=[],
        diagnostics=[],
        portfolio=[result],
    ))
    assert payload["portfolio"][0]["support"] == pytest.approx(0.6)


@pytest.mark.parametrize("subsample", (100, 500, 1000))
def test_proportional_subsampling_ignores_neutral_rows(subsample):
    x = np.concatenate((
        np.zeros(9900),
        np.arange(1.0, 101.0),
    ))
    frame = profile_dataframe(pd.DataFrame({
        "x": x,
        "y": 2.0 * x,
    }), proportional=True)
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name=f"proportional_neutral_subsample_{subsample}",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-9,
            hold_rate_threshold=0.62,
            band_mode="adaptive",
            seed=0,
            subsample=subsample,
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
    ))

    assert result.accepted
    assert result.n_points == 9
    assert result.support == pytest.approx(0.01)
    assert result.parameters["coefficient"] == 2.0


def test_proportional_subsampling_is_neutral_aware_per_group():
    group = np.repeat(["a", "b"], 5000)
    x = np.zeros(10000)
    x[4950:5000] = np.arange(1.0, 51.0)
    x[9950:10000] = np.arange(1.0, 51.0)
    y = np.where(group == "a", 2.0 * x, 3.0 * x)
    frame = profile_dataframe(
        pd.DataFrame({"group_id": group, "x": x, "y": y}),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="proportional_grouped_neutral_subsample",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-9,
            hold_rate_threshold=0.75,
            band_mode="global",
            seed=0,
            subsample=100,
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
    ))

    assert result.accepted
    assert result.n_points == 30
    assert result.support == pytest.approx(0.01)
    assert result.parameters["coefficients"] == {
        "a": 2.0,
        "b": 3.0,
    }


def test_proportional_subsampling_keeps_zero_predictor_violations():
    x = np.concatenate((
        np.zeros(9900),
        np.arange(1.0, 81.0),
        np.zeros(20),
    ))
    y = np.concatenate((
        np.zeros(9900),
        2.0 * np.arange(1.0, 81.0),
        np.ones(20),
    ))
    frame = profile_dataframe(pd.DataFrame({"x": x, "y": y}))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="proportional_subsample_zero_violations",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=2.0,
            hold_rate_threshold=0.8,
            band_mode="global",
            seed=0,
            subsample=100,
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
    ))

    assert not result.accepted
    assert result.n_points == 44
    assert result.support == pytest.approx(0.01)
    assert result.hold_rate == pytest.approx(24 / 44)


def test_proportional_fit_is_invariant_to_common_tiny_scaling():
    values = np.arange(1.0, 401.0)
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")),
    )
    results = []
    for scale in (1.0, 1e-15):
        frame = profile_dataframe(pd.DataFrame({
            "x": scale * values,
            "y": scale * 2.0 * values,
        }))
        dataset, _grammar_obj = build_dataframe_grammar(
            frame,
            _base_spec(),
            name=f"proportional_scale_{scale}",
        )
        results.append(DataOnlyEvaluator(
            dataset,
            DiscoveryConfig(
                tolerance=1e-9,
                hold_rate_threshold=0.9,
                band_mode="global",
                seed=0,
            ),
        ).evaluate(rule))

    assert all(result.accepted for result in results)
    assert all(
        abs(result.parameters["coefficient"] - 2.0) < 1e-12
        for result in results
    )


def test_proportional_rejects_zero_only_declared_group_as_unfittable():
    group = np.repeat(["valid", "zero_only"], [240, 120])
    x = np.concatenate([
        np.linspace(1.0, 240.0, 240),
        np.zeros(120),
    ])
    y = np.concatenate([
        2.0 * x[:240],
        np.zeros(120),
    ])
    frame = profile_dataframe(
        pd.DataFrame({"group_id": group, "x": x, "y": y}),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="proportional_zero_group",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.9,
            band_mode="global",
            seed=0,
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")))
    )

    assert not result.accepted
    assert "coefficient could not be fit" in result.reason


def test_proportional_evaluator_requires_every_group_to_hold():
    large_x = np.linspace(1.0, 990.0, 990)
    small_x = np.linspace(1.0, 10.0, 10)
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": np.repeat(["large", "small"], [990, 10]),
            "x": np.concatenate([large_x, small_x]),
            "y": np.concatenate([
                2.0 * large_x,
                np.where(np.arange(10) % 2 == 0, small_x, 10.0 * small_x),
            ]),
        }),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="proportional_group_obligation",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.9,
            band_mode="global",
            seed=0,
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")))
    )

    assert not result.accepted
    assert result.parameters["group_hold_rates"]["small"] < 0.9


@pytest.mark.parametrize(
    ("band_mode", "expected_n_points"),
    (("global", 170), ("adaptive", 65)),
)
def test_proportional_group_gate_ignores_neutral_rows_but_keeps_zero_violations(
    band_mode,
    expected_n_points,
):
    good_x = np.arange(1.0, 401.0)
    bad_x = np.arange(1.0, 101.0)
    group = np.repeat(
        ["good", "bad"],
        [good_x.size, bad_x.size + 300 + 20],
    )
    x = np.concatenate((
        good_x,
        bad_x,
        np.zeros(300),
        np.zeros(20),
    ))
    y = np.concatenate((
        2.0 * good_x,
        2.0 * bad_x,
        np.zeros(300),
        np.ones(20),
    ))
    frame = profile_dataframe(
        pd.DataFrame({"group_id": group, "x": x, "y": y}),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name=f"proportional_neutral_group_{band_mode}",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=2.0,
            hold_rate_threshold=0.8,
            band_mode=band_mode,
            seed=0,
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
    ))

    assert not result.accepted
    assert result.n_points == expected_n_points
    assert result.support == pytest.approx(520 / 820)
    assert result.parameters["group_hold_rates"]["bad"] < 0.8


def test_adaptive_proportional_band_keeps_every_fitted_group():
    large_x = np.linspace(1.0, 990.0, 990)
    small_x = np.linspace(1.0, 10.0, 10)
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": np.repeat(["large", "small"], [990, 10]),
            "x": np.concatenate([large_x, small_x]),
            "y": np.concatenate([
                2.0 * large_x,
                np.where(
                    np.arange(10) % 2 == 0,
                    small_x,
                    10.0 * small_x,
                ),
            ]),
        }),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="adaptive_proportional_group_obligation",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.62,
            band_mode="adaptive",
            seed=2,
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")))
    )

    assert not result.accepted
    assert set(result.parameters["group_hold_rates"]) == {"large", "small"}
    assert result.parameters["group_hold_rate_lows"]["small"] < 0.62


def test_adaptive_band_scores_only_held_out_rows():
    n_rows = 400
    permutation = np.random.default_rng(0).permutation(n_rows)
    n_calibration = int(round(0.7 * n_rows))
    calibration_rows = permutation[:n_calibration]
    x = np.ones(n_rows, dtype=float)
    y = np.full(n_rows, 2.0, dtype=float)
    y[calibration_rows] = 1.0
    frame = profile_dataframe(pd.DataFrame({"x": x, "y": y}))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="adaptive_holdout",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.1,
            hold_rate_threshold=0.62,
            band_mode="adaptive",
            band_holdout_frac=0.3,
            seed=0,
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~=", A.Ref("x")))
    )

    assert not result.accepted
    assert result.n_points == 120
    assert result.hold_rate == 0.0


def test_adaptive_band_keeps_and_gates_every_declared_group():
    group = np.repeat(["large", "small"], [100, 2])
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": group,
            "x": np.ones(102),
            "y": np.concatenate([
                np.ones(100),
                np.full(2, 10.0),
            ]),
        }),
        group_keys=("group_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="adaptive_group_holdout",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.8,
            band_mode="adaptive",
            seed=1,
        ),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~=", A.Ref("x")))
    )

    assert not result.accepted
    assert set(result.parameters["group_hold_rates"]) == {
        "large",
        "small",
    }
    assert result.parameters["group_hold_rate_lows"]["small"] < 0.8


def test_exact_equality_rejects_soft_law_and_matching_is_one_way():
    x = np.arange(1.0, 121.0)
    soft_frame = profile_dataframe(pd.DataFrame({
        "x": x,
        "y": 1.04 * x,
    }))
    soft_dataset, _grammar_obj = build_dataframe_grammar(
        soft_frame,
        _base_spec(),
        name="soft_equality",
    )
    evaluator = DataOnlyEvaluator(
        soft_dataset,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    )
    approximate = evaluator.evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "~=", A.Ref("x")))
    )
    exact = evaluator.evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "==", A.Ref("x")))
    )
    exact_known = [KnownInvariant("exact", "==", "y", "x")]

    assert approximate.accepted
    assert approximate.strictness == "soft"
    assert not exact.accepted
    assert recover_known(
        SimpleNamespace(dataset=soft_dataset, portfolio=[approximate]),
        exact_known,
    )["recall"] == 0.0

    exact_frame = profile_dataframe(pd.DataFrame({"x": x, "y": x.copy()}))
    exact_dataset, _grammar_obj = build_dataframe_grammar(
        exact_frame,
        _base_spec(),
        name="exact_equality",
    )
    exact_result = DataOnlyEvaluator(
        exact_dataset,
        DiscoveryConfig(hold_rate_threshold=0.9),
    ).evaluate(
        A.Rule("record", A.Compare(A.Ref("y"), "==", A.Ref("x")))
    )
    approximate_known = [KnownInvariant("approximate", "~=", "y", "x")]

    assert exact_result.accepted
    assert recover_known(
        SimpleNamespace(dataset=exact_dataset, portfolio=[exact_result]),
        approximate_known,
    )["recall"] == 1.0


def test_exact_equality_uses_ulp_scale_and_remains_distinct_in_archive():
    x = np.full(400, 1e12, dtype=float)
    y = x.copy()
    y[-20:] = np.nextafter(x[-20:], np.inf)
    too_far = x + 0.5
    frame = profile_dataframe(pd.DataFrame({
        "x": x,
        "y": y,
        "too_far": too_far,
    }))
    dataset, _grammar_obj = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="ulp_equality",
    )
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    )
    exact_rule = A.Rule(
        "record",
        A.Compare(A.Ref("y"), "==", A.Ref("x")),
    )
    approximate_rule = A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~=", A.Ref("x")),
    )
    too_far_rule = A.Rule(
        "record",
        A.Compare(A.Ref("too_far"), "==", A.Ref("x")),
    )
    exact = evaluator.evaluate(exact_rule)
    approximate = evaluator.evaluate(approximate_rule)
    archive = ParetoArchive()
    assert archive.add(approximate)
    assert archive.add(exact)

    assert exact.accepted
    assert approximate.accepted
    assert not evaluator.evaluate(too_far_rule).accepted
    assert not equivalent(exact_rule, approximate_rule)
    assert {evaluation.rule.atom.op for evaluation in archive.portfolio()} >= {
        "==",
        "~=",
    }


def test_ratio_and_proportional_known_signatures_recover():
    x = np.arange(1.0, 121.0)
    numerator = 3.0 * x
    denominator = x + 2.0
    ratio = numerator / denominator
    proportional = 1.25 * x
    frame = profile_dataframe(pd.DataFrame({
        "x": x,
        "numerator": numerator,
        "denominator": denominator,
        "ratio": ratio,
        "proportional": proportional,
    }))
    dataset, _grammar_obj = build_dataframe_grammar(frame, _base_spec(), name="known")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=1e-9, hold_rate_threshold=0.95, band_mode="global"),
    )
    ratio_rule = A.Rule(
        "record",
        A.Compare(
            A.Ref("ratio"),
            "==",
            A.Div(A.Ref("numerator"), A.Ref("denominator")),
        ),
    )
    proportional_rule = A.Rule(
        "record",
        A.Compare(A.Ref("proportional"), "~∝", A.Ref("x")),
    )
    portfolio = [evaluator.evaluate(ratio_rule), evaluator.evaluate(proportional_rule)]
    result = SimpleNamespace(portfolio=portfolio, dataset=dataset)
    known = [
        KnownInvariant(
            "ratio",
            "==",
            "ratio",
            {"ratio": ["numerator", "denominator"]},
        ),
        KnownInvariant("proportional", "~∝", "proportional", "x"),
    ]

    report = recover_known(result, known)

    assert report["recall"] == 1.0
    assert _signature(known[0])[:2] == ("equality", "exact")
    assert _signature(known[0])[2][0] == "ratio"
    assert _signature(known[1])[0] == "proportional"
    assert shapes_for_invariant(known[0]) == ["ratio"]
    assert shapes_for_invariant(known[1]) == ["proportional"]


def test_ratio_and_proportional_proxy_recovery_fields_are_numeric(monkeypatch):
    result = SimpleNamespace(portfolio=[])
    planted = {
        "ratio": {("ratio", "numerator", "denominator")},
        "proportional": {("proportional", "x")},
    }
    monkeypatch.setattr(
        "autogram.discovery.validate.portfolio_relations",
        lambda _result: {
            ("ratio", ("ratio", "numerator", "denominator")),
            ("proportional", ("proportional", "x")),
        },
    )

    recovery = score_recovery(result, planted)

    assert recovery.ratio == 1.0
    assert recovery.proportional == 1.0


def test_proportional_rule_rejects_one_coefficient_per_unique_group():
    rng = np.random.default_rng(11)
    n = 100
    frame = profile_dataframe(
        pd.DataFrame({
            "request_id": [f"request-{index}" for index in range(n)],
            "x": rng.uniform(1.0, 10.0, size=n),
            "y": rng.uniform(1.0, 10.0, size=n),
        }),
        group_keys=("request_id",),
    )
    dataset, _grammar_obj = build_dataframe_grammar(frame, _base_spec(), name="unique")

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.8,
            band_mode="global",
        ),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~∝", A.Ref("x")),
    ))

    assert not result.accepted
    assert "coefficient could not be fit" in result.reason
