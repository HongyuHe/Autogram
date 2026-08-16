"""Sustained predicates, conjunctions, and categorical definitions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogram.config import DiscoveryConfig
from autogram.calibrate import _capability_tiers, _widen_spec
from autogram.discovery.evaluate import DataOnlyEvaluator, _predicate_population
from autogram.discovery import synth, validate as validation
from autogram.discovery.known import KnownInvariant, _signature, shapes_for_invariant
from autogram.discovery.known import recover_known
from autogram.discovery.validate import score_recovery
from autogram.discovery.loop import build_dataframe_grammar
from autogram.discovery.propose import EnumerationProposer, normalize_rule
from autogram.dsl import ast as A
from autogram.dsl.evaluate import typed_group_key, typed_signature_value
from autogram.dsl.grammar import Grammar
from autogram.dsl.parser import rule_from_dict, rule_to_dict
from autogram.dsl.typecheck import is_admissible
from autogram.loader.gtib import profile_dataframe
from autogram.logic.solver import atom_expr
from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="logic",
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


def _profile(frame: pd.DataFrame) -> pd.DataFrame:
    return profile_dataframe(
        frame,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=tuple(
            column
            for column in frame.columns
            if column.startswith("is_") or column in {"label", "alert", "trajectory_alert"}
        ),
        temporal_windows=(3, 5),
        max_lag=5,
        run_lengths=(3,),
        advanced=True,
        max_conjunction_terms=3,
    )


def _sustained(values: np.ndarray, threshold: float, window: int) -> np.ndarray:
    below = values < threshold
    output = np.zeros(values.size, dtype=bool)
    for index in range(window - 1, values.size):
        output[index] = bool(np.all(below[index - window + 1:index + 1]))
    return output


def test_conjunction_cap_enumerates_every_arity_from_two():
    for cap in (2, 3, 4):
        grammar = Grammar(
            binders=("record",),
            ops=("~=", "=="),
            ref_roles={
                "record": ("target", "a", "b", "c", "d"),
            },
            fam_roles={"record": ()},
            boolean_roles={"record": ("target",)},
            advanced_enabled=True,
            max_conjunction_terms=cap,
            max_complexity=20,
            max_rules=0,
        )
        arities = {
            len(rule.atom.predicate.predicates)
            for rule in EnumerationProposer(grammar).propose()
            if isinstance(rule.atom, A.BooleanDefinition)
            and isinstance(rule.atom.predicate, A.Conjunction)
        }

        assert set(range(2, cap + 1)) <= arities


def test_proposer_enumerates_nonstrict_sustained_and_generic_conjunctions():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "=="),
        ref_roles={
            "record": ("target", "a", "b"),
        },
        fam_roles={"record": ()},
        boolean_roles={"record": ("target",)},
        advanced_enabled=True,
        run_lengths=(3,),
        max_conjunction_terms=2,
        max_complexity=20,
    )
    expected = (
        A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("target"),
                A.Sustained(
                    A.Bound(A.Ref("a"), "<=", None),
                    3,
                ),
            ),
        ),
        A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("target"),
                A.Conjunction((
                    A.Bound(A.Ref("a"), ">=", 0.0),
                    A.Bound(A.Ref("b"), "<=", 0.0),
                )),
            ),
        ),
    )

    signatures = {
        rule.signature()
        for rule in EnumerationProposer(grammar).propose()
    }

    assert all(
        is_admissible(rule, grammar)[0]
        and rule.signature() in signatures
        for rule in expected
    )


def test_generic_learned_conjunctions_are_admissible_not_hard_coded():
    # Round-18: the Boolean-definition grammar admits ANY conjunction of bounds over admissible
    # operands with <= 2 learned thresholds -- a generic ``(a < ?) AND (b > ?)`` and ``(a < ?) AND
    # (b > 0)`` are admissible, not just all-zero-threshold conjunctions or one hard-coded shape.
    # This proves the per-invariant structural special-casing is removed (the evaluator fits such
    # rules exactly; exhaustively enumerating them over every column tuple is a separate search-cost
    # trade-off, so admissibility is what the grammar guarantees).
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "=="),
        ref_roles={"record": ("target", "a", "b")},
        fam_roles={"record": ()},
        boolean_roles={"record": ("target",)},
        advanced_enabled=True,
        max_conjunction_terms=3,
        max_complexity=20,
    )
    one_learned = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Conjunction((
                A.Bound(A.Ref("a"), "<", None),
                A.Bound(A.Ref("b"), ">", 0.0),
            )),
        ),
    )
    two_learned = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Conjunction((
                A.Bound(A.Ref("a"), "<", None),
                A.Bound(A.Ref("b"), ">", None),
            )),
        ),
    )
    # A compound-operand conjunction (rolling pairwise-difference sign bound) is admitted generically
    # under a properly-windowed grammar -- see test_trajectory_template_combines_windowed_deficit_and
    # _lagged_slope, which exercises the real induced grammar rather than a bare stub.
    assert is_admissible(one_learned, grammar)[0]
    assert is_admissible(two_learned, grammar)[0]
    # Three joint learned thresholds exceed the evaluator's ceiling and must be rejected.
    three_learned = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Conjunction((
                A.Bound(A.Ref("a"), "<", None),
                A.Bound(A.Ref("b"), ">", None),
                A.Bound(A.Ref("target"), ">", None),
            )),
        ),
    )
    assert not is_admissible(three_learned, grammar)[0]


def test_conditioned_definition_respects_minimum_condition_support():
    # Round-22: Boolean and categorical definitions must clear the SAME conditional minimum-support
    # floor as comparisons and bands. A condition covering 10 of 2000 rows is not evidence of a law
    # no matter how cleanly it separates them, so it must be rejected before scoring.
    n = 2000
    rare = np.zeros(n, dtype=bool)
    rare[:10] = True
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "regime": np.where(rare, "rare", "common"),
        "signal": np.where(rare, 0.1, 0.9),
        "alert": rare.copy(),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("regime",),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="thin_condition")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(A.Ref("alert"), A.Bound(A.Ref("signal"), "<", None)),
        condition=A.Condition("regime", "==", ("rare",)),
    )

    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(min_condition_points=20, min_condition_fraction=0.01, band_mode="global"),
    ).evaluate(rule)

    assert not evaluation.accepted
    assert "condition support below minimum" in evaluation.reason

    # With the floor lowered to admit it, the same definition is scored normally (the guard is a
    # support policy, not a blanket rejection of conditioned definitions).
    permissive = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(min_condition_points=5, min_condition_fraction=0.001, band_mode="global"),
    ).evaluate(rule)
    assert "condition support below minimum" not in permissive.reason


def test_conditioned_definition_support_is_measured_on_gradeable_rows():
    # Round-23: the pre-grounding guard only counts the rows the CONDITION selects. A definition
    # grades a narrower population -- non-finite targets and unevaluable operands shrink it further
    # -- so a condition can clear the floor on paper while the rule is actually scored on a handful
    # of rows. The floor must therefore be re-applied to the rows that are genuinely gradeable.
    n = 2000
    selected = np.zeros(n, dtype=bool)
    selected[:30] = True
    alert = np.where(selected, 1.0, 0.0)
    # Twenty of the thirty condition rows carry no observable target, leaving ten gradeable rows.
    alert[10:30] = np.nan
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "regime": np.where(selected, "rare", "common"),
        "signal": np.where(selected, 0.1, 0.9),
        "alert": alert,
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("regime",),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="thin_after_grounding")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(A.Ref("alert"), A.Bound(A.Ref("signal"), "<", None)),
        condition=A.Condition("regime", "==", ("rare",)),
    )

    # 30 condition rows clear the pre-grounding floor outright, so only the post-grounding check
    # can reject here -- which is exactly the hole being closed.
    cfg = DiscoveryConfig(
        min_condition_points=20,
        min_condition_fraction=0.001,
        band_mode="global",
    )
    evaluation = DataOnlyEvaluator(dataset, cfg).evaluate(rule)

    assert not evaluation.accepted
    assert "after grounding" in evaluation.reason

    permissive = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            min_condition_points=5,
            min_condition_fraction=0.001,
            band_mode="global",
        ),
    ).evaluate(rule)
    assert "condition support below minimum" not in permissive.reason


def test_conditioned_definition_signature_keeps_its_condition():
    # Round-23: a conditioned rule states a strictly weaker claim than the unconditional one. If the
    # condition is dropped from the signature the two are indistinguishable, so a conditioned
    # discovery would be credited with recovering an unconditional known invariant.
    n = 400
    rare = np.zeros(n, dtype=bool)
    rare[:200] = True
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "regime": np.where(rare, "rare", "common"),
        "signal": np.where(rare, 0.1, 0.9),
        "alert": rare.copy(),
        "label": np.where(rare, "hot", "cold"),
        "is_hot": rare.copy(),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("regime", "is_hot", "label"),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="conditioned_signature")
    condition = A.Condition("regime", "==", ("rare",))

    for atom in (
        A.BooleanDefinition(A.Ref("alert"), A.Bound(A.Ref("signal"), "<", 0.5)),
        A.CategoryDefinition("label", (("is_hot", "hot"),), "cold"),
    ):
        plain = validation.rule_relations(A.Rule("record", atom), dataset)
        conditioned = validation.rule_relations(
            A.Rule("record", atom, condition=condition),
            dataset,
        )
        assert plain, f"unconditional {type(atom).__name__} produced no signature"
        assert conditioned, f"conditioned {type(atom).__name__} produced no signature"
        # The conditioned rule must not be mistakable for the unconditional one.
        assert plain.isdisjoint(conditioned)
        assert all(sig[0] == "conditional" for sig in conditioned)


def test_sustained_definition_grades_warmup_and_gap_rows_as_false():
    # Round-23: SUSTAINED is total. Before a group has produced `window` consecutive periods -- and
    # immediately after a cadence gap resets the run -- the predicate is definitively unmet, not
    # ungradeable. Excluding those rows let a target that is wrongly True during warm-up escape
    # scrutiny entirely and score a perfect hold rate.
    window = 5
    signal = np.full(60, 0.1)
    honest = _sustained(signal, 0.5, window)
    frame_kwargs = dict(
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=(),
        temporal_windows=(window,),
        max_lag=2,
    )
    timestamps = pd.date_range("2026-01-01", periods=signal.size, freq="1min")

    def _evaluate(alert):
        frame = profile_dataframe(
            pd.DataFrame({
                "timestamp": timestamps,
                "series_id": "a",
                "signal": signal,
                "alert": alert,
            }),
            **frame_kwargs,
        )
        dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="sustained_total")
        rule = A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("alert"),
                A.Sustained(A.Bound(A.Ref("signal"), "<", 0.5), window),
            ),
        )
        return DataOnlyEvaluator(dataset, DiscoveryConfig(band_mode="global")).evaluate(rule)

    truthful = _evaluate(honest)
    # Every row is now graded, including the warm-up rows the honest target reports as False.
    assert truthful.n_points == signal.size
    assert truthful.hold_rate == pytest.approx(1.0)

    # A target that claims the run is already sustained during warm-up is wrong on exactly those
    # rows, and must be charged for them instead of having them dropped.
    dishonest = honest.copy()
    dishonest[: window - 1] = True
    charged = _evaluate(dishonest)
    assert charged.n_points == signal.size
    assert charged.hold_rate < 1.0
    assert charged.hold_rate == pytest.approx(
        (signal.size - (window - 1)) / signal.size
    )


def test_sustained_definition_fits_threshold_and_round_trips():
    signal = np.resize(np.array([0.8, 0.4, 0.3, 0.2, 0.9, 0.7]), 240)
    alert = _sustained(signal, 0.5, 3)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=signal.size, freq="1min"),
        "series_id": "a",
        "signal": signal,
        "alert": alert,
    }))
    dataset, grammar = build_dataframe_grammar(frame, _base_spec(), name="sustained")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Sustained(A.Bound(A.Ref("signal"), "<", None), 3),
        ),
    )

    assert rule_from_dict(rule_to_dict(rule)) == rule
    assert atom_expr(rule.atom, {}) is not None
    assert is_admissible(rule, grammar)[0] is True
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.9),
    ).evaluate(rule)
    assert result.accepted
    assert result.hold_rate == 1.0
    threshold = next(iter(result.parameters["thresholds"].values()))
    assert 0.4 < threshold <= 0.7
    assert result.strictness == "definition"


def test_known_definition_thresholds_use_scale_aware_matching():
    signal = np.linspace(0.0, 0.01, 120)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=signal.size,
            freq="1min",
        ),
        "series_id": "a",
        "signal": signal,
        "alert": signal < 0.004,
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="threshold_matching",
    )
    learned = SimpleNamespace(
        rule=A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("alert"),
                A.Sustained(
                    A.Bound(A.Ref("signal"), "<", 0.004),
                    3,
                ),
            ),
        ),
        parameters={},
    )
    result = SimpleNamespace(
        dataset=dataset,
        portfolio=[learned],
    )

    wrong = KnownInvariant(
        "wrong",
        ":=",
        "alert",
        {
            "sustained": {
                "term": "signal",
                "op": "<",
                "threshold": 0.001,
                "window": 3,
            },
        },
    )
    matching = KnownInvariant(
        "matching",
        ":=",
        "alert",
        {
            "sustained": {
                "term": "signal",
                "op": "<",
                "threshold": 0.00400001,
                "window": 3,
            },
        },
    )

    assert recover_known(result, [wrong])["recall"] == 0.0
    assert recover_known(result, [matching])["recall"] == 1.0


def test_sustained_predicate_is_false_across_timestamp_gap():
    # A window that spans a cadence gap has not observed a full run of consecutive periods, so the
    # sustained predicate is definitively False there. Round-23: the row is graded rather than
    # dropped, so a target that claims the run is already sustained is charged for the mistake
    # instead of quietly escaping the population.
    frame = _profile(pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 00:01:00",
            "2026-01-01 00:10:00",
        ]),
        "series_id": "a",
        "signal": [1.0, 1.0, 1.0],
        "alert": [False, False, True],
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="sustained_gap",
    )
    predicted, valid = _predicate_population(
        A.Sustained(
            A.Bound(A.Ref("signal"), ">", 0.0),
            3,
        ),
        "record",
        {},
        dataset,
        {},
    )

    # The gap denies the run, but it does not make the row ungradeable.
    assert valid[2]
    assert not predicted[2]
    # Warm-up rows are decided the same way: no full run has occurred yet.
    assert valid.all()
    assert not predicted.any()

    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(band_mode="global"),
    ).evaluate(A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Sustained(A.Bound(A.Ref("signal"), ">", 0.0), 3),
        ),
    ))
    # The target asserts True on the gap row, so it must not score a perfect hold rate.
    assert evaluation.n_points == 3
    assert evaluation.hold_rate == pytest.approx(2.0 / 3.0)


def test_perfect_rare_definition_can_clear_baseline_lift():
    n = 400
    alert = np.zeros(n, dtype=bool)
    alert[:10] = True
    signal = alert.astype(float)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "alert": alert,
        "signal": signal,
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="rare_definition",
    )
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Bound(A.Ref("signal"), ">", 0.5),
        ),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.9,
            definition_min_lift=0.05,
        ),
    ).evaluate(rule)

    assert result.accepted
    assert result.parameters["baseline_agreement"] == 0.975
    assert result.threshold < 1.0


def test_three_term_generic_fixed_conjunction_is_accepted():
    n = 240
    low = np.resize(np.array([0.8, -0.4, -0.3, -0.2]), n)
    deficit = np.resize(np.array([-1.0, 2.0, 3.0, 4.0]), n)
    slope = np.resize(np.array([1.0, -1.0, -2.0, -3.0]), n)
    target = (low < 0.0) & (deficit > 0.0) & (slope <= 0.0)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "low": low,
        "deficit": deficit,
        "slope": slope,
        "trajectory_alert": target,
    }))
    dataset, grammar = build_dataframe_grammar(frame, _base_spec(), name="conjunction")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("trajectory_alert"),
            A.Conjunction((
                A.Bound(A.Ref("low"), "<", 0.0),
                A.Bound(A.Ref("deficit"), ">", 0.0),
                A.Bound(A.Ref("slope"), "<=", 0.0),
            )),
        ),
    )

    assert is_admissible(rule, grammar)[0] is True
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.9),
    ).evaluate(rule)
    assert result.accepted
    assert result.hold_rate == 1.0


def test_trajectory_template_combines_windowed_deficit_and_lagged_slope():
    n = 180
    ratio = np.linspace(0.9, 0.1, n)
    input_rate = np.full(n, 10.0)
    output_rate = np.full(n, 8.0)
    deficit = pd.Series(input_rate - output_rate).rolling(5, min_periods=5).sum().to_numpy()
    slope = pd.Series(ratio).diff(5).to_numpy()
    target = (ratio < 0.5) & (deficit > 0.0) & (slope <= 0.0)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "completeness_ratio_1h": ratio,
        "input_rate_bytes_per_min": input_rate,
        "output_rate_bytes_per_min": output_rate,
        "trajectory_alert": target,
    }))
    dataset, grammar = build_dataframe_grammar(frame, _base_spec(), name="trajectory")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("trajectory_alert"),
            A.Conjunction((
                A.Bound(A.Ref("completeness_ratio_1h"), "<", None),
                A.Bound(
                    A.Rolling(
                        A.Add((
                            A.Ref("input_rate_bytes_per_min"),
                            A.Scale(-1.0, A.Ref("output_rate_bytes_per_min")),
                        )),
                        5,
                        "SUM",
                    ),
                    ">",
                    0.0,
                ),
                A.Bound(A.Diff(A.Ref("completeness_ratio_1h"), 5), "<=", 0.0),
            )),
        ),
    )

    assert is_admissible(rule, grammar)[0] is True
    evaluator = DataOnlyEvaluator(dataset, DiscoveryConfig(hold_rate_threshold=0.9))
    result = evaluator.evaluate(rule)
    assert result.accepted
    # The learned cut lands within one grid step of the planted 0.5. It is scored on the parameter
    # holdout split only, so a one-step offset shows up there; that is the price of fitting the
    # threshold without looking at the rows it is graded on, not a semantic error.
    assert result.parameters["thresholds"][
        "completeness_ratio_1h < ?"
    ] == pytest.approx(0.5, abs=0.01)
    assert result.hold_rate >= 0.95

    # The stronger claim the plan actually makes (line 674): at the planted threshold the
    # conjunction matches the planted spans EXACTLY, on every row including the partial-window
    # warm-up rows -- no row is excused from the comparison.
    planted = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("trajectory_alert"),
            A.Conjunction((
                A.Bound(A.Ref("completeness_ratio_1h"), "<", 0.5),
            ) + rule.atom.predicate.predicates[1:]),
        ),
    )
    exact = evaluator.evaluate(planted)
    assert exact.n_points == n
    assert exact.hold_rate == 1.0
    rendered = {candidate.unparse() for candidate in EnumerationProposer(grammar).propose()}
    assert normalize_rule(rule).unparse() in rendered


def test_categorical_priority_map_is_evaluated_and_enumerated():
    n = 240
    is_true_loss = np.resize(np.array([False, True, True, False, False, False]), n)
    is_benign = np.resize(np.array([False, True, False, True, False, False]), n)
    is_artifact = np.resize(np.array([False, False, True, True, False, True]), n)
    label = np.full(n, "normal", dtype=object)
    label[is_artifact] = "artifact"
    label[is_benign] = "benign_burst"
    label[is_true_loss] = "true_loss"
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "is_true_loss": is_true_loss,
        "is_benign_burst": is_benign,
        "is_artifact": is_artifact,
        "label": label,
    }))
    dataset, grammar = build_dataframe_grammar(frame, _base_spec(), name="category")
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            target_column="label",
            cases=(
                ("is_true_loss", "true_loss"),
                ("is_benign_burst", "benign_burst"),
                ("is_artifact", "artifact"),
            ),
            default="normal",
        ),
    )

    assert is_admissible(rule, grammar)[0] is True
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.95),
    ).evaluate(rule)
    assert result.accepted
    assert result.hold_rate == 1.0
    rendered = {candidate.unparse() for candidate in EnumerationProposer(grammar).propose()}
    assert normalize_rule(rule).unparse() in rendered


def test_proxy_definition_credit_requires_exact_predicted_spans():
    n = 120
    values = np.linspace(0.0, 1.0, n)
    frame = profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "series_id": ["a"] * n,
            "target": values < 0.5,
            "x": values,
        }),
        time_index="timestamp",
        group_keys=("series_id",),
        temporal_windows=(1,),
        advanced=True,
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="exact_proxy_spans",
    )

    def evaluation(threshold):
        rule = A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("target"),
                A.Sustained(
                    A.Bound(A.Ref("x"), "<", threshold),
                    1,
                ),
            ),
        )
        return SimpleNamespace(rule=rule, parameters={})

    result = SimpleNamespace(dataset=dataset)
    assert validation._definition_matches_planted_mask(
        evaluation(0.5),
        result,
    )
    assert not validation._definition_matches_planted_mask(
        evaluation(0.4),
        result,
    )


def test_categorical_definition_scores_typed_identity_and_remains_enumerable():
    """A rule that only works because ``True == 1`` must be rejected.

    The two Boolean cases deliberately emit the other's typed label. Python equality calls every
    prediction correct; typed equality calls only the overlap and default rows correct.
    """
    n = 400
    phase = np.arange(n) % 4
    is_true = (phase == 0) | (phase == 2)
    is_one = (phase == 1) | (phase == 2)
    labels = np.empty(n, dtype=object)
    labels[phase == 0] = True
    labels[phase == 1] = 1
    labels[phase == 2] = 1
    labels[phase == 3] = "other"
    frame = profile_dataframe(
        pd.DataFrame({
            "is_true": is_true,
            "is_one": is_one,
            "label": pd.Series(labels, dtype=object),
        }),
        condition_columns=("is_true", "is_one", "label"),
        advanced=True,
    )
    dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="typed_category_definition",
    )
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            target_column="label",
            cases=(
                ("is_true", 1),
                ("is_one", True),
            ),
            default="other",
        ),
    )

    domain = grammar.condition_columns["label"]
    assert len({typed_group_key(value) for value in domain}) == 3
    assert normalize_rule(rule).unparse() in {
        normalize_rule(candidate).unparse()
        for candidate in EnumerationProposer(grammar).propose()
        if isinstance(candidate.atom, A.CategoryDefinition)
    }

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.8,
            definition_min_lift=0.0,
        ),
    ).evaluate(rule)

    assert not result.accepted
    assert result.hold_rate == 0.5
    assert result.parameters["baseline_agreement"] == 0.5


def test_categorical_enumeration_is_boolean_column_rename_invariant():
    def candidate_count(first_flag):
        grammar = Grammar(
            binders=("record",),
            ops=("~=", "=="),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
            condition_columns={
                first_flag: (False, True),
                "flag_b": (False, True),
                "flag_c": (False, True),
                "category": (
                    "baseline",
                    "class_a",
                    "class_b",
                    "class_c",
                ),
            },
            advanced_enabled=True,
        )
        return sum(
            isinstance(rule.atom, A.CategoryDefinition)
            for rule in EnumerationProposer(grammar).propose()
        )

    lexical = candidate_count("is_flag_a")
    neutral = candidate_count("flag_a")

    assert lexical == neutral
    assert neutral > 0


def test_boolean_refs_are_excluded_from_numeric_search():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "!=", "<=", ">=", "<|>", "~\u221d"),
        ref_roles={
            "record": ("flag_a", "flag_b", "value"),
        },
        fam_roles={"record": ()},
        boolean_roles={"record": ("flag_a", "flag_b")},
        temporal_enabled=True,
        max_lag=2,
        windows=(2,),
        max_degree=2,
    )

    rules = EnumerationProposer(grammar).propose()
    compare_rules = [
        rule
        for rule in rules
        if isinstance(rule.atom, A.Compare)
    ]

    assert A.Rule(
        "record",
        A.Compare(A.Ref("flag_a"), "==", A.Ref("flag_b")),
    ).signature() in {rule.signature() for rule in compare_rules}
    assert not any(
        rule.atom.op in {"<", "<=", ">", ">=", "~\u221d"}
        and (
            "flag_a" in rule.atom.unparse()
            or "flag_b" in rule.atom.unparse()
        )
        for rule in compare_rules
    )
    inadmissible = A.Rule(
        "record",
        A.Compare(A.Ref("flag_a"), ">=", A.Const(0)),
    )
    assert not is_admissible(inadmissible, grammar)[0]


def test_boolean_related_aggregates_keep_only_boolean_equality():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "!=", "<|>", ">=", "<="),
        ref_roles={"record": ("signal", "alert")},
        fam_roles={"record": ()},
        related_roles={"record": ("event_mask", "related_total")},
        boolean_roles={"record": ("alert",)},
        boolean_related_roles={"record": ("event_mask",)},
        max_complexity=8,
    )

    rules = EnumerationProposer(grammar).propose()
    expected = A.Rule(
        "record",
        A.Compare(
            A.Ref("alert"),
            "==",
            A.RelatedAgg("event_mask"),
        ),
    )

    assert normalize_rule(expected).signature() in {
        rule.signature()
        for rule in rules
    }
    assert all(
        not (
            isinstance(rule.atom, A.Compare)
            and (
                rule.atom.left == A.RelatedAgg("event_mask")
                or rule.atom.right == A.RelatedAgg("event_mask")
            )
            and rule.atom.op != "=="
        )
        for rule in rules
    )


def test_categorical_proxy_vocabulary_is_parameterized():
    vocabulary = synth.Vocab(
        category_target="kind",
        category_flags=("flag_x", "flag_y", "flag_z"),
        category_values=("class_x", "class_y", "class_z"),
        category_default="baseline",
    )
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=60,
        noise=0.0,
        seed=0,
        vocab=vocabulary,
        families=("categorical",),
    )

    assert set(data.row_context) >= {
        "kind",
        "flag_x",
        "flag_y",
        "flag_z",
    }
    assert data.planted["categorical"] == {
        (
            "kind",
            (
                ("flag_x", typed_signature_value("class_x")),
                ("flag_y", typed_signature_value("class_y")),
                ("flag_z", typed_signature_value("class_z")),
            ),
            typed_signature_value("baseline"),
        ),
    }


def test_categorical_priority_requires_overlapping_cases():
    n = 120
    is_true_loss = np.resize(
        np.array([True, False, False], dtype=bool),
        n,
    )
    is_benign = np.resize(
        np.array([False, True, False], dtype=bool),
        n,
    )
    label = np.where(
        is_true_loss,
        "true_loss",
        np.where(is_benign, "benign_burst", "normal"),
    )
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "is_true_loss": is_true_loss,
        "is_benign_burst": is_benign,
        "label": label,
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="ambiguous_priority",
    )
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            target_column="label",
            cases=(
                ("is_true_loss", "true_loss"),
                ("is_benign_burst", "benign_burst"),
            ),
            default="normal",
        ),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.9),
    ).evaluate(rule)

    assert not result.accepted
    assert "priority" in result.reason


def test_categorical_definition_ignores_nullable_targets_and_cases():
    target = pd.Series(
        ["true_loss", "benign_burst", pd.NA, "normal", np.nan],
        dtype="string",
    )
    true_loss = pd.Series([True, True, pd.NA, False, False], dtype="boolean")
    benign = pd.Series([True, True, False, False, pd.NA], dtype="boolean")
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        "series_id": "a",
        "is_true_loss": true_loss,
        "is_benign_burst": benign,
        "label": target,
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="nullable_category",
    )
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            target_column="label",
            cases=(
                ("is_true_loss", "true_loss"),
                ("is_benign_burst", "benign_burst"),
            ),
            default="normal",
        ),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            hold_rate_threshold=0.5,
            definition_min_lift=0.0,
        ),
    ).evaluate(rule)

    assert result.n_points == 3
    assert np.isfinite(result.hold_rate)


def test_learned_threshold_ceiling_is_fail_loud_not_silently_truncated():
    # A learned-threshold bound is swept exhaustively over every observed midpoint by default; a
    # positive max_threshold_candidates is a fail-loud safety ceiling, never a silent quantile grid
    # that could miss the true threshold.
    from autogram.discovery.propose import SearchSpaceTruncatedError

    rng = np.random.default_rng(0)
    n = 240
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "signal": rng.normal(size=n),
        "alert": rng.random(n) < 0.3,
    }))
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="threshold_ceiling",
    )
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Bound(A.Ref("signal"), "<", None),
        ),
    )

    with pytest.raises(SearchSpaceTruncatedError, match="max_threshold_candidates=8"):
        DataOnlyEvaluator(
            dataset,
            DiscoveryConfig(max_threshold_candidates=8),
        ).evaluate(rule)

    # The default (0) sweeps every fit-distinct midpoint without raising.
    result = DataOnlyEvaluator(dataset, DiscoveryConfig()).evaluate(rule)
    assert result.reason


def test_fit_threshold_candidates_are_fit_only_exhaustive_with_edges():
    # Candidates come from the fit rows only (an evaluation-only value never appears), span every
    # distinct-value midpoint (exact, including across mixed-label duplicates), and add below-min /
    # above-max edge sentinels so an all-True / all-False optimum is reachable.
    from autogram.discovery.evaluate import _fit_threshold_candidates

    effective = np.array([1.0, 1.0, 2.0, 3.0, 100.0], dtype=float)
    fit_mask = np.array([True, True, True, True, False])  # the 100.0 row is evaluation-only

    candidates = _fit_threshold_candidates(effective, fit_mask, DiscoveryConfig())

    assert all(c < 50 for c in candidates), "an evaluation-only value leaked into fit candidates"
    # midpoints of distinct fit values {1,2,3} -> {1.5, 2.5} plus the two edge sentinels
    assert any(abs(c - 1.5) < 1e-9 for c in candidates)
    assert any(abs(c - 2.5) < 1e-9 for c in candidates)
    assert min(candidates) < 1.0 and max(candidates) > 3.0


def test_fit_threshold_candidates_are_overflow_safe_at_float_extremes():
    # Near the float maximum a naive (a+b)/2 overflows to inf and yields no separating threshold;
    # the overflow-safe midpoint must produce a finite separator between two extreme finite values.
    from autogram.discovery.evaluate import _fit_threshold_candidates

    effective = np.array([9e307, 1e308], dtype=float)
    fit_mask = np.array([True, True])

    candidates = _fit_threshold_candidates(effective, fit_mask, DiscoveryConfig())

    interior = [c for c in candidates if 9e307 < c < 1e308]
    assert interior, "no finite separating threshold between extreme finite values"
    assert all(np.isfinite(c) for c in candidates)


def test_fit_threshold_candidates_are_overflow_safe_at_antipodal_extremes():
    # Opposite-sign extremes make ``b - a`` overflow (MAX - (-MAX) == inf), which previously produced
    # only [-inf, inf] and blocked any separating threshold. The half-sum fallback must yield a finite
    # interior separator (~0) between -MAX and +MAX, and every candidate must be finite (also so the
    # fitted threshold stays JSON-serialisable).
    from autogram.discovery.evaluate import _fit_threshold_candidates

    fmax = float(np.finfo(float).max)
    effective = np.array([-fmax, fmax], dtype=float)
    fit_mask = np.array([True, True])

    candidates = _fit_threshold_candidates(effective, fit_mask, DiscoveryConfig())

    assert all(np.isfinite(c) for c in candidates), f"non-finite candidate in {candidates}"
    interior = [c for c in candidates if -fmax < c < fmax]
    assert interior, "no finite separating threshold between antipodal extremes"
    # the separator must actually split the two rows (< t on one side, >= t on the other)
    t = interior[0]
    assert (effective < t).tolist() == [True, False]


def test_direct_definition_recovers_exact_separating_threshold():
    # A perfectly separable direct definition must be accepted: the exhaustive fit-distinct sweep
    # localises the exact threshold even with repeated values on either side of the boundary.
    n = 200
    signal = np.concatenate([
        np.full(n // 2, 0.2),
        np.full(n // 2, 0.8),
    ])
    alert = signal < 0.5
    frame = profile_dataframe(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "signal": signal,
        "alert": alert,
    }), time_index="timestamp", group_keys=("series_id",),
        condition_columns=("alert",), advanced=True)
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="separable")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(A.Ref("alert"), A.Bound(A.Ref("signal"), "<", None)),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.9),
    ).evaluate(rule)

    assert result.accepted
    threshold = result.parameters["thresholds"][A.Bound(A.Ref("signal"), "<", None).unparse()]
    assert 0.2 < threshold < 0.8


def test_advanced_logic_stays_disabled_without_profile_capability():
    frame = profile_dataframe(pd.DataFrame({
        "signal": [1.0, 2.0, 3.0],
        "alert": [False, True, True],
    }))
    _dataset, grammar = build_dataframe_grammar(frame, _base_spec(), name="disabled")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Sustained(A.Bound(A.Ref("signal"), ">", None), 2),
        ),
    )

    assert is_admissible(rule, grammar)[0] is False


def test_evaluator_rejects_boolean_refs_in_numeric_comparisons():
    frame = profile_dataframe(
        pd.DataFrame({
            "signal": [0.0, 1.0, 2.0],
            "alert": [False, True, False],
        }),
        condition_columns=("alert",),
        advanced=True,
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="boolean_boundary",
    )
    evaluator = DataOnlyEvaluator(dataset, DiscoveryConfig())

    ordered = evaluator.evaluate(
        A.Rule(
            "record",
            A.Compare(A.Ref("alert"), ">=", A.Const(0.0)),
        )
    )
    mixed = evaluator.evaluate(
        A.Rule(
            "record",
            A.Compare(A.Ref("alert"), "==", A.Ref("signal")),
        )
    )
    unknown_aggregation = evaluator.evaluate(
        A.Rule(
            "record",
            A.Compare(
                A.Rolling(A.Ref("signal"), 2, "BOGUS"),
                "~=",
                A.Ref("signal"),
            ),
        )
    )
    invalid_temporal = evaluator.evaluate(
        A.Rule(
            "record",
            A.Compare(
                A.Diff(A.Ref("signal"), -1),
                "==",
                A.Ref("signal"),
            ),
        )
    )
    null_dataset = validation._runtime_null_dataset(
        dataset,
        seed=0,
        definition_targets=False,
    )

    assert not ordered.accepted
    assert not mixed.accepted
    assert not unknown_aggregation.accepted
    assert not invalid_temporal.accepted
    assert set(null_dataset.observed.col("alert").tolist()) <= {0.0, 1.0}
    assert ordered.reason == "Boolean refs cannot enter numeric comparisons"
    assert mixed.reason == "Boolean refs cannot enter numeric comparisons"
    assert unknown_aggregation.reason == "unknown intrinsic aggregation 'BOGUS'"
    assert invalid_temporal.reason == "temporal steps must be positive"


def test_learned_definition_rejects_independent_imbalanced_target():
    rng = np.random.default_rng(23)
    n = 240
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "signal": rng.normal(size=n),
        "alert": rng.random(n) < 0.1,
    }))
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="definition_null")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Sustained(A.Bound(A.Ref("signal"), "<", None), 3),
        ),
    )

    result = DataOnlyEvaluator(dataset, DiscoveryConfig()).evaluate(rule)

    assert not result.accepted
    assert "baseline" in result.reason


def test_advanced_logic_is_the_final_capability_tier():
    tier = _capability_tiers()[-1]
    assert tier["advanced"] is True
    widened = _widen_spec(
        _base_spec(),
        temporal=True,
        max_lag=60,
        windows=(45, 60),
        advanced=True,
        run_lengths=(10,),
        max_conjunction_terms=3,
    )
    assert widened.advanced_enabled is True
    assert widened.run_lengths == (10,)
    assert widened.max_conjunction_terms == 3


def test_advanced_known_signatures_and_recovery_fields():
    sustained = KnownInvariant(
        "static",
        ":=",
        "static_alert",
        {
            "sustained": {
                "term": "ratio_1h",
                "op": "<",
                "threshold": 0.99,
                "window": 10,
            },
        },
    )
    conjunction = KnownInvariant(
        "trajectory",
        ":=",
        "trajectory_alert",
        {
            "and": [
                {"bound": ["ratio_1h", "<", 0.98]},
                {"bound": [{"roll_sum": [{"difference": ["input", "output"]}, 45]}, ">", 0]},
                {"bound": [{"delta": ["ratio_1h", 45]}, "<=", 0]},
            ],
        },
    )
    reversed_conjunction = KnownInvariant(
        "trajectory_reversed",
        ":=",
        "trajectory_alert",
        {"and": list(reversed(conjunction.rhs["and"]))},
    )
    categorical = KnownInvariant(
        "labels",
        ":=",
        "label",
        {
            "priority": [
                {"when": "is_true_loss", "value": "true_loss"},
                {"when": "is_benign_burst", "value": "benign_burst"},
                {"when": "is_artifact", "value": "artifact"},
            ],
            "default": "normal",
        },
    )

    assert _signature(sustained)[0] == "sustained_definition"
    assert _signature(conjunction)[0] == "conjunction_definition"
    assert _signature(reversed_conjunction) == _signature(conjunction)
    assert _signature(categorical)[0] == "categorical_definition"
    assert shapes_for_invariant(sustained) == ["sustained"]
    assert shapes_for_invariant(conjunction) == ["conjunction"]
    assert shapes_for_invariant(categorical) == ["categorical"]

    from autogram.discovery import validate as validation
    result = type("Result", (), {"portfolio": []})()
    planted = {
        "sustained": {_signature(sustained)[1]},
        "conjunction": {_signature(conjunction)[1]},
        "categorical": {_signature(categorical)[1]},
    }
    original = validation.portfolio_relations
    try:
        validation.portfolio_relations = lambda _result: {
            _signature(sustained),
            _signature(conjunction),
            _signature(categorical),
        }
        recovery = score_recovery(result, planted)
    finally:
        validation.portfolio_relations = original
    # Structural signatures alone no longer earn sustained/conjunction credit: the scorer requires
    # an actual Evaluation whose predictions match the planted spans exactly. Dedicated live tests
    # above cover that path; this mocked portfolio has no dataset/predictions, so credit stays zero.
    assert recovery.sustained == 0.0
    assert recovery.conjunction == 0.0
    assert recovery.categorical == 1.0


def test_known_definition_matching_checks_fitted_threshold_value():
    signal = np.resize(np.array([0.8, 0.4, 0.3, 0.2]), 80)
    frame = _profile(pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=80, freq="1min"),
        "series_id": "a",
        "signal": signal,
        "alert": _sustained(signal, 0.99, 3),
    }))
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="threshold_match")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Sustained(A.Bound(A.Ref("signal"), "<", None), 3),
        ),
    )
    known = [
        KnownInvariant(
            "static",
            ":=",
            "alert",
            {
                "sustained": {
                    "term": "signal",
                    "op": "<",
                    "threshold": 0.99,
                    "window": 3,
                },
            },
        ),
    ]

    wrong = type("Result", (), {
        "dataset": dataset,
        "portfolio": [
            type("Evaluation", (), {
                "rule": rule,
                "parameters": {"thresholds": {"signal < ?": 0.5}},
            })(),
        ],
    })()
    correct = type("Result", (), {
        "dataset": dataset,
        "portfolio": [
            type("Evaluation", (), {
                "rule": rule,
                "parameters": {"thresholds": {"signal < ?": 0.99001}},
            })(),
        ],
    })()

    assert recover_known(wrong, known)["recall"] == 0.0
    assert recover_known(correct, known)["recall"] == 1.0


def test_advanced_proxy_generators_plant_each_definition_shape():
    for shape in ("sustained", "conjunction", "categorical"):
        data = synth.make_synthetic(
            n_entities=3,
            n_snapshots=120,
            noise=0.0,
            seed=5,
            families=(shape,),
            temporal_window=5,
        )
        assert set(data.planted) == {shape}
        assert data.planted[shape]
        if shape == "categorical":
            assert set(data.row_context) >= {
                "flag_a",
                "flag_b",
                "flag_c",
                "category",
            }


def test_conjunction_is_decided_when_an_observed_conjunct_is_false():
    # Round-24: a conjunction whose operands are not all evaluable is still DECIDED when some
    # conjunct we could evaluate is already False -- a missing operand cannot rescue it. Dropping
    # those rows excused the definition on exactly the rows where its target most often disagrees,
    # so a run-length/conjunction rule no longer matched its planted spans exactly.
    n = 40
    left = np.full(n, 1.0)
    # The second conjunct is unobservable on the back half of the frame.
    right = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, np.nan)])
    # The FIRST conjunct is false everywhere on that back half, so those rows are decided False.
    left[n // 2:] = -1.0
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "left_signal": left,
        "right_signal": right,
        # A target that wrongly claims the conjunction holds on the undecidable-looking rows.
        "alert": np.concatenate([np.ones(n // 2, dtype=bool), np.ones(n // 2, dtype=bool)]),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=(),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="three_valued_and")
    predicate = A.Conjunction((
        A.Bound(A.Ref("left_signal"), ">", 0.0),
        A.Bound(A.Ref("right_signal"), ">", 0.0),
    ))

    predicted, valid = _predicate_population(predicate, "record", {}, dataset, {})

    # Front half: both conjuncts observed and true.
    assert valid[: n // 2].all()
    assert predicted[: n // 2].all()
    # Back half: the second conjunct is missing, but the first is observed False, so the
    # conjunction is decided False rather than dropped.
    assert valid[n // 2:].all()
    assert not predicted[n // 2:].any()

    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(band_mode="global"),
    ).evaluate(A.Rule("record", A.BooleanDefinition(A.Ref("alert"), predicate)))
    # The target claims True on all 40 rows, so it is wrong on the 20 decided-False rows.
    assert evaluation.n_points == n
    assert evaluation.hold_rate == pytest.approx(0.5)


def test_conjunction_stays_undecided_when_no_observed_conjunct_is_false():
    # The complement of the rule above. When every conjunct that could be evaluated is True and
    # another is genuinely unobservable, the conjunction is undecidable and must NOT be graded.
    # Treating the missing conjunct as vacuously true would invent a verdict: measured against the
    # emitted GTIB trajectory alert that convention grades more rows but drops agreement to 0.954.
    n = 20
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "left_signal": np.full(n, 1.0),
        "right_signal": np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, np.nan)]),
        "alert": np.ones(n, dtype=bool),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=(),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="undecided_and")
    predicted, valid = _predicate_population(
        A.Conjunction((
            A.Bound(A.Ref("left_signal"), ">", 0.0),
            A.Bound(A.Ref("right_signal"), ">", 0.0),
        )),
        "record",
        {},
        dataset,
        {},
    )

    assert valid[: n // 2].all()
    assert predicted[: n // 2].all()
    assert not valid[n // 2:].any()


def test_unidentifiable_categorical_priority_is_rejected_despite_perfect_agreement():
    """Round-25: (S5) in docs/autogram_guarantees.md is a real conjunct, not decoration.

    A priority map whose precedence edges are never exercised predicts the target perfectly, so it
    clears triviality, support, variation, the Wilson bound, the per-group gate and the baseline
    lift. It is still rejected, because the data never pinned down the ORDER of its cases: any
    permutation of the unexercised cases fits equally well, so the rule's parameters are not
    determined. Documenting `Acc` without this conjunct would make the published predicate false.
    """
    n = 400
    # The two case columns never fire on the same row, so their relative priority is unobservable.
    is_hot = np.zeros(n, dtype=bool); is_hot[:120] = True
    is_cold = np.zeros(n, dtype=bool); is_cold[200:320] = True
    label = np.where(is_hot, "hot", np.where(is_cold, "cold", "none"))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "is_hot": is_hot,
        "is_cold": is_cold,
        "label": label,
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("is_hot", "is_cold", "label"),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="unidentifiable")
    rule = A.Rule(
        "record",
        A.CategoryDefinition("label", (("is_hot", "hot"), ("is_cold", "cold")), "none"),
    )

    evaluation = DataOnlyEvaluator(dataset, DiscoveryConfig(band_mode="global")).evaluate(rule)

    assert not evaluation.accepted
    assert "identifiable" in evaluation.reason

    # Make the edge observable -- one row where both fire, so "hot wins" is witnessed -- and the
    # very same rule is now scored normally.
    df.loc[50, "is_cold"] = True
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("is_hot", "is_cold", "label"),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="identifiable")
    witnessed = DataOnlyEvaluator(dataset, DiscoveryConfig(band_mode="global")).evaluate(rule)
    assert "identifiable" not in witnessed.reason


def test_categorical_baseline_handles_mixed_type_labels():
    # Round-26: a categorical target may legitimately mix types after a CSV/parquet round-trip (an
    # integer code beside a string label). The majority-class baseline must not sort those values --
    # `np.unique` raises TypeError on mixed types, so a perfectly valid rule crashed the evaluator
    # instead of being scored.
    n = 300
    is_hot = np.zeros(n, dtype=bool); is_hot[:100] = True
    label = np.empty(n, dtype=object)
    label[:100] = "hot"
    label[100:] = 0            # an integer code alongside a string label
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "is_hot": is_hot,
        "label": label,
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("is_hot", "label"),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="mixed_labels")
    rule = A.Rule("record", A.CategoryDefinition("label", (("is_hot", "hot"),), 0))

    evaluation = DataOnlyEvaluator(dataset, DiscoveryConfig(band_mode="global")).evaluate(rule)

    # The point is that it is *scored* rather than raising; the mapping is exact here.
    assert evaluation.hold_rate == 1.0
