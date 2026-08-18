"""Deterministic end-to-end recovery of GTIB emitted identities."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.calibrate import _capability_tiers, _split_known, _widen_spec
from autogram.discovery import validate as validation
from autogram.discovery.archive import _same_fitted_semantics
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.known import KnownInvariant, _signature, load_known, recover_known
from autogram.discovery.loop import build_dataframe_grammar, run_prepared
from autogram.discovery.propose import EnumerationProposer, normalize_rule
from autogram.discovery.validate import (
    null_definitions_at,
    null_equalities_at,
    null_temporal_at,
    prepare_runtime_null_controls,
)
from autogram.dsl import ast as A
from autogram.dsl.evaluate import typed_group_key
import pandas as pd

from autogram.loader.gtib import prepare_gtib_files, prepare_gtib_raw
from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="gtib",
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
        max_degree=2,
        advanced_enabled=True,
        run_lengths=(10,),
        max_conjunction_terms=3,
    )


def _rule_set():
    ratio = A.Rule(
        "record",
        A.Compare(
            A.Ref("completeness_ratio"),
            "==",
            A.Div(
                A.Ref("output_rate_bytes_per_min"),
                A.Ref("input_rate_bytes_per_min"),
            ),
        ),
    )
    windowed = A.Rule(
        "record",
        A.Compare(
            A.Ref("completeness_ratio_1h"),
            "==",
            A.Div(
                A.Rolling(A.Ref("output_rate_bytes_per_min"), 60, "SUM"),
                A.Rolling(A.Ref("input_rate_bytes_per_min"), 60, "SUM"),
            ),
        ),
    )
    input_from_raw = A.Rule(
        "record",
        A.Compare(
            A.Ref("input_rate_bytes_per_min"),
            "==",
            A.RelatedAgg("raw_input_rate"),
        ),
    )
    static = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("static_alert"),
            A.Sustained(
                A.Bound(A.Ref("completeness_ratio_1h"), "<", None),
                10,
            ),
        ),
    )
    trajectory = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("traj_alert"),
            A.Conjunction((
                A.Bound(A.Ref("completeness_ratio_1h"), "<", None),
                A.Bound(
                    A.Rolling(
                        A.Add((
                            A.Ref("input_rate_bytes_per_min"),
                            A.Scale(-1.0, A.Ref("output_rate_bytes_per_min")),
                        )),
                        45,
                        "SUM",
                    ),
                    ">",
                    0.0,
                ),
                A.Bound(
                    A.Diff(A.Ref("completeness_ratio_1h"), 45),
                    "<=",
                    0.0,
                ),
            )),
        ),
    )
    oracle = A.Rule(
        "record",
        A.Compare(A.Ref("oracle_alert"), "==", A.Ref("is_true_loss")),
    )
    labels = A.Rule(
        "record",
        A.CategoryDefinition(
            "label",
            (
                ("is_true_loss", "true_loss"),
                ("is_benign_burst", "benign_burst"),
                ("is_artifact", "artifact"),
            ),
            "normal",
        ),
    )
    monotone_loss = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("cum_lost_bytes"), 1), ">=", A.Const(0)),
    )
    benign_loss = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("cum_lost_bytes"), 1), "~=", A.Const(0)),
        condition=A.Condition("label", "in", ("benign_burst", "artifact")),
    )
    true_loss = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("cum_lost_bytes"), 1), ">", A.Const(0)),
        condition=A.Condition("label", "==", ("true_loss",)),
    )
    return (
        ratio,
        windowed,
        input_from_raw,
        static,
        trajectory,
        oracle,
        labels,
        monotone_loss,
        benign_loss,
        true_loss,
    )


def test_gtib_emitted_identities_are_recovered_end_to_end():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="gtib")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-8,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    )

    evaluations = [evaluator.evaluate(rule) for rule in _rule_set()]

    assert all(evaluation.accepted for evaluation in evaluations), [
        (evaluation.rule.unparse(), evaluation.reason, evaluation.hold_rate)
        for evaluation in evaluations
        if not evaluation.accepted
    ]
    assert evaluations[3].parameters["thresholds"]
    assert evaluations[4].parameters["thresholds"]


def test_gtib_raw_counters_are_monotone_outside_resets():
    raw = prepare_gtib_raw(pd.read_csv("data/gtib-emulation/timeseries_raw.csv"))
    dataset, _grammar = build_dataframe_grammar(raw, _base_spec(), name="gtib_raw")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-8,
            hold_rate_threshold=0.99,
            band_mode="global",
        ),
    )
    for column in ("collector_input_counted", "presenter_output_counted"):
        result = evaluator.evaluate(A.Rule(
            "record",
            A.Compare(A.Diff(A.Ref(column), 1), ">=", A.Const(0)),
            condition=A.Condition("reset_flag", "==", (False,)),
        ))
        assert result.accepted, (column, result.reason, result.hold_rate)


def test_gtib_hidden_state_boundary_aggregates_are_recovered_from_raw():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="gtib_hidden")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-8,
            hold_rate_threshold=0.99,
            band_mode="global",
        ),
    )
    for target, role in (
        ("backlog_bytes", "raw_backlog"),
        ("cum_lost_bytes", "raw_cum_lost"),
    ):
        result = evaluator.evaluate(A.Rule(
            "record",
            A.Compare(A.Ref(target), "==", A.RelatedAgg(role)),
        ))
        assert result.accepted, (target, result.reason, result.hold_rate)


def test_gtib_event_spans_define_each_per_minute_mask():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="gtib_events")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-8,
            hold_rate_threshold=0.99,
            band_mode="global",
        ),
    )
    for target, role in (
        ("is_true_loss", "event_true_loss"),
        ("is_benign_burst", "event_benign_burst"),
        ("is_artifact", "event_artifact"),
    ):
        result = evaluator.evaluate(A.Rule(
            "record",
            A.Compare(A.Ref(target), "==", A.RelatedAgg(role)),
        ))
        assert result.accepted, (target, result.reason, result.hold_rate)


def test_gtib_steady_normal_ratio_recovers_healthy_band():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="gtib_band")
    rule = A.Rule(
        "record",
        A.BandDefinition(A.Ref("completeness_ratio"), None),
        condition=A.Condition(
            "",
            "all",
            (
                A.Condition("archetype", "==", ("steady",)),
                A.Condition("label", "==", ("normal",)),
            ),
        ),
    )
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.03,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(rule)

    assert result.accepted, (result.reason, result.hold_rate)
    assert abs(result.parameters["center"] - 0.998) < 0.01


def test_checked_in_gtib_known_files_use_supported_shapes():
    for path in ("configs/gtib_known.yaml", "configs/gtib_raw_known.yaml"):
        known = load_known(path)
        assert known
        assert all(_signature(invariant) is not None for invariant in known), path


def test_checked_in_gtib_known_catalog_reaches_full_recall():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    search = SearchConfig(
        max_complexity=16,
        max_add_arity=3,
        max_rules=400_000,
        max_nonlinear_leaves=64,
        max_linear_leaves=32,
        max_conditioned_rules=0,
        max_lag=45,
        windows=(10, 45, 60),
        seed=0,
    )
    dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        search_cfg=search,
        name="gtib_catalog",
    )
    result = run_prepared(
        dataset,
        grammar,
        discovery_cfg=DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.62,
            band_mode="adaptive",
            seed=0,
        ),
        search_cfg=search,
    )

    report = recover_known(result, load_known("configs/gtib_known.yaml"))

    assert report["recall"] == 1.0, [
        invariant
        for invariant in report["invariants"]
        if not invariant["recovered"]
    ]
    # No conditioned rule may be *redundant* with a coexisting unconditional rule: a
    # conditioned refinement is only allowed to survive alongside the same unconditional
    # atom+fitted-semantics when the unconditional law is strictly weaker (lower hold-rate),
    # so it does not logically subsume the refinement. If the unconditional held at least as
    # strongly, the archive must have already dropped the conditioned copy.
    by_rule = {
        evaluation.rule: evaluation
        for evaluation in result.portfolio
    }
    for evaluation in result.portfolio:
        if evaluation.rule.condition is None:
            continue
        unconditional = replace(
            evaluation.rule,
            condition=None,
        )
        counterpart = by_rule.get(unconditional)
        if counterpart is None:
            continue
        if _same_fitted_semantics(evaluation, counterpart):
            assert counterpart.hold_rate + 1e-12 < evaluation.hold_rate, (
                evaluation.rule.unparse(),
                counterpart.hold_rate,
                evaluation.hold_rate,
            )


def test_gtib_candidate_space_remains_bounded_after_materialization():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        search_cfg=SearchConfig(
            max_nonlinear_leaves=64,
            max_linear_leaves=32,
            max_conditioned_rules=0,
            max_lag=45,
        ),
        name="gtib_bound",
    )

    candidates = EnumerationProposer(grammar).propose()
    ceiling = 400_000
    budgeted = EnumerationProposer(
        replace(grammar, max_rules=ceiling)
    ).propose()
    rendered = {candidate.unparse() for candidate in budgeted}
    required = [
        *_rule_set(),
        A.Rule(
            "record",
            A.BandDefinition(A.Ref("completeness_ratio"), None),
            condition=A.Condition(
                "",
                "all",
                (
                    A.Condition("archetype", "==", ("steady",)),
                    A.Condition("label", "==", ("normal",)),
                ),
            ),
        ),
    ]

    assert len(grammar.refs_for("record")) == 12
    assert len(candidates) < ceiling
    assert len(budgeted) == len(candidates)
    assert all(normalize_rule(rule).unparse() in rendered for rule in required)


def test_runtime_null_controls_match_gtib_search_multiplicity():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    search = SearchConfig(
        max_rules=400_000,
        max_nonlinear_leaves=64,
        max_linear_leaves=32,
        max_conditioned_rules=0,
        max_lag=45,
        windows=(10, 45, 60),
    )
    dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        search_cfg=search,
        name="gtib_runtime_null",
    )
    assert grammar.binders == ("record",)
    real_rules = EnumerationProposer(grammar).propose()

    controls = validation.prepare_runtime_null_controls(
        dataset,
        grammar,
        search,
        seed=0,
        rules=real_rules,
    )

    expected = {
        "all": len(real_rules),
        "equalities": sum(
            (
                isinstance(rule.atom, A.Compare)
                and rule.atom.op in {
                    "~=",
                    "==",
                    "~\u221d",
                    ">=",
                    "<=",
                    ">",
                    "<",
                    "<|>",
                }
            )
            or isinstance(rule.atom, A.BandDefinition)
            for rule in real_rules
        ),
        "temporal": sum(
            validation._rule_has_temporal(rule)
            for rule in real_rules
        ),
        "definitions": sum(
            isinstance(
                rule.atom,
                (A.BooleanDefinition, A.CategoryDefinition),
            )
            for rule in real_rules
        ),
    }
    assert controls.candidate_counts == expected
    assert controls.null.ds.observed.matrix.shape == (
        dataset.observed.matrix.shape
    )
    assert set(controls.null.ds.relations) == set(dataset.relations)
    assert set(controls.null.G.condition_columns) == set(
        grammar.condition_columns
    )
    for column in (
        "is_true_loss",
        "is_benign_burst",
        "is_artifact",
        "static_alert",
        "oracle_alert",
        "traj_alert",
    ):
        assert set(
            controls.null.ds.observed.col(column).tolist()
        ) <= {0.0, 1.0}
    null_context = controls.temporal_null.ds.row_context
    source_context = dataset.observed.row_context
    for column, values in grammar.condition_columns.items():
        counts = Counter(
            typed_group_key(value)
            for value in null_context[column]
        )
        assert set(counts) == {
            typed_group_key(value)
            for value in values
        }
        source_counts = Counter(
            typed_group_key(value)
            for value in source_context[column]
        )
        assert counts == source_counts
    combinations = Counter(
        (
            typed_group_key(archetype),
            typed_group_key(label),
        )
        for archetype, label in zip(
            null_context["archetype"],
            null_context["label"],
        )
    )
    source_combinations = Counter(
        (
            typed_group_key(archetype),
            typed_group_key(label),
        )
        for archetype, label in zip(
            source_context["archetype"],
            source_context["label"],
        )
    )
    assert combinations == source_combinations


def test_gtib_known_file_shapes_match_the_recovered_rules():
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="gtib")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-8,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    )
    result = SimpleNamespace(
        dataset=dataset,
        portfolio=[evaluator.evaluate(rule) for rule in _rule_set()],
    )
    known = [
        KnownInvariant(
            "ratio",
            "==",
            "completeness_ratio",
            {"ratio": ["output_rate_bytes_per_min", "input_rate_bytes_per_min"]},
        ),
        KnownInvariant(
            "windowed",
            "==",
            "completeness_ratio_1h",
            {
                "ratio": [
                    {"roll_sum": ["output_rate_bytes_per_min", 60]},
                    {"roll_sum": ["input_rate_bytes_per_min", 60]},
                ],
            },
        ),
        KnownInvariant(
            "input_from_raw",
            "==",
            "input_rate_bytes_per_min",
            {"related": "raw_input_rate"},
        ),
        KnownInvariant(
            "static",
            ":=",
            "static_alert",
            {
                "sustained": {
                    "term": "completeness_ratio_1h",
                    "op": "<",
                    "threshold": 0.99,
                    "window": 10,
                },
            },
        ),
        KnownInvariant(
            "trajectory",
            ":=",
            "traj_alert",
            {
                "and": [
                    {"bound": ["completeness_ratio_1h", "<", 0.98]},
                    {
                        "bound": [
                            {
                                "roll_sum": [
                                    {
                                        "difference": [
                                            "input_rate_bytes_per_min",
                                            "output_rate_bytes_per_min",
                                        ],
                                    },
                                    45,
                                ],
                            },
                            ">",
                            0,
                        ],
                    },
                    {
                        "bound": [
                            {"delta": ["completeness_ratio_1h", 45]},
                            "<=",
                            0,
                        ],
                    },
                ],
            },
        ),
        KnownInvariant("oracle", "==", "oracle_alert", "is_true_loss"),
        KnownInvariant(
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
        ),
    ]

    report = recover_known(result, known)

    assert report["recall"] == 1.0, report["invariants"]


def test_gtib_conjunction_and_run_length_match_their_emitted_spans_exactly():
    """Plan line 674: run-length and conjunction rules must match their planted spans exactly.

    "Exactly" is a claim about EVERY row, so this test asserts the graded population is the whole
    frame -- not a convenient subset with the partial-window rows quietly removed. Both alerts are
    total but under different conventions, and both conventions are what the emitted data uses:

    * ``static_alert`` is a SUSTAINED run. Before ten consecutive periods have elapsed the run has
      simply not happened, so those rows are a determinate False.
    * ``traj_alert`` is a CONJUNCTION whose rolling operands need history. An operand that cannot
      be evaluated carries no evidence and must not falsify the conjunction, so the rule reduces to
      the AND over the conjuncts that can be observed.
    """
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="gtib_spans")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=1e-8, hold_rate_threshold=0.9, band_mode="global"),
    )
    n_rows = dataset.observed.n_rows

    sustained = evaluator.evaluate(A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("static_alert"),
            A.Sustained(A.Bound(A.Ref("completeness_ratio_1h"), "<", 0.99), 10),
        ),
    ))
    assert sustained.n_points == n_rows
    assert sustained.hold_rate == 1.0

    trajectory = evaluator.evaluate(A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("traj_alert"),
            A.Conjunction((
                A.Bound(A.Ref("completeness_ratio_1h"), "<", 0.98),
                A.Bound(
                    A.Rolling(
                        A.Add((
                            A.Ref("input_rate_bytes_per_min"),
                            A.Scale(-1.0, A.Ref("output_rate_bytes_per_min")),
                        )),
                        45,
                        "SUM",
                    ),
                    ">",
                    0.0,
                ),
                A.Bound(A.Diff(A.Ref("completeness_ratio_1h"), 45), "<=", 0.0),
            )),
        ),
    ))
    # With every threshold explicit, all three conjuncts are parameter-free and can settle a row,
    # so the rule is graded on 1730 of 2160 rows and matches the emitted spans exactly on all of
    # them. The old all-or-nothing rule graded only 1501, excusing 229 rows that an observed
    # conjunct already decides -- including rows where the target fires.
    assert trajectory.n_points == 1730
    assert trajectory.hold_rate == 1.0


def _phase_capabilities(grammar, *, degree, proportional, max_lag, windows,
                        conditions, advanced, run_lengths, conjunction_terms, related):
    """Pin one phase's capabilities directly on the built grammar.

    `_widen_spec` alone is not enough here, and assuming otherwise produced a ladder that only
    looked like five rungs. `build_dataframe_grammar` derives the grammar from the *frame's*
    adapter, and `prepare_gtib_files` profiles the GTIB frame with every capability already
    declared -- degree 2, temporal with `max_lag=60` and windows `(10, 45, 60)`, advanced logic,
    run lengths, conjunctions and the related-grain roles (`loader/gtib.py`). Widening a bare spec
    therefore changed nothing.

    Each capability has to be switched off the way the engine actually reads it, not the way it is
    spelled in the spec:

    * proportional laws are gated by whether `~?` is in `grammar.ops` (`validate.py`), NOT by any
      `proportional_enabled` flag -- there is no such field on `Grammar`, so setting one silently
      does nothing and leaves the operator enumerable;
    * cross-grain reach is gated by `related_roles` / `boolean_related_roles`, which have to be
      emptied rather than merely left alone.

    The caller asserts both the nesting of the candidate sets and the per-phase absence of these
    shapes, so a future regression here fails loudly instead of quietly flattening the ladder.
    """
    grammar.max_degree = int(degree)
    grammar.max_degree_by_binder = {
        binder: min(int(cap), int(degree))
        for binder, cap in dict(getattr(grammar, "max_degree_by_binder", {}) or {}).items()
    }
    available = tuple(grammar.ops)
    ops = [op for op in available if op != "~\u221d"]
    if proportional and "~\u221d" in available:
        ops.append("~\u221d")
    grammar.ops = tuple(ops)
    grammar.temporal_enabled = bool(max_lag or windows)
    grammar.max_lag = int(max_lag)
    grammar.windows = tuple(sorted(windows))
    grammar.conditional_enabled = bool(conditions) and bool(grammar.condition_columns)
    grammar.advanced_enabled = bool(advanced)
    grammar.run_lengths = tuple(sorted(run_lengths))
    grammar.max_conjunction_terms = int(conjunction_terms)
    if not related:
        grammar.related_roles = {}
        grammar.boolean_related_roles = {}
    return grammar


def _phase_base_spec() -> GrammarSpec:
    """The narrowest spec: no ratios, no temporal operators, no advanced logic."""
    return GrammarSpec(
        name="gtib_phase",
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
        max_degree=1,
    )


def test_gtib_phase_ladder_climbs_held_out_recall_with_zero_null_acceptances():
    """Plan line 676: held-out recall climbs monotonically per phase, and null stays 0.

    The rungs are the plan's own phases -- base bounds and bands, then ratios and proportional laws,
    then temporal operators, then row conditions, then the advanced tier (run lengths, conjunctions,
    related grain). Each rung's capabilities are pinned on the grammar by `_phase_capabilities`,
    because the profiled GTIB frame already declares all of them and widening a bare spec would
    silently produce five copies of the same search.

    Three properties are checked:

    * **Nesting.** Every rung's candidate set is asserted to be a subset of the next, and the union
      of all rungs to equal the widest. This is what makes the ladder a ladder, and it is asserted
      rather than assumed because the previous version of this test violated it without saying so.
    * **Monotone recall.** Recall is measured on ONE fixed held-out split, chosen once before the
      loop, so the rungs are comparable and no phase is scored against a split picked for it. The
      full catalogue is tracked alongside as a second curve, so a ladder that stopped adding reach
      would be caught even if the held-out sample did not notice.
    * **Zero null acceptances.** The runtime-parity null controls are asserted on the widest rung.
      Given the nesting assertion above this settles every rung: `Acc` is a pure per-rule function
      of the data and the null count is just the number of accepted rules of a given shape, so any
      rule an earlier rung could accept on the null is also in the widest rung's candidate set and
      would be accepted there too. Re-running the nulls at all five rungs was measured at roughly
      two and a half extra hours and can only reproduce the same verdict.

    The split seed is pinned at 3 because that split spans the ladder -- a sign bound, a proportional
    law, two temporal laws and a sustained definition -- so the curve can actually move. Seed 0's
    split happens to be reachable in full at phase 0, which would make monotonicity vacuous.

    The complexity ceiling is smaller than the production profile so the ladder stays affordable,
    which bounds how much of the catalogue any rung reaches; `test_checked_in_gtib_known_catalog_
    reaches_full_recall` is what pins full recall at production settings.
    """
    frame = prepare_gtib_files(
        "data/gtib-emulation/timeseries_derived.csv",
        "data/gtib-emulation/timeseries_raw.csv",
    )
    known = load_known("configs/gtib_known.yaml")
    _calibration, validation = _split_known(known, frac=0.3, seed=3)
    assert validation, "the held-out split must be non-empty for this test to mean anything"

    windows = (10, 45, 60)
    base = dict(
        degree=1,
        proportional=False,
        max_lag=0,
        windows=(),
        conditions=False,
        advanced=False,
        run_lengths=(),
        conjunction_terms=0,
        related=False,
    )
    ratios = {**base, "degree": 2, "proportional": True}
    temporal = {**ratios, "max_lag": 60, "windows": windows}
    conditioned = {**temporal, "conditions": True}
    advanced = {
        **conditioned,
        "advanced": True,
        "run_lengths": (10,),
        "conjunction_terms": 3,
        "related": True,
    }
    phases = [base, ratios, temporal, conditioned, advanced]

    discovery = DiscoveryConfig(
        tolerance=0.05,
        hold_rate_threshold=0.62,
        band_mode="global",
        seed=0,
    )
    search = SearchConfig(
        max_complexity=10,
        max_add_arity=2,
        max_rules=400_000,
        max_nonlinear_leaves=64,
        max_linear_leaves=32,
        max_conditioned_rules=0,
        max_lag=60,
        windows=windows,
        seed=0,
    )

    held_out = []
    catalogue = []
    candidate_sets = []
    shapes = []
    widest = None
    for phase, capabilities in enumerate(phases):
        dataset, grammar = build_dataframe_grammar(
            frame,
            _phase_base_spec(),
            search_cfg=search,
            name=f"gtib_phase_{phase}",
        )
        _phase_capabilities(grammar, **capabilities)
        proposer = EnumerationProposer(grammar)
        rules = list(proposer.propose())
        candidate_sets.append({rule.signature() for rule in rules})
        shapes.append({
            "proportional": sum(
                1 for rule in rules
                if isinstance(rule.atom, A.Compare) and rule.atom.op == "~\u221d"
            ),
            "related": sum(1 for rule in rules if "RELATED" in rule.unparse()),
            "temporal": sum(
                1 for rule in rules
                if any(token in rule.unparse() for token in ("LAG_", "ROLL_", "DELTA_"))
            ),
            "conditioned": sum(1 for rule in rules if rule.condition is not None),
        })
        result = run_prepared(
            dataset,
            grammar,
            discovery_cfg=discovery,
            search_cfg=search,
            proposer=proposer,
        )
        held_out.append(recover_known(result, validation)["recall"])
        catalogue.append(recover_known(result, known)["recall"])
        widest = (dataset, grammar, search, proposer)

    # The ladder must actually nest, or the null argument below is unfounded.
    for phase in range(len(candidate_sets) - 1):
        escaped = candidate_sets[phase] - candidate_sets[phase + 1]
        assert not escaped, (phase, len(escaped))
    assert candidate_sets[0] < candidate_sets[-1], "the ladder must add reach, not just repeat"

    # Each capability must be genuinely absent before its phase introduces it. Asserting the counts
    # rather than trusting the helper is the point: an earlier version set a `proportional_enabled`
    # attribute that `Grammar` does not have and never cleared the related-grain roles, so phase 0
    # silently enumerated 30 proportional and 1,969 cross-grain candidates.
    assert shapes[0]["proportional"] == 0 and shapes[1]["proportional"] > 0, shapes
    assert shapes[1]["temporal"] == 0 and shapes[2]["temporal"] > 0, shapes
    assert shapes[2]["conditioned"] == 0 and shapes[3]["conditioned"] > 0, shapes
    assert shapes[3]["related"] == 0 and shapes[4]["related"] > 0, shapes

    # Each phase strictly grows the search space and `Acc` is a pure function of the data, so a rule
    # accepted at one rung is still accepted at the next: recall can only climb.
    assert held_out == sorted(held_out), held_out
    assert held_out[0] < held_out[-1], held_out
    assert catalogue == sorted(catalogue), catalogue
    assert catalogue[0] < catalogue[-1], catalogue

    dataset, grammar, search, proposer = widest
    nulls = prepare_runtime_null_controls(
        dataset,
        grammar,
        search,
        seed=1_004,
        proposer=proposer,
    )
    assert null_equalities_at(nulls.null, discovery, 0) == 0
    assert null_temporal_at(nulls.temporal_null, discovery, 0) == 0
    assert null_definitions_at(nulls.definition_null, discovery, 0) == 0
