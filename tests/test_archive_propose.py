"""Archive + exhaustive enumeration proposer."""

from __future__ import annotations

import pytest

from autogram.config import DiscoveryConfig
from autogram.discovery.archive import ParetoArchive
from autogram.discovery.evaluate import DataOnlyEvaluator, Evaluation
from autogram.discovery.propose import (
    EnumerationProposer,
    SearchSpaceTruncatedError,
    normalize_rule,
)
from autogram.dsl import ast as A
from autogram.dsl.grammar import Grammar
from autogram.dsl.typecheck import is_admissible


def test_archive_keeps_solver_distinct_representatives(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    arch = ParetoArchive()
    r1 = ev.evaluate(A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev"))))
    r2 = ev.evaluate(A.Rule("node", A.Compare(A.Agg("SUM", "demand_row"), "~=", A.Ref("measurement_source"))))
    assert arch.add(r1) and arch.add(r2)
    assert len(arch.portfolio()) == 2
    assert arch.progress() > 0


def test_front_is_non_dominated(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    arch = ParetoArchive()
    for rule in (A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev"))),
                 A.Rule("node", A.Compare(A.Agg("SUM", "demand_row"), "~=", A.Ref("measurement_source")))):
        arch.add(ev.evaluate(rule))
    front = arch.front()
    assert front and all(f.accepted for f in front)


def test_archive_retains_lag_shadow_unless_exact_atomic_suppresses_it(dataset):
    # A lag one-sided sign bound is RETAINED as an independent temporal law unless an EXACT atomic
    # (hold-rate 1.0) proves it redundant: an exact ``x >= 0`` evicts ``LAG_k(x) >= 0`` (which keeps
    # the temporal-null control at zero), while a lag whose atomic is not exact survives so a genuine
    # lag law is never dropped. A compound one-sided target stays bloated/dropped.
    from autogram.discovery.archive import _is_bloated_one_sided

    def ev(rule, hold=1.0, raw_exact=None):
        # ``raw_exact`` defaults to "exact iff hold_rate is 1.0" for convenience, but a caller can
        # force a tolerance-absorbed atomic (hold 1.0, raw-inexact) to prove suppression is gated on
        # tolerance-free exactness, not hold_rate.
        exact = (hold >= 1.0) if raw_exact is None else raw_exact
        return Evaluation(
            rule=rule, accepted=True, reason="test", eps=0.0, hold_rate=hold,
            hold_rate_lo=hold, hold_rate_hi=hold, statistic="test", support=1.0,
            n_points=90, n_bindings=1, mdl_gain=0.0, strictness="one-sided", descriptor=(),
            raw_exact_sign=bool(exact),
        )

    lag = A.Rule("record", A.Compare(A.Lag(A.Ref("x"), 5), ">=", A.Const(0)))
    diff = A.Rule("record", A.Compare(A.Diff(A.Ref("x"), 3), ">=", A.Const(0)))
    compound = A.Rule("record", A.Compare(A.Add((A.Ref("a"), A.Ref("b"))), ">=", A.Const(0)))
    atomic_exact = A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))

    # A lag / diff sign bound is not bloated; a compound target is.
    assert _is_bloated_one_sided(ev(lag)) is False
    assert _is_bloated_one_sided(ev(diff)) is False
    assert _is_bloated_one_sided(ev(compound)) is True

    # The lag alone is retained (no atomic to prove it redundant).
    only_lag = ParetoArchive()
    assert only_lag.add(ev(lag)) is True
    assert lag in [e.rule for e in only_lag.portfolio(non_redundant=True)]

    # An EXACT atomic ``x >= 0`` suppresses the lag shadow in the non-redundant portfolio.
    with_exact = ParetoArchive()
    with_exact.add(ev(lag))
    with_exact.add(ev(atomic_exact, hold=1.0))
    kept = [e.rule for e in with_exact.portfolio(non_redundant=True)]
    assert atomic_exact in kept and lag not in kept

    # A NON-exact atomic (hold-rate < 1.0) does not prove the lag redundant, so the lag survives.
    with_inexact = ParetoArchive()
    with_inexact.add(ev(lag))
    with_inexact.add(ev(atomic_exact, hold=0.95))
    kept_inexact = [e.rule for e in with_inexact.portfolio(non_redundant=True)]
    assert lag in kept_inexact

    # A tolerance-absorbed atomic (hold-rate 1.0 but NOT raw-exact) must NOT suppress the lag: the
    # acceptance tolerance can hide a real violation against a large scale, so hold_rate 1.0 is not
    # exactness. This is the round-20 soundness case.
    with_absorbed = ParetoArchive()
    with_absorbed.add(ev(lag))
    with_absorbed.add(ev(atomic_exact, hold=1.0, raw_exact=False))
    kept_absorbed = [e.rule for e in with_absorbed.portfolio(non_redundant=True)]
    assert lag in kept_absorbed


def test_archive_exact_zero_equality_removes_redundant_one_sided_bounds(dataset):
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(seed=0, hold_rate_threshold=0.9),
    )
    archive = ParetoArchive()
    ref = A.Ref("demand_self")
    bounds = [
        A.Rule("node", A.Compare(ref, ">=", A.Const(0))),
        A.Rule("node", A.Compare(ref, "<=", A.Const(0))),
    ]
    equality = A.Rule("node", A.Compare(A.Const(0), "==", ref))

    for rule in [*bounds, equality]:
        assert archive.add(evaluator.evaluate(rule))

    kept = [evaluation.rule for evaluation in archive.portfolio()]
    assert equality in kept
    assert not any(rule in kept for rule in bounds)


def test_archive_prefers_strict_atomic_reference_bound():
    # Between a strict and a non-strict one-sided bound in the same direction, the STRICT bound is
    # kept: it entails the non-strict (archive completeness -- the survivor covers the discarded
    # rule), and a strict known can only be recovered from a strict portfolio rule.
    def accepted(rule):
        return Evaluation(
            rule=rule,
            accepted=True,
            reason="test",
            eps=0.0,
            hold_rate=1.0,
            hold_rate_lo=1.0,
            hold_rate_hi=1.0,
            statistic="test",
            support=1.0,
            n_points=1,
            n_bindings=1,
            mdl_gain=0.0,
            strictness="one-sided",
            descriptor=(),
        )

    ref = A.Ref("measurement_source")
    nonstrict = A.Rule("node", A.Compare(ref, ">=", A.Const(0)))
    strict = normalize_rule(
        A.Rule("node", A.Compare(A.Const(0), "<", ref))
    )
    # Non-strict added first, then strict: the strict evicts the non-strict.
    archive = ParetoArchive()
    assert archive.add(accepted(nonstrict))
    assert archive.add(accepted(strict))
    assert [evaluation.rule for evaluation in archive.portfolio()] == [strict]

    # Strict added first, then non-strict: the non-strict is rejected (already covered).
    archive2 = ParetoArchive()
    assert archive2.add(accepted(strict))
    assert not archive2.add(accepted(nonstrict))
    assert [evaluation.rule for evaluation in archive2.portfolio()] == [strict]


def test_unconditional_rule_removes_identical_conditioned_atoms():
    def accepted(rule, hold_rate=1.0, eps=0.0, parameters=None):
        return Evaluation(
            rule=rule,
            accepted=True,
            reason="test",
            eps=eps,
            hold_rate=hold_rate,
            hold_rate_lo=hold_rate,
            hold_rate_hi=hold_rate,
            statistic="test",
            support=1.0,
            n_points=10,
            n_bindings=1,
            mdl_gain=0.0,
            strictness="exact",
            descriptor=(),
            parameters=dict(parameters or {}),
        )

    atom = A.Compare(A.Ref("x"), ">=", A.Const(0))
    conditioned = A.Rule(
        "record",
        atom,
        condition=A.Condition("label", "==", ("normal",)),
    )
    unconditional = A.Rule("record", atom)

    for insertion_order in (
        (conditioned, unconditional),
        (unconditional, conditioned),
    ):
        archive = ParetoArchive()
        for index, rule in enumerate(insertion_order):
            archive.add(accepted(
                rule,
                parameters={
                    "group_hold_rates": {
                        "g": 1.0 - index * 0.01,
                    },
                },
            ))
        assert [
            evaluation.rule
            for evaluation in archive.portfolio()
        ] == [unconditional]


def test_archive_preserves_conditioned_rules_with_distinct_fitted_semantics():
    atom = A.Compare(A.Ref("x"), "~=", A.Ref("y"))
    unconditional = A.Rule("record", atom)
    conditioned = A.Rule(
        "record",
        atom,
        condition=A.Condition("label", "==", ("normal",)),
    )

    def accepted(rule, eps):
        return Evaluation(
            rule=rule,
            accepted=True,
            reason="test",
            eps=eps,
            hold_rate=1.0,
            hold_rate_lo=1.0,
            hold_rate_hi=1.0,
            statistic="test",
            support=1.0,
            n_points=10,
            n_bindings=1,
            mdl_gain=0.0,
            strictness="approximate",
            descriptor=(),
        )

    archive = ParetoArchive()
    archive.add(accepted(unconditional, 0.1))
    archive.add(accepted(conditioned, 0.01))

    assert {
        evaluation.rule
        for evaluation in archive.portfolio()
    } == {
        unconditional,
        conditioned,
    }


def test_weaker_unconditional_rule_keeps_stronger_conditioned_refinement():
    # A conditioned law that holds exactly (hold-rate 1.0) is strictly more informative than
    # a weaker unconditional shadow of the same atom (e.g. hold-rate 0.67): the unconditional
    # rule does not logically hold everywhere, so it cannot evict the exact refinement.
    atom = A.Compare(A.Diff(A.Ref("x"), 1), ">=", A.Const(0))
    unconditional = A.Rule("record", atom)
    conditioned = A.Rule(
        "record",
        atom,
        condition=A.Condition("regime", "==", ("positive",)),
    )

    def accepted(rule, hold_rate):
        return Evaluation(
            rule=rule,
            accepted=True,
            reason="test",
            eps=0.05,
            hold_rate=hold_rate,
            hold_rate_lo=hold_rate,
            hold_rate_hi=hold_rate,
            statistic="test",
            support=1.0,
            n_points=90,
            n_bindings=1,
            mdl_gain=0.0,
            strictness="one-sided",
            descriptor=(),
        )

    for insertion_order in (
        (accepted(unconditional, 0.67), accepted(conditioned, 1.0)),
        (accepted(conditioned, 1.0), accepted(unconditional, 0.67)),
    ):
        archive = ParetoArchive()
        for evaluation in insertion_order:
            archive.add(evaluation)
        # Both the raw archive and the final non-redundant compaction pass (the production path)
        # must retain the stronger conditioned refinement alongside the weaker unconditional shadow.
        for portfolio in (
            archive.portfolio(),
            archive.portfolio(non_redundant=True),
        ):
            assert {
                evaluation.rule
                for evaluation in portfolio
            } == {
                unconditional,
                conditioned,
            }


def test_final_compaction_keeps_conditioned_rule_with_distinct_fitted_coefficient():
    # A proportional law shares a single symbolic Z3 coefficient across the unconditional and
    # conditioned forms, so Z3 subsumption would tautologically drop the conditioned rule even
    # though its *fitted* coefficient differs. The final non-redundant compaction must respect the
    # fitted-semantics guard and keep the genuinely distinct conditioned fit.
    atom = A.Compare(A.Ref("x"), "~\u221d", A.Ref("y"))
    unconditional = A.Rule("record", atom)
    conditioned = A.Rule(
        "record",
        atom,
        condition=A.Condition("regime", "==", ("steady",)),
    )

    def accepted(rule, coefficient):
        return Evaluation(
            rule=rule,
            accepted=True,
            reason="test",
            eps=0.02,
            hold_rate=1.0,
            hold_rate_lo=1.0,
            hold_rate_hi=1.0,
            statistic="test",
            support=1.0,
            n_points=90,
            n_bindings=1,
            mdl_gain=0.0,
            strictness="proportional",
            descriptor=(),
            parameters={"coefficient": coefficient},
        )

    archive = ParetoArchive()
    archive.add(accepted(unconditional, 1.7))
    archive.add(accepted(conditioned, 2.4))

    kept = {
        evaluation.rule
        for evaluation in archive.portfolio(non_redundant=True)
    }
    assert kept == {unconditional, conditioned}


def test_enumeration_yields_admissible_unique_rules(grammar):
    prop = EnumerationProposer(grammar)
    rules = prop.propose()
    assert rules
    for r in rules:
        ok, _ = is_admissible(r, grammar)
        assert ok
    assert len({r.signature() for r in rules}) == len(rules)
    rendered = {r.unparse() for r in rules}
    assert "[forall node] measurement_source >= 0" in rendered
    assert "[forall link] o0 != o0_rev" in rendered


def test_enumeration_includes_bounded_scale_and_add_forms(grammar):
    rules = EnumerationProposer(grammar).propose()
    atoms = [r.atom for r in rules]
    assert any(isinstance(a.left, A.Scale) or isinstance(a.right, A.Scale) for a in atoms)
    assert any(isinstance(a.left, A.Add) or isinstance(a.right, A.Add) for a in atoms)


def test_enumeration_fails_loudly_when_rule_ceiling_would_truncate():
    limited = Grammar(
        binders=("record",),
        ops=("~=", "==", "!=", "<=", ">="),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        max_rules=7,
    )

    with pytest.raises(RuntimeError, match="max_rules=7"):
        EnumerationProposer(limited).propose()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_linear_leaves", 2),
        ("max_nonlinear_leaves", 2),
    ],
)
def test_enumeration_fails_loudly_when_leaf_ceiling_would_truncate(
    field,
    value,
):
    kwargs = {
        "binders": ("record",),
        "ops": ("~=", "==", "<=", ">="),
        "ref_roles": {"record": ("x", "y", "z")},
        "fam_roles": {"record": ()},
        "max_degree": 2,
        field: value,
    }

    with pytest.raises(
        SearchSpaceTruncatedError,
        match=rf"{field}={value}",
    ):
        EnumerationProposer(Grammar(**kwargs)).propose()


def test_condition_value_cap_bounds_subset_size_not_domain_coverage():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "=="),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        conditional_enabled=True,
        condition_columns={
            "label": ("alpha", "beta", "gamma"),
        },
        max_condition_values=2,
    )

    conditions = EnumerationProposer(grammar)._conditions()

    assert {
        condition.values
        for condition in conditions
        if condition.column == "label"
        and condition.op == "=="
    } == {
        ("alpha",),
        ("beta",),
        ("gamma",),
    }
    assert {
        condition.values
        for condition in conditions
        if condition.column == "label"
        and condition.op == "in"
    } == {
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
    }


def test_monotone_completeness_requires_the_auxiliary_acceptance_gates():
    """The (C4) hypothesis of docs/autogram_guarantees.md is load-bearing, not decoration.

    A candidate can dominate a recovered known invariant on every statistical axis the theorem
    compares -- a strictly higher hold rate at identical `n` and identical threshold -- and still
    be rejected, because the aggregate Wilson bound says nothing about the PER-GROUP gate. If the
    published theorem omitted the auxiliary conjuncts it would therefore be false, so this test
    pins the counterexample that forces them into the hypotheses.
    """
    import numpy as np
    import pandas as pd

    from autogram.discovery.loop import build_dataframe_grammar
    from autogram.loader.gtib import profile_dataframe
    from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology

    spec = GrammarSpec(
        name="gates",
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

    per_group = 500
    groups = ("g0", "g1")
    n = per_group * len(groups)
    rng = np.random.default_rng(0)

    # `spread` breaks uniformly at a 25% rate in BOTH groups, so each group holds 75%.
    spread_break = np.zeros(n, dtype=bool)
    for index in range(len(groups)):
        offset = index * per_group
        picks = rng.choice(per_group, size=per_group // 4, replace=False)
        spread_break[offset + picks] = True

    # `focused` breaks LESS often overall (22.5%), but every break lands in one group, so that
    # group alone holds only 55% of the time.
    focused_break = np.zeros(n, dtype=bool)
    picks = rng.choice(per_group, size=int(per_group * 0.45), replace=False)
    focused_break[per_group + picks] = True

    base = np.ones(n)
    df = pd.DataFrame({
        "timestamp": np.tile(
            pd.date_range("2026-01-01", periods=per_group, freq="1min"),
            len(groups),
        ),
        "series_id": np.repeat(np.array(groups), per_group),
        "spread_left_bytes": base,
        "spread_right_bytes": np.where(spread_break, 5.0, 1.0),
        "focused_left_bytes": base,
        "focused_right_bytes": np.where(focused_break, 5.0, 1.0),
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=(),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, spec, name="auxiliary_gates")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(band_mode="global", tolerance=0.05, hold_rate_threshold=0.6, seed=0),
    )

    known = evaluator.evaluate(
        A.Rule("record", A.Compare(A.Ref("spread_left_bytes"), "~=", A.Ref("spread_right_bytes")))
    )
    dominating = evaluator.evaluate(
        A.Rule("record", A.Compare(A.Ref("focused_left_bytes"), "~=", A.Ref("focused_right_bytes")))
    )

    # The recovered known invariant is accepted.
    assert known.accepted
    # (C1) at least as clean, (C2) equal evidence, (C3) equal bar -- the candidate dominates.
    assert dominating.hold_rate > known.hold_rate
    assert dominating.n_points == known.n_points
    assert dominating.threshold == pytest.approx(known.threshold)
    assert dominating.hold_rate_lo >= known.hold_rate_lo
    # ...and yet it is rejected, because one group fails the per-group gate on its own.
    assert not dominating.accepted
