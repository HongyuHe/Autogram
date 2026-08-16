"""Categorical and Boolean conditions on invariant rules."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogram.config import DiscoveryConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery import synth
from autogram.discovery.known import KnownInvariant, _signature, recover_known, shapes_for_invariant
from autogram.discovery.loop import build_dataframe_grammar
from autogram.discovery.propose import EnumerationProposer, normalize_rule
from autogram.discovery.validate import score_recovery
from autogram.dsl import ast as A
from autogram.dsl.evaluate import (
    _condition_mask,
    typed_group_key,
    typed_signature_value,
)
from autogram.dsl.grammar import Grammar
from autogram.dsl.parser import rule_from_dict, rule_to_dict
from autogram.dsl.typecheck import is_admissible
from autogram.loader.gtib import profile_dataframe
from autogram.logic.solver import is_trivial
from autogram.logic.solver import equivalent
from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="conditional",
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


def _data() -> pd.DataFrame:
    n = 240
    label = np.resize(
        np.array(
            ["normal", "true_loss", "benign_burst", "artifact"],
            dtype=object,
        ),
        n,
    )
    increment = np.select(
        [label == "true_loss", np.isin(label, ["benign_burst", "artifact"])],
        [2.0, 0.0],
        default=-2.0,
    )
    loss = np.cumsum(increment)
    return profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "consumer_id": "consumer",
            "label": label,
            "reset_flag": label == "artifact",
            "loss": loss,
        }),
        time_index="timestamp",
        group_keys=("consumer_id",),
        condition_columns=("label", "reset_flag"),
        temporal_windows=(3,),
        max_lag=3,
    )


def test_condition_domains_and_masks_preserve_typed_identity():
    """Condition profiling and evaluation must distinguish ``True``, ``1``, and ``"1"``."""
    values = np.empty(120, dtype=object)
    values[0::3] = True
    values[1::3] = 1
    values[2::3] = "1"
    frame = profile_dataframe(
        pd.DataFrame({
            "category": pd.Series(values, dtype=object),
            "x": np.arange(values.size, dtype=float),
        }),
        condition_columns=("category",),
    )
    dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="typed_conditions",
    )

    domain = grammar.condition_columns["category"]
    assert len(domain) == 3
    assert len({typed_group_key(value) for value in domain}) == 3

    masks = [
        _condition_mask(
            A.Condition("category", "==", (value,)),
            dataset.observed,
        )
        for value in (True, 1, "1")
    ]
    assert all(mask is not None for mask in masks)
    assert [int(np.count_nonzero(mask)) for mask in masks] == [40, 40, 40]
    assert not np.any(masks[0] & masks[1])
    assert not np.any(masks[0] & masks[2])
    assert not np.any(masks[1] & masks[2])

    in_mask = _condition_mask(
        A.Condition("category", "in", (True, "1")),
        dataset.observed,
    )
    assert in_mask is not None
    assert int(np.count_nonzero(in_mask)) == 80
    assert not np.any(in_mask & masks[1])


def test_nullable_string_condition_domain_drops_all_missing_scalars():
    frame = profile_dataframe(
        pd.DataFrame({
            "kind": pd.Series(["a", pd.NA, "b", None], dtype="string"),
            "x": np.arange(4, dtype=float),
        }),
        condition_columns=("kind",),
    )
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="nullable_string_condition",
    )

    assert grammar.condition_columns["kind"] == ("a", "b")
    # Most importantly, proposal does not try to hash/sort pd.NA.
    list(EnumerationProposer(grammar).propose())


def test_numpy_condition_scalars_canonicalize_before_dedup_and_render():
    frame = profile_dataframe(
        pd.DataFrame({
            "kind": pd.Series(
                [np.float64(1.0), 1.0, np.float64(2.0), 2.0],
                dtype=object,
            ),
            "x": np.arange(4, dtype=float),
        }),
        condition_columns=("kind",),
    )
    _dataset, grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="numpy_scalar_conditions",
    )

    assert grammar.condition_columns["kind"] == (1.0, 2.0)
    assert all(type(value) is float for value in grammar.condition_columns["kind"])
    assert (
        A.Condition("kind", "==", (np.bool_(True),)).unparse()
        == A.Condition("kind", "==", (True,)).unparse()
    )
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("x"), "~=", A.Ref("x")),
        condition=A.Condition("kind", "==", (np.bool_(True),)),
    )
    payload = rule_to_dict(rule)
    json.dumps(payload)
    restored = rule_from_dict(payload)
    assert restored.condition is not None
    assert restored.condition.values == (True,)
    assert type(restored.condition.values[0]) is bool


def test_large_integer_condition_values_sort_and_propose_without_float_conversion():
    huge = 10 ** 400
    grammar = Grammar(
        binders=("record",),
        ops=("~=",),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        condition_columns={"kind": (0, huge, huge + 1)},
        max_condition_values=3,
        conditional_enabled=True,
    )

    proposed = list(EnumerationProposer(grammar).propose())

    assert proposed
    assert any(
        rule.condition is not None
        and huge in rule.condition.values
        for rule in proposed
    )


def test_all_numeric_ast_fields_serialize_numpy_scalars_to_json():
    rules = [
        A.Rule(
            "record",
            A.Compare(
                A.Scale(
                    np.float64(2.0),
                    A.Lag(A.Ref("x"), np.int64(2)),
                ),
                "~=",
                A.Const(np.int64(4)),
            ),
        ),
        A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("target"),
                A.Sustained(
                    A.Bound(
                        A.Ref("x"),
                        ">",
                        np.float64(1.5),
                    ),
                    np.int64(3),
                ),
            ),
        ),
        A.Rule(
            "record",
            A.BandDefinition(
                A.Ref("x"),
                np.float64(2.5),
            ),
        ),
    ]

    for rule in rules:
        payload = rule_to_dict(rule)
        json.dumps(payload)
        rule_from_dict(payload)


def test_parser_rejects_string_numbers_and_float_integer_fields():
    with pytest.raises(ValueError, match="finite number"):
        rule_from_dict({
            "binder": "record",
            "op": "~=",
            "left": {"k": "Ref", "role": "x"},
            "right": {"k": "Const", "value": "1.5"},
        })
    with pytest.raises(ValueError, match="positive integer"):
        rule_from_dict({
            "binder": "record",
            "op": "~=",
            "left": {
                "k": "Lag",
                "term": {"k": "Ref", "role": "x"},
                "steps": 2.0,
            },
            "right": {"k": "Ref", "role": "y"},
        })


def test_solver_and_membership_enumeration_preserve_typed_conditions():
    grammar = Grammar(
        binders=("record",),
        ops=("~=",),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        condition_columns={"kind": (True, 1, "1")},
        max_condition_values=3,
        conditional_enabled=True,
    )
    true_rule = A.Rule(
        "record",
        A.Compare(A.Ref("x"), "~=", A.Ref("y")),
        condition=A.Condition("kind", "==", (True,)),
    )
    one_rule = A.Rule(
        "record",
        A.Compare(A.Ref("x"), "~=", A.Ref("y")),
        condition=A.Condition("kind", "==", (1,)),
    )
    membership = A.Rule(
        "record",
        A.Compare(A.Ref("x"), "~=", A.Ref("y")),
        condition=A.Condition("kind", "in", (True, 1)),
    )

    assert not equivalent(true_rule, one_rule)
    assert is_admissible(membership, grammar)[0]
    proposed = {
        normalize_rule(rule).signature()
        for rule in EnumerationProposer(grammar).propose()
    }
    assert normalize_rule(membership).signature() in proposed


def test_condition_round_trip_typecheck_and_solver_antecedent():
    dataset, grammar = build_dataframe_grammar(_data(), _base_spec(), name="conditions")
    rule = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("loss"), 1), ">=", A.Const(0)),
        condition=A.Condition("label", "==", ("true_loss",)),
    )

    assert rule_from_dict(rule_to_dict(rule)) == rule
    assert is_admissible(rule, grammar)[0] is True
    assert not is_trivial(rule)
    assert 'where label == "true_loss"' in rule.unparse()
    assert dataset.row_context["label"].shape == (240,)


def test_conditioned_temporal_rules_accept_only_on_the_declared_subset():
    dataset, _grammar = build_dataframe_grammar(_data(), _base_spec(), name="conditions")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-12,
            hold_rate_threshold=0.9,
            band_mode="global",
            min_condition_points=20,
        ),
    )
    positive = A.Compare(A.Diff(A.Ref("loss"), 1), ">=", A.Const(0))
    zero = A.Compare(A.Diff(A.Ref("loss"), 1), "~=", A.Const(0))
    unconditioned = A.Rule("record", positive)
    true_loss = A.Rule(
        "record",
        positive,
        condition=A.Condition("label", "==", ("true_loss",)),
    )
    benign = A.Rule(
        "record",
        zero,
        condition=A.Condition("label", "in", ("benign_burst", "artifact")),
    )

    assert not evaluator.evaluate(unconditioned).accepted
    true_result = evaluator.evaluate(true_loss)
    benign_result = evaluator.evaluate(benign)
    assert true_result.accepted and benign_result.accepted
    assert 0.20 < true_result.support < 0.30
    assert 0.45 < benign_result.support < 0.55


def test_nullable_condition_values_are_boolean_non_matches():
    frame = _data()
    frame.loc[[1, 2, 3], "label"] = [pd.NA, None, np.nan]
    frame = profile_dataframe(
        frame,
        time_index="timestamp",
        group_keys=("consumer_id",),
        condition_columns=("label",),
        temporal_windows=(3,),
        max_lag=3,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="nullable")
    values = np.asarray(dataset.row_context["label"], dtype=object)
    equals_true = np.array([
        False if pd.isna(value) else value == "true_loss"
        for value in values
    ], dtype=bool)
    in_losses = np.array([
        False if pd.isna(value) else value in {"true_loss", "artifact"}
        for value in values
    ], dtype=bool)

    cases = (
        (A.Condition("label", "==", ("true_loss",)), equals_true),
        (A.Condition("label", "!=", ("true_loss",)), ~equals_true & ~pd.isna(values)),
        (
            A.Condition("label", "in", ("true_loss", "artifact")),
            in_losses,
        ),
        (
            A.Condition("label", "not in", ("true_loss", "artifact")),
            ~in_losses & ~pd.isna(values),
        ),
    )
    for condition, expected in cases:
        mask = _condition_mask(condition, dataset.observed)
        assert mask.dtype == np.bool_
        assert np.array_equal(mask, expected)
        assert not mask[[1, 2, 3]].any()


def test_condition_with_too_little_support_is_rejected():
    frame = _data()
    frame.loc[:, "rare"] = False
    frame.loc[:2, "rare"] = True
    profile = frame.attrs.copy()
    frame = profile_dataframe(
        frame,
        time_index="timestamp",
        group_keys=("consumer_id",),
        condition_columns=("label", "reset_flag", "rare"),
        temporal_windows=(3,),
        max_lag=3,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="rare")
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(min_condition_points=10),
    ).evaluate(A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("loss"), 1), ">=", A.Const(0)),
        condition=A.Condition("rare", "==", (True,)),
    ))

    assert not result.accepted
    assert "condition support" in result.reason


def test_strict_positive_condition_does_not_count_zero_deltas():
    frame = _data()
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="strict")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=1e-12,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    )
    true_loss = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("loss"), 1), ">", A.Const(0)),
        condition=A.Condition("label", "==", ("true_loss",)),
    )
    benign = A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("loss"), 1), ">", A.Const(0)),
        condition=A.Condition("label", "in", ("benign_burst", "artifact")),
    )

    assert evaluator.evaluate(true_loss).accepted
    assert not evaluator.evaluate(benign).accepted


def test_proposer_enumerates_generic_conditioned_near_equality():
    # Round-21: a generic conditioned relation ``x ~= y where label == L`` between two distinct
    # measurements must be ENUMERATED, not only proportional/delta-zero shapes. This proves
    # conditioning is generic over relation shapes rather than restricted to the GTIB invariant types.
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"record": ("a", "b")},
        fam_roles={"record": ()},
        max_complexity=10,
        conditional_enabled=True,
        condition_columns={"label": ("normal", "alert")},
        max_rules=0,
    )
    proposed = {rule.unparse() for rule in EnumerationProposer(grammar).propose()}
    target = normalize_rule(A.Rule(
        "record",
        A.Compare(A.Ref("a"), "~=", A.Ref("b")),
        condition=A.Condition("label", "==", ("normal",)),
    ))
    assert target.unparse() in proposed


def test_proposer_enumerates_bounded_conditions_for_temporal_rules():
    _dataset, grammar = build_dataframe_grammar(_data(), _base_spec(), name="conditions")
    target = normalize_rule(A.Rule(
        "record",
        A.Compare(A.Diff(A.Ref("loss"), 1), ">=", A.Const(0)),
        condition=A.Condition("label", "==", ("true_loss",)),
    ))

    rendered = {rule.unparse() for rule in EnumerationProposer(grammar).propose()}

    assert target.unparse() in rendered


def test_proposer_omits_full_domain_membership_conditions():
    _dataset, grammar = build_dataframe_grammar(_data(), _base_spec(), name="conditions")

    conditions = [
        rule.condition
        for rule in EnumerationProposer(grammar).propose()
        if rule.condition is not None and rule.condition.op == "in"
    ]

    assert conditions
    assert all(
        set(condition.values)
        != set(grammar.condition_columns[condition.column])
        for condition in conditions
    )


def test_typechecker_rejects_condition_operators_the_proposer_does_not_admit():
    _dataset, grammar = build_dataframe_grammar(
        _data(),
        _base_spec(),
        name="condition_operator_boundary",
    )
    atom = A.Compare(
        A.Diff(A.Ref("loss"), 1),
        ">=",
        A.Const(0),
    )
    for condition in (
        A.Condition("label", "!=", ("normal",)),
        A.Condition(
            "label",
            "not in",
            ("normal", "artifact"),
        ),
    ):
        admissible, _reason = is_admissible(
            A.Rule("record", atom, condition=condition),
            grammar,
        )
        assert not admissible


def test_proposer_enumerates_every_admitted_membership_subset():
    _dataset, grammar = build_dataframe_grammar(
        _data(),
        _base_spec(),
        name="condition_membership_subsets",
    )
    expected = A.Condition(
        "label",
        "in",
        ("artifact", "normal", "true_loss"),
    )

    assert expected in EnumerationProposer(grammar)._conditions()


def test_conditioned_temporal_enumeration_is_rename_invariant():
    def conditioned_count(last_role):
        roles = tuple(f"r{index}" for index in range(8)) + (last_role,)
        grammar = Grammar(
            binders=("record",),
            ops=("~=", "==", "<=", ">="),
            ref_roles={"record": roles},
            fam_roles={"record": ()},
            max_complexity=10,
            max_add_arity=2,
            temporal_enabled=True,
            max_lag=1,
            windows=(),
            conditional_enabled=True,
            condition_columns={"label": ("normal", "alert")},
            max_rules=0,
        )
        return sum(
            rule.condition is not None
            for rule in EnumerationProposer(grammar).propose()
        )

    neutral = conditioned_count("r8")
    lexical = conditioned_count("loss_counter")
    assert neutral > 0
    assert neutral == lexical


def test_conditioned_rule_ceiling_fails_instead_of_selecting_a_prefix():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"record": ("x",)},
        fam_roles={"record": ()},
        temporal_enabled=True,
        max_lag=1,
        conditional_enabled=True,
        condition_columns={
            "first": ("off", "on"),
            "second": ("off", "on"),
        },
        max_conditioned_rules=1,
    )

    with pytest.raises(RuntimeError, match="max_conditioned_rules=1"):
        EnumerationProposer(grammar).propose()


def test_exhaustive_condition_search_uses_every_declared_column():
    condition_columns = {
        f"c{index}": ("off", "on")
        for index in range(7)
    }
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"record": ("x",)},
        fam_roles={"record": ()},
        max_complexity=10,
        max_add_arity=2,
        temporal_enabled=True,
        max_lag=1,
        windows=(),
        conditional_enabled=True,
        condition_columns=condition_columns,
        max_rules=0,
        max_conditioned_rules=0,
    )
    used = {
        rule.condition.column
        for rule in EnumerationProposer(grammar).propose()
        if rule.condition is not None
        and rule.condition.op != "all"
    }

    assert used == set(condition_columns)


def test_condition_conjunction_search_is_column_name_invariant():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"record": ("x",)},
        fam_roles={"record": ()},
        conditional_enabled=True,
        condition_columns={
            "segment": ("steady", "bursty"),
            "state": ("normal", "alert"),
        },
    )
    expected = A.Condition(
        "",
        "all",
        (
            A.Condition("segment", "==", ("steady",)),
            A.Condition("state", "==", ("normal",)),
        ),
    )

    assert expected in EnumerationProposer(grammar)._conditions()


def test_proposer_rejects_self_conditioned_temporal_rules():
    grammar = Grammar(
        binders=("record",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"record": ("flag",)},
        fam_roles={"record": ()},
        temporal_enabled=True,
        max_lag=1,
        conditional_enabled=True,
        condition_columns={"flag": (False, True)},
    )

    rules = EnumerationProposer(grammar).propose()

    assert not any(
        rule.condition is not None
        and rule.condition.column == "flag"
        and isinstance(rule.atom, A.Compare)
        and (
            isinstance(rule.atom.left, A.Diff)
            or isinstance(rule.atom.right, A.Diff)
        )
        for rule in rules
    )


def test_conditional_known_signatures_and_proxy_recovery():
    known = [
        KnownInvariant(
            "positive",
            ">=",
            {"delta": "loss"},
            0,
            where={"label": "true_loss"},
        ),
        KnownInvariant(
            "zero",
            "~=",
            {"delta": "loss"},
            0,
            where={"label_in": ["benign_burst", "artifact"]},
        ),
    ]
    assert _signature(known[0])[0] == "conditional"
    assert _signature(known[1])[0] == "conditional"
    assert shapes_for_invariant(known[0]) == ["conditional_positive"]
    assert shapes_for_invariant(known[1]) == ["conditional_zero"]

    result = SimpleNamespace(portfolio=[])
    planted = {
        "conditional_positive": {
            (("label", "==", ("true_loss",)), ("delta_bound", ("loss", 1, ">="))),
        },
        "conditional_zero": {
            (
                ("label", "in", ("artifact", "benign_burst")),
                ("delta_zero", ("loss", 1)),
            ),
        },
    }
    from autogram.discovery import validate as validation
    original_relations = validation.portfolio_relations
    try:
        validation.portfolio_relations = lambda _result: {
            (
                "conditional",
                (("label", "==", ("true_loss",)), ("delta_bound", ("loss", 1, ">="))),
            ),
            (
                "conditional",
                (
                    ("label", "in", ("artifact", "benign_burst")),
                    ("delta_zero", ("loss", 1)),
                ),
            ),
        }
        recovery = score_recovery(result, planted)
    finally:
        validation.portfolio_relations = original_relations

    assert recovery.conditional_positive == 1.0
    assert recovery.conditional_zero == 1.0


def test_conditioned_bound_does_not_recover_unconditional_one_sided_law():
    dataset, _grammar = build_dataframe_grammar(
        _data(),
        _base_spec(),
        name="conditional_one_sided",
    )
    result = SimpleNamespace(
        dataset=dataset,
        portfolio=[
            SimpleNamespace(
                rule=A.Rule(
                    "record",
                    A.Compare(A.Ref("loss"), ">=", A.Const(0)),
                    condition=A.Condition("label", "==", ("true_loss",)),
                ),
            )
        ],
    )
    known = [KnownInvariant("nonnegative_loss", ">=", "loss", 0)]

    assert recover_known(result, known)["recall"] == 0.0
    assert score_recovery(result, {"nonneg": {"loss"}}).nonneg == 0.0


def test_unconditional_representative_does_not_recover_conditioned_target():
    # A conditioned known invariant requires conditioned evidence: an unconditional
    # portfolio rule with the same atom is a broader (and possibly weaker) claim, so it
    # must not be counted as recovering the specifically-conditioned law. The archive is
    # responsible for keeping genuinely-conditioned refinements (see the archive tests),
    # so recovery never has to fall back to an unconditional shadow.
    dataset, _grammar = build_dataframe_grammar(
        _data(),
        _base_spec(),
        name="unconditional_representative",
    )
    evaluation = SimpleNamespace(
        rule=A.Rule(
            "record",
            A.Compare(
                A.Diff(A.Ref("loss"), 1),
                ">=",
                A.Const(0),
            ),
        ),
        parameters={},
    )
    result = SimpleNamespace(
        dataset=dataset,
        portfolio=[evaluation],
    )
    known = [
        KnownInvariant(
            "conditioned_positive",
            ">=",
            {"delta": "loss"},
            0,
            where={"label": "true_loss"},
        ),
    ]
    planted = {
        "conditional_positive": {
            (
                ("label", "==", ("true_loss",)),
                ("delta_bound", ("loss", 1, ">=")),
            ),
        },
    }

    assert recover_known(result, known)["recall"] == 0.0
    assert score_recovery(
        result,
        planted,
    ).conditional_positive == 0.0


def test_conditional_known_membership_preserves_scalar_types():
    # An integer membership condition (`code_in: [1, 2]`) must keep its scalar type so it can match
    # a learned rule whose condition values retain the observed integer dtype; stringifying the
    # known would make a valid recovery impossible.
    from autogram.discovery.known import _known_condition_signature

    assert _known_condition_signature({"code_in": [1, 2]}) == (
        "code",
        "in",
        (typed_signature_value(1), typed_signature_value(2)),
    )

    known = [
        KnownInvariant(
            "guarded",
            ">=",
            {"delta": "loss"},
            0,
            where={"code_in": [1, 2]},
        ),
    ]
    assert _signature(known[0]) == (
        "conditional",
        (
            (
                "code",
                "in",
                (typed_signature_value(1), typed_signature_value(2)),
            ),
            ("delta_bound", ("loss", 1, ">=")),
        ),
    )

    # The typed known signature matches a learned signature whose membership values retain the
    # observed integer dtype; a stringified known ("1","2") would fail this exact-match check.
    from autogram.discovery.validate import relation_signature_matches

    learned = (
        "conditional",
        (
            (
                "code",
                "in",
                (typed_signature_value(1), typed_signature_value(2)),
            ),
            ("delta_bound", ("loss", 1, ">=")),
        ),
    )
    assert relation_signature_matches(_signature(known[0]), learned)
    assert not relation_signature_matches(
        ("conditional", (("code", "in", ("1", "2")), ("delta_bound", ("loss", 1, ">=")))),
        learned,
    )


def test_conditional_proxy_generators_plant_only_guarded_relations():
    for shape, base_tag in (
        ("conditional_positive", "delta_bound"),
        ("conditional_zero", "delta_zero"),
    ):
        data = synth.make_synthetic(
            n_entities=3,
            n_snapshots=120,
            noise=0.02,
            seed=9,
            families=(shape,),
        )
        assert set(data.planted) == {shape}
        assert set(data.row_context["regime"]) == {"positive", "zero", "other"}
        assert data.planted[shape]
        assert all(payload[1][0] == base_tag for payload in data.planted[shape])


def test_conditioned_comparison_support_is_measured_on_gradeable_rows():
    # Round-24: the floor must be read off the rows the rule can actually GRADE. Measuring only
    # the rows the CONDITION selects overstates the evidence whenever operands are non-finite, so
    # a rule scored on a handful of points can clear a percentage floor it does not really meet.
    n = 2000
    selected = np.zeros(n, dtype=bool)
    selected[:1000] = True
    left = np.full(n, np.nan)
    right = np.full(n, np.nan)
    # Only 20 of the 1000 condition rows carry finite operands, so the real evidence is 1% of rows.
    left[:20] = np.arange(20, dtype=float)
    right[:20] = np.arange(20, dtype=float)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "series_id": "a",
        "regime": np.where(selected, "rare", "common"),
        "left_bytes": left,
        "right_bytes": right,
    })
    frame = profile_dataframe(
        df,
        time_index="timestamp",
        group_keys=("series_id",),
        condition_columns=("regime",),
        temporal_windows=(2,),
        max_lag=2,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="graded_condition")
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("left_bytes"), "==", A.Ref("right_bytes")),
        condition=A.Condition("regime", "==", ("rare",)),
    )

    evaluation = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            min_condition_points=10,
            min_condition_fraction=0.10,
            band_mode="global",
        ),
    ).evaluate(rule)

    # 20 gradeable rows out of 2000 is 1%, well under the configured 10% floor, even though the
    # condition itself selects half the frame.
    assert not evaluation.accepted
    assert "condition support below minimum" in evaluation.reason
    assert "0.010 of rows" in evaluation.reason

    permissive = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            min_condition_points=10,
            min_condition_fraction=0.001,
            band_mode="global",
        ),
    ).evaluate(rule)
    assert "condition support below minimum" not in permissive.reason
