"""Fast, offline tests for scoreboard canonicalization of near-zero sum groupings."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogram.dsl import ast as A
from autogram.dsl.evaluate import typed_group_key, typed_signature_value
from autogram.dsl.scalar_codec import scalar_to_json
from autogram.loader.loader import Frame
from autogram.discovery.known import (
    KnownInvariant,
    _canonical_known_condition,
    _canonicalize,
    _drop_negligible,
    _known_condition_signature,
    _signature,
    load_known,
    recover_known,
    validate_known_conditions,
)
from autogram.discovery import known as known_module
from autogram.discovery.propose import EnumerationProposer
from autogram.discovery.validate import (
    _condition_signature,
    relation_signature_matches,
)
from autogram.dsl.grammar import Grammar


def _frame():
    # ref (origination) big; two real demand cells; a self cell and a dead cell that are all-zero.
    names = ["orig", "d1", "d2", "self", "dead"]
    n = 100
    mat = np.zeros((n, len(names)))
    mat[:, 0] = 1000.0 + np.arange(n)      # orig ~ 1000
    mat[:, 1] = 400.0                       # real cell
    mat[:, 2] = 600.0                       # real cell
    mat[:, 3] = 0.0                         # self-demand: structurally zero
    mat[:, 4] = 0.0                         # dead destination: structurally zero
    return Frame(mat, names)


def test_drop_negligible_removes_zero_columns_only():
    f = _frame()
    kept = _drop_negligible(frozenset({"d1", "d2", "self", "dead"}), "orig", f, zero_tol=1e-4)
    assert kept == frozenset({"d1", "d2"})          # zero-carriers dropped, real cells kept


def test_canonicalize_equates_incl_and_excl_self():
    f = _frame()
    incl = ("ref_sum", ("orig", frozenset({"d1", "d2", "self"})))   # user's phrasing (incl self)
    excl = ("ref_sum", ("orig", frozenset({"d1", "d2"})))           # engine's learned family
    assert _canonicalize(incl, f, 1e-4) == _canonicalize(excl, f, 1e-4)


def test_canonicalize_does_not_equate_genuinely_different_sums():
    f = _frame()
    full = ("ref_sum", ("orig", frozenset({"d1", "d2"})))
    missing_big = ("ref_sum", ("orig", frozenset({"d1"})))          # drops a real 40%-of-total cell
    assert _canonicalize(full, f, 1e-4) != _canonicalize(missing_big, f, 1e-4)


def test_zero_tol_zero_is_exact_match():
    f = _frame()
    incl = ("ref_sum", ("orig", frozenset({"d1", "d2", "self"})))
    excl = ("ref_sum", ("orig", frozenset({"d1", "d2"})))
    # with zero_tol=0 nothing is dropped, so the two column sets remain distinct
    assert _canonicalize(incl, f, 0.0) != _canonicalize(excl, f, 0.0)


def test_canonicalize_never_empties_a_group():
    f = _frame()
    all_zero = ("ref_sum", ("orig", frozenset({"self", "dead"})))
    canon = _canonicalize(all_zero, f, 1e-4)
    assert canon[1][1] == frozenset({"self", "dead"})   # guard: not reduced to empty


def test_non_sum_signatures_pass_through_unchanged():
    f = _frame()
    for sig in [("pair", frozenset({"a", "b"})), ("zero", "a"), ("one_sided", "a", ">=")]:
        assert _canonicalize(sig, f, 1e-4) == sig


def test_typed_categorical_signatures_are_stable_and_distinct():
    known_true = _known_condition_signature({"kind": True})
    known_one = _known_condition_signature({"kind": 1})
    learned_true = _condition_signature(
        A.Condition("kind", "==", (True,))
    )
    learned_one = _condition_signature(
        A.Condition("kind", "==", (1,))
    )

    assert known_true == learned_true
    assert known_one == learned_one
    assert known_true != known_one
    assert len({known_true, known_one}) == 2

    known_membership = _known_condition_signature({
        "kind_in": ["1", 1],
    })
    learned_membership = _condition_signature(
        A.Condition("kind", "in", (1, "1"))
    )
    assert known_membership == learned_membership

    float_category = (
        "categorical_definition",
        (
            "label",
            (("flag", typed_signature_value(1.0)),),
            typed_signature_value("none"),
        ),
    )
    int_category = (
        "categorical_definition",
        (
            "label",
            (("flag", typed_signature_value(1)),),
            typed_signature_value("none"),
        ),
    )
    assert not relation_signature_matches(float_category, int_category)
    nearby_float = (
        "categorical_definition",
        (
            "label",
            (("flag", typed_signature_value(1.005)),),
            typed_signature_value("none"),
        ),
    )
    assert not relation_signature_matches(float_category, nearby_float)


def test_known_categorical_signatures_canonicalize_only_same_label_blocks():
    def priority(name, cases):
        return KnownInvariant(
            name,
            ":=",
            "label",
            {
                "priority": [
                    {"when": column, "value": value}
                    for column, value in cases
                ],
                "default": "none",
            },
        )

    grouped = priority(
        "grouped",
        (("is_a", "alert"), ("is_b", "alert"), ("is_c", "warning")),
    )
    grouped_permutation = priority(
        "grouped_permutation",
        (("is_b", "alert"), ("is_a", "alert"), ("is_c", "warning")),
    )
    precedence_sensitive = priority(
        "precedence_sensitive",
        (("is_a", "alert"), ("is_c", "warning"), ("is_b", "alert")),
    )
    typed_true = priority("typed_true", (("is_a", True),))
    typed_one = priority("typed_one", (("is_a", 1),))

    assert _signature(grouped) == _signature(grouped_permutation)
    assert _signature(grouped) != _signature(precedence_sensitive)
    assert _signature(typed_true) != _signature(typed_one)


def test_known_conditions_reject_missing_and_character_iterable_memberships():
    assert _known_condition_signature({"kind": np.nan}) is None
    assert _known_condition_signature({"kind_in": "ab"}) is None
    assert _known_condition_signature({"kind_in": []}) is None
    assert _known_condition_signature({"all": "ab"}) is None
    assert _known_condition_signature({"all": []}) is None


def test_known_memberships_typed_deduplicate_and_canonicalize_singletons():
    assert _known_condition_signature({
        "kind_in": [True, 1, True, 1],
    }) == (
        "kind",
        "in",
        (
            typed_signature_value(True),
            typed_signature_value(1),
        ),
    )
    assert _known_condition_signature({
        "label_in": ["active", "active"],
    }) == (
        "label",
        "==",
        (typed_signature_value("active"),),
    )
    assert _known_condition_signature({
        "label_in": ["active", "idle", "active"],
    }) == (
        "label",
        "in",
        (
            typed_signature_value("active"),
            typed_signature_value("idle"),
        ),
    )


def test_load_known_canonicalizes_typed_memberships(tmp_path):
    path = tmp_path / "typed-memberships.json"
    path.write_text(json.dumps({
        "invariants": [
            {
                "name": "typed_multi",
                "op": "==",
                "lhs": "left",
                "rhs": "right",
                "where": {
                    "kind_in": [True, 1, True, 1],
                },
            },
            {
                "name": "singleton",
                "op": "==",
                "lhs": "left",
                "rhs": "right",
                "where": {
                    "label_in": ["active", "active"],
                },
            },
        ],
    }))

    typed_multi, singleton = load_known(str(path))

    assert tuple(type(value) for value in typed_multi.where["kind_in"]) == (
        bool,
        int,
    )
    assert typed_multi.where["kind_in"] == [True, 1]
    assert singleton.where == {"label": "active"}
    assert _signature(typed_multi)[1][0] == (
        "kind",
        "in",
        (
            typed_signature_value(True),
            typed_signature_value(1),
        ),
    )
    assert _signature(singleton)[1][0] == (
        "label",
        "==",
        (typed_signature_value("active"),),
    )


@pytest.mark.parametrize(
    ("where", "message"),
    [
        ({"all": []}, "exactly 2"),
        ({"all": [{"kind": "active"}]}, "exactly 2"),
        (
            {
                "all": [
                    {"c0": 0},
                    {"c1": 1},
                    {"c2": 2},
                ],
            },
            "exactly 2",
        ),
        (
            {
                "all": [
                    {"c0": 0},
                    {"c1": 1},
                    {"c2": 2},
                    {"c3": 3},
                ],
            },
            "exactly 2",
        ),
        (
            {
                "all": [
                    {"kind": "active"},
                    {"label_in": ["idle", "paused"]},
                ],
            },
            "children must use equality",
        ),
        (
            {
                "all": [
                    {"kind": "active"},
                    {"kind": "idle"},
                ],
            },
            "distinct columns",
        ),
        (
            {
                "all": [
                    {
                        "all": [
                            {"kind": "active"},
                            {"label": "normal"},
                        ],
                    },
                    {"state": "ready"},
                ],
            },
            "may not nest",
        ),
    ],
)
def test_load_known_rejects_unexecutable_condition_composites(
    tmp_path,
    where,
    message,
):
    path = tmp_path / "invalid-condition-composite.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "guarded",
            "op": "==",
            "lhs": "left",
            "rhs": "right",
            "where": where,
        }],
    }))

    with pytest.raises(ValueError, match=message):
        load_known(str(path))


@pytest.mark.parametrize("value", [True, 1])
def test_load_known_does_not_infer_conjunction_domains_from_literals(
    tmp_path,
    value,
):
    path = tmp_path / "structural-conjunction.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "guarded",
            "op": "==",
            "lhs": "left",
            "rhs": "right",
            "where": {
                "all": [
                    {"first": value},
                    {"second": value},
                ],
            },
        }],
    }))

    invariant = load_known(str(path))[0]

    assert invariant.where == {
        "all": [
            {"first": value},
            {"second": value},
        ],
    }


def test_load_known_decodes_tagged_timestamp_condition(tmp_path):
    timestamp = pd.Timestamp("2026-08-26T03:55:17.072-05:00")
    path = tmp_path / "timestamp-condition.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "guarded",
            "op": "==",
            "lhs": "left",
            "rhs": "right",
            "where": {
                "observed_at": scalar_to_json(
                    timestamp,
                    "known condition",
                ),
            },
        }],
    }))

    invariant = load_known(str(path))[0]

    assert invariant.where == {"observed_at": timestamp}
    assert _known_condition_signature(invariant.where) == (
        "observed_at",
        "==",
        (typed_signature_value(timestamp),),
    )


def test_runtime_known_condition_validation_uses_complete_domains():
    numeric = KnownInvariant(
        "numeric",
        "==",
        "left",
        "right",
        where={
            "all": [
                {"first": 1},
                {"second": 1},
            ],
        },
    )

    validate_known_conditions(
        [numeric],
        {
            "first": (0, 1, 2),
            "second": (1, 2, 3),
        },
        max_condition_values=2,
    )

    with pytest.raises(
        ValueError,
        match="non-binary categorical columns",
    ):
        validate_known_conditions(
            [numeric],
            {
                "first": (0, 1),
                "second": (0, 1),
            },
            max_condition_values=2,
        )


def test_runtime_known_condition_validation_is_typed_exact_and_capped():
    typed_mismatch = KnownInvariant(
        "typed",
        "==",
        "left",
        "right",
        where={"kind": True},
    )
    oversized_membership = KnownInvariant(
        "membership",
        "==",
        "left",
        "right",
        where={"kind_in": [1, 2, 3]},
    )

    with pytest.raises(ValueError, match="not observed"):
        validate_known_conditions(
            [typed_mismatch],
            {"kind": (1, 2, 3)},
            max_condition_values=3,
        )
    with pytest.raises(ValueError, match="value cap"):
        validate_known_conditions(
            [oversized_membership],
            {"kind": (1, 2, 3, 4)},
            max_condition_values=2,
        )


@pytest.mark.parametrize(
    ("where", "canonical"),
    [
        (
            {"label": "normal"},
            {"label": "normal"},
        ),
        (
            {"label_in": ["normal", "alert"]},
            {"label_in": ["alert", "normal"]},
        ),
        (
            {
                "all": [
                    {"archetype": "steady"},
                    {"label": "normal"},
                ],
            },
            {
                "all": [
                    {"archetype": "steady"},
                    {"label": "normal"},
                ],
            },
        ),
    ],
    ids=("equality", "membership", "categorical-conjunction"),
)
def test_known_condition_accepted_form_matrix(where, canonical):
    assert _canonical_known_condition(where) == canonical


@pytest.mark.parametrize(
    "invariant",
    [
        KnownInvariant(
            "pair",
            "==",
            "left",
            "right",
            where={"regime": "active"},
        ),
        KnownInvariant(
            "band",
            "~band",
            "signal",
            {"center": 1.0},
            where={"regime": "active"},
        ),
        KnownInvariant(
            "proportional",
            "~∝",
            "left",
            "right",
            where={"regime": "active"},
        ),
        KnownInvariant(
            "delta_bound",
            ">",
            {"delta": "counter"},
            0,
            where={"regime": "active"},
        ),
        KnownInvariant(
            "delta_zero",
            "~=",
            {"delta": "counter"},
            0,
            where={"regime": "active"},
        ),
    ],
    ids=("pair", "band", "proportional", "delta-bound", "delta-zero"),
)
def test_known_conditions_accept_only_attached_rule_families(invariant):
    assert _signature(invariant)[0] == "conditional"


@pytest.mark.parametrize(
    "invariant",
    [
        KnownInvariant(
            "one_sided",
            ">=",
            "signal",
            0,
            where={"regime": "active"},
        ),
        KnownInvariant(
            "ratio",
            "==",
            "ratio",
            {"ratio": ["left", "right"]},
            where={"regime": "active"},
        ),
        KnownInvariant(
            "related",
            "==",
            "signal",
            {"related": "raw_signal"},
            where={"regime": "active"},
        ),
        KnownInvariant(
            "definition",
            ":=",
            "alert",
            {
                "sustained": {
                    "term": "signal",
                    "op": ">",
                    "threshold": 0,
                    "window": 2,
                },
            },
            where={"regime": "active"},
        ),
    ],
    ids=("one-sided", "ratio", "related", "definition"),
)
def test_known_conditions_reject_unattached_rule_families(invariant):
    with pytest.raises(ValueError, match="conditions are not enumerable"):
        _signature(invariant)


def test_enumerated_condition_forms_recover_known_relations(monkeypatch):
    grammar = Grammar(
        binders=("record",),
        ops=("==",),
        ref_roles={"record": ("left", "right")},
        fam_roles={"record": ()},
        conditional_enabled=True,
        condition_columns={
            "archetype": ("steady", "bursty"),
            "label": ("normal", "alert", "idle"),
        },
        max_condition_values=2,
    )
    known = [
        KnownInvariant(
            "categorical_pair",
            "==",
            "left",
            "right",
            where={
                "all": [
                    {"archetype": "steady"},
                    {"label": "normal"},
                ],
            },
        ),
        KnownInvariant(
            "membership",
            "==",
            "left",
            "right",
            where={"label_in": ["normal", "alert"]},
        ),
    ]
    pair = (
        "equality",
        "exact",
        ("pair", frozenset({"left", "right"})),
    )
    learned = {
        (
            "conditional",
            (_condition_signature(rule.condition), pair),
        )
        for rule in EnumerationProposer(grammar).propose()
        if rule.condition is not None
        and isinstance(rule.atom, A.Compare)
        and rule.atom.op == "=="
        and {
            rule.atom.left,
            rule.atom.right,
        } == {
            A.Ref("left"),
            A.Ref("right"),
        }
    }

    assert {_signature(invariant) for invariant in known} <= learned
    monkeypatch.setattr(
        known_module,
        "portfolio_relations",
        lambda _result, **_kwargs: learned,
    )
    monkeypatch.setattr(
        known_module,
        "_one_sided_columns",
        lambda _result, _op: set(),
    )
    result = SimpleNamespace(
        dataset=SimpleNamespace(
            observed=Frame(
                np.column_stack((
                    np.arange(8, dtype=float),
                    np.arange(8, dtype=float),
                )),
                ["left", "right"],
            ),
        ),
        portfolio=[],
    )

    report = recover_known(result, known)

    assert report["recovered"] == 2
    assert report["recall"] == 1.0


@pytest.mark.parametrize(
    "predicates",
    [
        [],
        [{"bound": ["signal", ">", 0]}],
        [
            {"bound": [f"signal_{index}", ">", 0]}
            for index in range(17)
        ],
    ],
)
def test_load_known_rejects_unexecutable_definition_conjunction_arity(
    tmp_path,
    predicates,
):
    path = tmp_path / "invalid-definition-conjunction.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "alert",
            "op": ":=",
            "lhs": "alert",
            "rhs": {"and": predicates},
        }],
    }))

    with pytest.raises(ValueError, match="between 2 and 16"):
        load_known(str(path))


def test_load_known_rejects_nested_definition_conjunction(tmp_path):
    path = tmp_path / "nested-definition-conjunction.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "alert",
            "op": ":=",
            "lhs": "alert",
            "rhs": {
                "and": [
                    {
                        "and": [
                            {"bound": ["signal", ">", 0]},
                            {"bound": ["signal", "<", 10]},
                        ],
                    },
                    {"bound": ["other", ">", 0]},
                ],
            },
        }],
    }))

    with pytest.raises(
        ValueError,
        match="bound predicate must contain exactly one discriminator",
    ):
        load_known(str(path))


def test_known_definition_conjunction_signature_requires_two_predicates():
    with pytest.raises(ValueError, match="between 2 and 16"):
        _signature(KnownInvariant(
            "singleton",
            ":=",
            "alert",
            {"and": [{"bound": ["signal", ">", 0]}]},
        ))

    signature = _signature(KnownInvariant(
        "valid",
        ":=",
        "alert",
        {
            "and": [
                {"bound": ["signal", ">", 0]},
                {"bound": ["signal", "<", 10]},
            ],
        },
    ))
    assert signature[0] == "conjunction_definition"
    assert len(signature[1][1]) == 2


def test_recovery_matches_canonicalized_known_memberships(monkeypatch):
    frame = Frame(
        np.column_stack((
            np.arange(8, dtype=float),
            np.arange(8, dtype=float),
        )),
        ["left", "right"],
    )
    pair = (
        "equality",
        "exact",
        ("pair", frozenset({"left", "right"})),
    )
    learned = {
        (
            "conditional",
            (
                (
                    "label",
                    "==",
                    (typed_signature_value("active"),),
                ),
                pair,
            ),
        ),
        (
            "conditional",
            (
                (
                    "kind",
                    "in",
                    (
                        typed_signature_value(True),
                        typed_signature_value(1),
                    ),
                ),
                pair,
            ),
        ),
    }
    monkeypatch.setattr(
        known_module,
        "portfolio_relations",
        lambda _result, **_kwargs: learned,
    )
    monkeypatch.setattr(
        known_module,
        "_one_sided_columns",
        lambda _result, _op: set(),
    )
    result = SimpleNamespace(
        dataset=SimpleNamespace(observed=frame),
        portfolio=[],
    )
    known = [
        KnownInvariant(
            "singleton_membership",
            "==",
            "left",
            "right",
            where={"label_in": ["active", "active"]},
        ),
        KnownInvariant(
            "typed_membership",
            "==",
            "left",
            "right",
            where={"kind_in": [True, 1, True, 1]},
        ),
    ]

    report = recover_known(result, known)

    assert report["recovered"] == 2
    assert report["recall"] == 1.0


@pytest.mark.parametrize("unknown_key", ["wher", "note"])
def test_load_known_rejects_unknown_top_level_keys_before_construction(
    monkeypatch,
    tmp_path,
    unknown_key,
):
    path = tmp_path / "unknown-key.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "valid_shape",
            "op": "==",
            "lhs": "left",
            "rhs": "right",
            unknown_key: {"kind": "active"},
        }],
    }))
    monkeypatch.setattr(
        known_module,
        "KnownInvariant",
        lambda *args, **kwargs: pytest.fail(
            "KnownInvariant constructed before entry key validation"
        ),
    )

    with pytest.raises(
        ValueError,
        match=rf"known invariant entry 0 has unexpected key\(s\): "
        rf"'{unknown_key}'",
    ):
        load_known(str(path))


@pytest.mark.parametrize("missing_key", ["name", "op", "lhs", "rhs"])
def test_load_known_rejects_missing_required_top_level_keys(
    tmp_path,
    missing_key,
):
    invariant = {
        "name": "complete",
        "op": "==",
        "lhs": "left",
        "rhs": "right",
    }
    del invariant[missing_key]
    path = tmp_path / "missing-key.json"
    path.write_text(json.dumps({"invariants": [invariant]}))

    with pytest.raises(
        ValueError,
        match=rf"known invariant entry 0 is missing required key\(s\): "
        rf"'{missing_key}'",
    ):
        load_known(str(path))


@pytest.mark.parametrize(
    "where",
    [
        None,
        {
            "kind": "active",
            "kind_in": ["active"],
        },
    ],
)
def test_load_known_rejects_ambiguous_where_forms(tmp_path, where):
    path = tmp_path / "ambiguous-where.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "ambiguous_where",
            "op": "==",
            "lhs": "left",
            "rhs": "right",
            "where": where,
        }],
    }))

    with pytest.raises(ValueError, match="has an invalid condition"):
        load_known(str(path))


@pytest.mark.parametrize(
    ("rhs", "operator_label"),
    [
        (
            {"and": [{"bound": ["signal", "BOGUS", 0]}]},
            "bound",
        ),
        (
            {"and": [{"bound": ["signal", 7, 0]}]},
            "bound",
        ),
        (
            {
                "sustained": {
                    "term": "signal",
                    "op": "BOGUS",
                    "threshold": 0,
                    "window": 3,
                },
            },
            "sustained",
        ),
        (
            {
                "sustained": {
                    "term": "signal",
                    "op": ["<"],
                    "threshold": 0,
                    "window": 3,
                },
            },
            "sustained",
        ),
    ],
)
def test_load_known_rejects_malformed_nested_comparison_operators(
    tmp_path,
    rhs,
    operator_label,
):
    path = tmp_path / "invalid-operator.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "invalid_operator",
            "op": ":=",
            "lhs": "alert",
            "rhs": rhs,
        }],
    }))

    with pytest.raises(
        ValueError,
        match=rf"{operator_label} operator must be a string "
        r"in the supported comparison set",
    ):
        load_known(str(path))


@pytest.mark.parametrize(
    ("case_value", "default", "label"),
    [
        (["alert"], "normal", "priority case 'value'"),
        ({"category": "alert"}, "normal", "priority case 'value'"),
        (float("nan"), "normal", "priority case 'value'"),
        ("alert", ["normal"], "priority default"),
        ("alert", {"category": "normal"}, "priority default"),
        ("alert", float("inf"), "priority default"),
    ],
)
def test_load_known_rejects_nonscalar_priority_payloads(
    tmp_path,
    case_value,
    default,
    label,
):
    path = tmp_path / "invalid-priority.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "invalid_priority",
            "op": ":=",
            "lhs": "label",
            "rhs": {
                "priority": [{
                    "when": "is_alert",
                    "value": case_value,
                }],
                "default": default,
            },
        }],
    }))

    with pytest.raises(
        ValueError,
        match=rf"{label} must be a JSON scalar",
    ):
        load_known(str(path))


def test_load_known_preserves_valid_typed_priority_scalars(tmp_path):
    values = [None, True, 1, 1.0, "1"]
    path = tmp_path / "typed-priority.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "typed_priority",
            "op": ":=",
            "lhs": "label",
            "rhs": {
                "priority": [
                    {
                        "when": f"case_{index}",
                        "value": value,
                    }
                    for index, value in enumerate(values)
                ],
                "default": None,
            },
        }],
    }))

    invariant = load_known(str(path))[0]
    loaded_values = [
        case["value"]
        for case in invariant.rhs["priority"]
    ]
    assert [type(value) for value in loaded_values] == [
        type(None),
        bool,
        int,
        float,
        str,
    ]
    signature = _signature(invariant)
    assert tuple(
        value_key
        for _column, value_key in signature[1][1]
    ) == tuple(
        typed_signature_value(value)
        for value in values
    )
    assert signature[1][2] == typed_signature_value(None)


@pytest.mark.parametrize(
    ("rhs", "message"),
    [
        (
            {
                "priority": [{
                    "when": "is_alert",
                    "value": "alert",
                }],
            },
            "explicit 'default' key",
        ),
        (
            {
                "and": [{"bound": ["signal", ">", 0]}],
                "priority": [{
                    "when": "is_alert",
                    "value": "alert",
                }],
                "default": "normal",
            },
            "exactly one",
        ),
        (
            {
                "and": [{"bound": ["signal", ">", 0]}],
                "default": None,
            },
            "unexpected top-level key",
        ),
    ],
)
def test_load_known_rejects_nonexclusive_definition_rhs(
    tmp_path,
    rhs,
    message,
):
    path = tmp_path / "invalid-definition.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "invalid_definition",
            "op": ":=",
            "lhs": "result",
            "rhs": rhs,
        }],
    }))

    with pytest.raises(ValueError, match=message):
        load_known(str(path))


def test_load_known_accepts_explicit_null_priority_default(tmp_path):
    path = tmp_path / "null-default.json"
    path.write_text(json.dumps({
        "invariants": [{
            "name": "nullable_priority",
            "op": ":=",
            "lhs": "label",
            "rhs": {
                "priority": [{
                    "when": "is_alert",
                    "value": "alert",
                }],
                "default": None,
            },
        }],
    }))

    invariant = load_known(str(path))[0]
    assert invariant.rhs["default"] is None
    assert _signature(invariant)[1][2] == typed_signature_value(None)


@pytest.mark.parametrize(
    ("invariant", "message"),
    [
        (
            {
                "name": "ambiguous_rhs",
                "op": "==",
                "lhs": "total",
                "rhs": {
                    "sum": ["left", "right"],
                    "ratio": ["left", "right"],
                },
            },
            "structured rhs must contain exactly one discriminator",
        ),
        (
            {
                "name": "ambiguous_lhs",
                "op": ">=",
                "lhs": {
                    "delta": "counter",
                    "lag": ["counter", 1],
                },
                "rhs": 0,
            },
            "structured lhs must contain exactly one discriminator",
        ),
        (
            {
                "name": "extra_rhs_key",
                "op": "==",
                "lhs": "total",
                "rhs": {
                    "sum": ["left", "right"],
                    "note": "ignored before validation",
                },
            },
            r"structured rhs has unexpected key\(s\): 'note'",
        ),
        (
            {
                "name": "extra_lhs_key",
                "op": ">=",
                "lhs": {
                    "delta": "counter",
                    "note": "ignored before validation",
                },
                "rhs": 0,
            },
            r"structured lhs has unexpected key\(s\): 'note'",
        ),
        (
            {
                "name": "ambiguous_nested_term",
                "op": ":=",
                "lhs": "alert",
                "rhs": {
                    "and": [{
                        "bound": [
                            {
                                "roll_sum": [
                                    {
                                        "delta": "signal",
                                        "lag": ["signal", 1],
                                    },
                                    3,
                                ],
                            },
                            ">",
                            0,
                        ],
                    }],
                },
            },
            "term must contain exactly one discriminator",
        ),
        (
            {
                "name": "extra_nested_term_key",
                "op": ":=",
                "lhs": "alert",
                "rhs": {
                    "and": [{
                        "bound": [
                            {
                                "difference": ["left", "right"],
                                "ref": "left",
                            },
                            ">",
                            0,
                        ],
                    }],
                },
            },
            r"term has unexpected key\(s\): 'ref'",
        ),
        (
            {
                "name": "ambiguous_windowed_ratio_operand",
                "op": "==",
                "lhs": "ratio",
                "rhs": {
                    "ratio": [
                        {
                            "roll_sum": ["numerator", 2],
                            "ref": "numerator",
                        },
                        {"roll_sum": ["denominator", 2]},
                    ],
                },
            },
            r"roll_sum reference has unexpected key\(s\): 'ref'",
        ),
        (
            {
                "name": "extra_bound_key",
                "op": ":=",
                "lhs": "alert",
                "rhs": {
                    "and": [{
                        "bound": ["signal", ">", 0],
                        "note": "ignored before validation",
                    }],
                },
            },
            r"bound predicate has unexpected key\(s\): 'note'",
        ),
        (
            {
                "name": "extra_sustained_key",
                "op": ":=",
                "lhs": "alert",
                "rhs": {
                    "sustained": {
                        "term": "signal",
                        "op": ">",
                        "threshold": 0,
                        "window": 3,
                        "note": "ignored before validation",
                    },
                },
            },
            r"sustained predicate has unexpected key\(s\): 'note'",
        ),
        (
            {
                "name": "missing_sustained_key",
                "op": ":=",
                "lhs": "alert",
                "rhs": {
                    "sustained": {
                        "term": "signal",
                        "op": ">",
                        "window": 3,
                    },
                },
            },
            r"sustained predicate is missing required key\(s\): 'threshold'",
        ),
        (
            {
                "name": "extra_priority_case_key",
                "op": ":=",
                "lhs": "label",
                "rhs": {
                    "priority": [{
                        "when": "is_alert",
                        "value": "alert",
                        "note": "ignored before validation",
                    }],
                    "default": "normal",
                },
            },
            r"priority case has unexpected key\(s\): 'note'",
        ),
    ],
)
def test_load_known_rejects_ambiguous_or_extra_nested_structures(
    tmp_path,
    invariant,
    message,
):
    path = tmp_path / "invalid-structured-union.json"
    path.write_text(json.dumps({"invariants": [invariant]}))

    with pytest.raises(ValueError, match=message):
        load_known(str(path))


@pytest.mark.parametrize(
    "path",
    [
        "configs/gtib_known.yaml",
        "configs/gtib_raw_known.yaml",
        "docs/known_invariants.example.yaml",
    ],
)
def test_checked_in_known_catalogs_pass_strict_structure_validation(path):
    invariants = load_known(path)

    assert invariants
    assert all(_signature(invariant) is not None for invariant in invariants)


def test_gtib_known_condition_forms_remain_executable():
    known = {
        invariant.name: invariant
        for invariant in load_known("configs/gtib_known.yaml")
    }

    assert known["steady_normal_healthy_band"].where == {
        "all": [
            {"archetype": "steady"},
            {"label": "normal"},
        ],
    }
    assert known["benign_spans_lose_no_bytes"].where == {
        "label_in": ["artifact", "benign_burst"],
    }
    assert _signature(known["steady_normal_healthy_band"])[0] == (
        "conditional"
    )
    assert _signature(known["benign_spans_lose_no_bytes"])[0] == (
        "conditional"
    )


@pytest.mark.parametrize(
    "invariant",
    [
        KnownInvariant(
            "lag_list",
            ">=",
            {"lag": [["column"], 1]},
            0,
        ),
        KnownInvariant(
            "lag_mapping",
            ">=",
            {"lag": [{"column": "x"}, 1]},
            0,
        ),
        KnownInvariant(
            "delta_number",
            ">=",
            {"delta": 7},
            0,
        ),
        KnownInvariant(
            "delta_mapping",
            "~=",
            {"delta": {"column": "x"}},
            0,
        ),
        KnownInvariant(
            "related_list",
            "==",
            "x",
            {"related": ["raw_x"]},
        ),
        KnownInvariant(
            "related_mapping",
            "==",
            "x",
            {"related": {"role": "raw_x"}},
        ),
        KnownInvariant(
            "ratio_number",
            "==",
            "ratio",
            {"ratio": ["numerator", 2]},
        ),
        KnownInvariant(
            "ratio_mapping",
            "==",
            "ratio",
            {"ratio": [{"column": "numerator"}, "denominator"]},
        ),
        KnownInvariant(
            "windowed_ratio_list",
            "==",
            "ratio",
            {
                "ratio": [
                    {"roll_sum": [["numerator"], 2]},
                    {"roll_sum": ["denominator", 2]},
                ],
            },
        ),
        KnownInvariant(
            "difference_list",
            ":=",
            "flag",
            {
                "and": [{
                    "bound": [
                        {"difference": [["left"], "right"]},
                        ">",
                        0,
                    ],
                }],
            },
        ),
        KnownInvariant(
            "difference_mapping",
            ":=",
            "flag",
            {
                "and": [{
                    "bound": [
                        {
                            "difference": [
                                {"column": "left"},
                                "right",
                            ],
                        },
                        ">",
                        0,
                    ],
                }],
            },
        ),
        KnownInvariant(
            "priority_role",
            ":=",
            "label",
            {
                "priority": [{
                    "when": ["flag"],
                    "value": "active",
                }],
                "default": "inactive",
            },
        ),
    ],
)
def test_structured_known_operands_reject_nonstring_columns_and_roles(
    invariant,
):
    with pytest.raises(ValueError):
        _signature(invariant)


def _bimodal_frame():
    """A member that is zero on 51% of the rows and large on the other 49%.

    Its MEDIAN absolute value is 0, so a central-statistic negligibility test judges it negligible
    and drops it -- crediting ``total == SUM(d1, bimodal)`` as recovered by a learned
    ``total == SUM(d1)`` that is violated on 49% of the rows.
    """
    names = ["total", "d1", "bimodal", "zero"]
    n = 100
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0                    # real cell
    mat[51:, 2] = 1000.0                  # zero on 51 rows, 1000 on 49 rows -> median 0
    mat[:, 0] = mat[:, 1] + mat[:, 2]     # the honest total
    return Frame(mat, names)


def test_bimodal_member_is_not_negligible_even_though_its_median_is_zero():
    f = _bimodal_frame()

    kept = _drop_negligible(
        frozenset({"d1", "bimodal", "zero"}), "total", f, zero_tol=1e-4,
    )

    assert "bimodal" in kept          # materially non-zero on 49% of rows
    assert "zero" not in kept         # identically zero everywhere
    assert kept == frozenset({"d1", "bimodal"})


def test_bimodal_member_keeps_two_sums_distinct():
    f = _bimodal_frame()
    known = ("ref_sum", ("total", frozenset({"d1", "bimodal"})))
    learned = ("ref_sum", ("total", frozenset({"d1"})))

    assert _canonicalize(known, f, 1e-4) != _canonicalize(learned, f, 1e-4)


def test_all_zero_member_is_still_dropped_when_the_anchor_has_zero_rows():
    # The intended use case must survive the pointwise test: a structurally-zero member stays
    # droppable even on data whose anchor is itself zero on some rows.
    names = ["total", "d1", "self"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[10:, 1] = 250.0
    mat[:, 0] = mat[:, 1]
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"d1", "self"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"d1"})


def test_member_larger_than_the_anchor_on_a_single_row_is_kept():
    # A member that is negligible almost everywhere but spikes on one row still changes the sum
    # there, so it cannot be canonicalized away.
    names = ["total", "d1", "spiky"]
    n = 200
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 5000.0
    mat[:, 2] = 1e-9
    mat[7, 2] = 4000.0
    mat[:, 0] = mat[:, 1] + mat[:, 2]
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"d1", "spiky"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"d1", "spiky"})


def test_individually_tiny_members_are_not_dropped_when_they_add_up():
    """Round-29 review: a per-member bound does not bound the sum of the members removed.

    Six hundred members each under the tolerance contributed 6% of the total between them, which is
    outside any plausible acceptance band -- yet all of them were canonicalized away.
    """
    n_members = 600
    names = ["total", "real", *[f"m{index}" for index in range(n_members)]]
    n = 30
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 940.0                    # a materially large member, so the group is never emptied
    for index in range(n_members):
        mat[:, 2 + index] = 0.1          # 1e-4 of the anchor each, 6% together
    mat[:, 0] = 1000.0
    f = Frame(mat, names)
    members = frozenset(names[1:])

    kept = _drop_negligible(members, "total", f, zero_tol=1e-4)

    # Aggregate contribution is material, so nothing may be canonicalized away -- and the result
    # must not depend on the "never empty a group" guard rescuing it.
    assert kept == members


def test_exactly_zero_members_are_still_dropped_alongside_material_ones():
    # The fallback must keep the intended use case working: a structurally-zero member is removable
    # even when other individually-tiny members are not.
    names = ["total", "real", "tiny0", "tiny1", "zero"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 900.0
    mat[:, 2] = 0.09                     # individually negligible ...
    mat[:, 3] = 0.09                     # ... but 0.18 together, over the 0.1 budget
    mat[:, 4] = 0.0
    mat[:, 0] = 1000.0
    f = Frame(mat, names)

    kept = _drop_negligible(
        frozenset({"real", "tiny0", "tiny1", "zero"}), "total", f, zero_tol=1e-4,
    )

    assert "zero" not in kept
    assert kept == frozenset({"real", "tiny0", "tiny1"})


def test_member_nonzero_only_where_the_sum_is_ungradeable_is_dropped():
    """A member that never affects a gradeable row must not split an alias pair.

    The sum is undefined wherever any member is missing, so a member that is non-zero only on those
    rows provably never changes the relation.
    """
    names = ["total", "w", "z"]
    n = 50
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[20:, 1] = np.nan                 # `w` missing on the tail -> the sum is ungradeable there
    mat[20:, 2] = 900.0                  # `z` is large only where the sum cannot be graded
    mat[:, 0] = 500.0
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"w", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"w"})


def test_canonicalize_is_idempotent():
    for f, sig in (
        (_frame(), ("ref_sum", ("orig", frozenset({"d1", "d2", "self", "dead"})))),
        (_bimodal_frame(), ("ref_sum", ("total", frozenset({"d1", "bimodal", "zero"})))),
    ):
        once = _canonicalize(sig, f, 1e-4)
        assert _canonicalize(once, f, 1e-4) == once

    names = ["total", "real", "tiny0", "tiny1", "zero"]
    n = 30
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 900.0
    mat[:, 2] = 0.09
    mat[:, 3] = 0.09
    mat[:, 0] = 1000.0
    f = Frame(mat, names)
    sig = ("ref_sum", ("total", frozenset({"real", "tiny0", "tiny1", "zero"})))
    once = _canonicalize(sig, f, 1e-4)
    assert _canonicalize(once, f, 1e-4) == once


def test_removal_may_not_widen_the_graded_population():
    """Round-30 review: a member's own missingness restricts the sum's domain.

    ``z`` is 0 on ten rows and missing on the other ninety, so ``total == SUM(real, z)`` is graded
    on ten rows only. Dropping ``z`` produced ``total == SUM(real)``, which is graded on all one
    hundred -- and fails on ninety of them. Canonicalisation may not hand the reduced relation rows
    the original never had to satisfy.
    """
    names = ["total", "real", "z"]
    n = 100
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[:, 0] = 500.0
    mat[10:, 0] = 999.0          # `total == SUM(real)` is false on the last ninety rows
    mat[:, 2] = np.nan
    mat[:10, 2] = 0.0            # `z` is defined (and zero) only on the first ten rows
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"real", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"real", "z"})
    assert _canonicalize(("ref_sum", ("total", frozenset({"real", "z"}))), f, 1e-4) != (
        "ref_sum", ("total", frozenset({"real"}))
    )


def test_all_defined_zero_member_is_still_dropped():
    # The domain-preserving requirement must not break the intended case: a member defined on every
    # row where the anchor is defined, and zero throughout, is still removable.
    names = ["total", "real", "z"]
    n = 60
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[:, 0] = 500.0
    mat[:, 2] = 0.0
    mat[40:, 0] = np.nan         # the anchor itself is missing on the tail; `z` still qualifies
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"real", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"real"})


def test_stable_row_sum_is_order_independent_and_exact():
    """Round-30 review: the aggregate bound must not depend on hash-randomised iteration order.

    Floating-point addition is not associative, so accumulating per-column magnitudes in
    ``frozenset``/``dict`` order makes the result depend on string hash randomisation -- and with it
    which members are canonicalized away, and therefore which side of the held-out boundary a
    catalogue entry lands on. The summation must be both deterministically ordered and exact.

    Tested at this seam rather than through ``_drop_negligible``: a whole-function test cannot force
    an adversarial iteration order, so it passes whether or not the summation is ordered, which is
    no test at all (round-32 review).
    """
    from autogram.discovery.known import _stable_row_sum

    columns = {
        "a": np.array([0.03630875, 1e16]),
        "b": np.array([0.04805217, 1.0]),
        "c": np.array([0.04447561, -1e16]),
        "d": np.array([0.02008216, 1.0]),
    }

    reference = _stable_row_sum(columns)
    for order in (("d", "c", "b", "a"), ("b", "a", "d", "c"), ("c", "a", "d", "b")):
        shuffled = {name: columns[name] for name in order}
        assert _stable_row_sum(shuffled).tobytes() == reference.tobytes()

    # Exact, not merely reproducible: naive accumulation of the second row loses the two ones.
    assert reference[1] == 2.0
    assert float(np.stack([columns[name] for name in ("a", "b", "c", "d")], axis=0).sum(axis=0)[1]) != 2.0


def test_member_missing_only_where_a_retained_member_is_also_missing_is_dropped():
    """Round-31 review: domain preservation must compare masks, not each member to the anchor.

    ``z`` is missing exactly where ``w`` is missing, so the sum is ungradeable on those rows either
    way and removing ``z`` changes nothing. Requiring ``z`` to be finite wherever the *anchor* is
    kept it, which split two identically-evaluated sums across the held-out boundary.
    """
    names = ["total", "w", "z"]
    n = 50
    mat = np.zeros((n, len(names)))
    mat[:, 0] = 500.0
    mat[:, 1] = 500.0
    mat[30:, 1] = np.nan          # `w` missing on the tail -> the sum is ungradeable there
    mat[:, 2] = 0.0
    mat[30:, 2] = np.nan          # `z` missing on exactly the same rows
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"w", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"w"})


def test_stable_row_sum_reports_an_unrepresentable_total_as_infinite():
    """An aggregate beyond float64 is past any finite budget; it must not crash the run."""
    from autogram.discovery.known import _stable_row_sum

    columns = {
        f"m{index}": np.array([1.5e308, 1.0])
        for index in range(8)
    }

    total = _stable_row_sum(columns)

    assert np.isinf(total[0])
    assert total[1] == 8.0


def test_exact_relation_may_only_drop_identically_zero_members():
    """Round-33 review: an exact relation has no tolerance to spend.

    A member that is merely small still breaks ``total == SUM(...)`` on every row it is non-zero, so
    crediting a learned exact sum with recovering a known exact sum that omits it reports a law the
    data does not satisfy anywhere.
    """
    names = ["total", "a", "z"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0
    mat[:, 2] = 0.05                 # 5e-5 of the anchor: under zero_tol, but not zero
    mat[:, 0] = 1000.05
    f = Frame(mat, names)

    known = ("equality", "exact", ("ref_sum", ("total", frozenset({"a", "z"}))))
    learned = ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))
    assert _canonicalize(known, f, 1e-4) != _canonicalize(learned, f, 1e-4)

    # An approximate relation may still absorb it -- that is what its tolerance is for.
    known_approx = ("equality", "approximate", ("ref_sum", ("total", frozenset({"a", "z"}))))
    learned_approx = ("equality", "approximate", ("ref_sum", ("total", frozenset({"a"}))))
    assert _canonicalize(known_approx, f, 1e-4) == _canonicalize(learned_approx, f, 1e-4)


def test_exact_relation_still_drops_a_structurally_zero_member():
    names = ["total", "a", "z"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0
    mat[:, 2] = 0.0
    mat[:, 0] = 1000.0
    f = Frame(mat, names)

    known = ("equality", "exact", ("ref_sum", ("total", frozenset({"a", "z"}))))
    learned = ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))

    assert _canonicalize(known, f, 1e-4) == _canonicalize(learned, f, 1e-4)


def test_exact_zero_member_that_is_missing_elsewhere_is_not_dropped():
    """Round-34 review: exactness must not bypass the domain-preserving fixpoint.

    A member that is zero where defined but MISSING elsewhere restricts the sum's domain, so
    removing it hands the reduced relation rows the original never had to satisfy -- an exact known
    graded on ten rows credited to a learned sum that fails on the other ninety.
    """
    names = ["total", "real", "z"]
    n = 100
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[:, 0] = 500.0
    mat[10:, 0] = 999.0          # `total == SUM(real)` is false on the last ninety rows
    mat[:, 2] = np.nan
    mat[:10, 2] = 0.0            # `z` is defined (and exactly zero) only on the first ten rows
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"real", "z"}), "total", f, zero_tol=1e-4, exact=True)

    assert kept == frozenset({"real", "z"})


def test_exact_learned_sum_recovers_the_approximate_known_it_satisfies():
    """Round-35 review: matching must use the tolerance the KNOWN relation permits.

    Canonicalising the learned side by its own exactness left an exact learned
    ``total == SUM(a, z)`` unmatchable by the approximate known ``total ~= SUM(a)`` it satisfies,
    reporting recall 0.0 for an invariant the portfolio does contain.
    """
    names = ["total", "a", "z"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0
    mat[:, 2] = 0.05                 # z/total ~ 5e-5, inside the approximate tolerance
    mat[:, 0] = 1000.05
    f = Frame(mat, names)

    learned_exact = ("equality", "exact", ("ref_sum", ("total", frozenset({"a", "z"}))))
    known_approx = ("equality", "approximate", ("ref_sum", ("total", frozenset({"a"}))))

    # Canonicalised under the KNOWN relation's tolerance, the two describe one relation ...
    assert (
        _canonicalize(learned_exact, f, 1e-4, exact=False)[2]
        == _canonicalize(known_approx, f, 1e-4)[2]
    )
    # ... while an exact known still refuses to absorb the same member.
    known_exact = ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))
    assert (
        _canonicalize(learned_exact, f, 1e-4, exact=True)[2]
        != _canonicalize(known_exact, f, 1e-4)[2]
    )


def test_recovery_uses_approximate_known_tolerance_for_exact_witness(
    monkeypatch,
):
    names = ["total", "a", "z"]
    matrix = np.column_stack((
        np.full(40, 1000.05),
        np.full(40, 1000.0),
        np.full(40, 0.05),
    ))
    frame = Frame(matrix, names)
    learned_exact = (
        "equality",
        "exact",
        ("ref_sum", ("total", frozenset({"a", "z"}))),
    )
    monkeypatch.setattr(
        known_module,
        "portfolio_relations",
        lambda _result, **_kwargs: {learned_exact},
    )
    monkeypatch.setattr(
        known_module,
        "_one_sided_columns",
        lambda _result, _op: set(),
    )
    result = SimpleNamespace(
        dataset=SimpleNamespace(observed=frame),
        portfolio=[],
    )
    known = KnownInvariant(
        "approximate",
        "~=",
        "total",
        {"sum": ["a"]},
    )

    assert recover_known(result, [known])["recall"] == 1.0


def test_overlapping_aggregate_balance_is_not_set_canonicalized():
    frame = Frame(
        np.ones((20, 5)),
        ["a", "b", "c", "d", "e"],
    )
    overlapping = (
        "agg_ref_balance",
        frozenset({
            ("a", frozenset({"a", "b"})),
            ("c", frozenset({"d", "e"})),
        }),
    )

    assert _canonicalize(
        overlapping,
        frame,
        1e-4,
        exact=True,
    ) == overlapping


def test_singleton_sum_balance_canonicalizes_to_ref_sum_alias():
    names = ["total", "a", "b"]
    matrix = np.column_stack((
        np.full(40, 10.0),
        np.full(40, 4.0),
        np.full(40, 6.0),
    ))
    frame = Frame(matrix, names)
    learned_add = (
        "equality",
        "exact",
        (
            "sum_balance",
            frozenset({
                frozenset({"total"}),
                frozenset({"a", "b"}),
            }),
        ),
    )
    known_ref_sum = (
        "equality",
        "exact",
        ("ref_sum", ("total", frozenset({"a", "b"}))),
    )

    assert _canonicalize(learned_add, frame, 1e-4) == _canonicalize(
        known_ref_sum,
        frame,
        1e-4,
    )


def test_singleton_sums_canonicalize_to_scalar_pair():
    frame = Frame(
        np.column_stack((
            np.ones(40),
            np.ones(40),
        )),
        ["a", "c"],
    )
    pair = (
        "equality",
        "exact",
        ("pair", frozenset({"a", "c"})),
    )
    ref_sum = (
        "equality",
        "exact",
        ("ref_sum", ("a", frozenset({"c"}))),
    )
    balance = (
        "equality",
        "exact",
        (
            "sum_balance",
            frozenset({
                frozenset({"a"}),
                frozenset({"c"}),
            }),
        ),
    )

    assert _canonicalize(ref_sum, frame, 1e-4) == pair
    assert _canonicalize(balance, frame, 1e-4) == pair


def test_conditional_exactness_is_detected_through_the_nesting():
    """Round-36 review: a conditioned exact equality is still exact.

    Reading only the outer shape canonicalised it with an approximate tolerance, which matched a
    conditioned exact known against a learned sum it is false against on every applicable row.
    """
    from autogram.discovery.known import _candidate_is_exact

    base_exact = ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))
    base_approx = ("equality", "approximate", ("ref_sum", ("total", frozenset({"a"}))))
    condition = ("label", "==", ("alert",))

    assert _candidate_is_exact(("conditional", (condition, base_exact)))
    assert not _candidate_is_exact(("conditional", (condition, base_approx)))
    assert _candidate_is_exact(base_exact)
    assert not _candidate_is_exact(("zero", "a"))


def test_conditional_exact_known_is_not_credited_by_a_different_sum():
    names = ["total", "a", "z"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0
    mat[:, 2] = 0.05
    mat[:, 0] = 1000.05
    f = Frame(mat, names)
    condition = ("label", "==", ("alert",))

    known = ("conditional", (condition, ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))))
    learned = ("conditional", (condition, ("equality", "exact", ("ref_sum", ("total", frozenset({"a", "z"}))))))

    assert _canonicalize(known, f, 1e-4) != _canonicalize(learned, f, 1e-4, exact=True)
