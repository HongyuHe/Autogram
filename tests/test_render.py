"""Explicit (schema-aware) rendering + its inverse parser -- fast, no live model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from dateutil import tz as dateutil_tz

from autogram.discovery.export import portfolio_to_dl
from autogram.dsl import ast as A
from autogram.dsl.evaluate import typed_condition_key, typed_group_key
from autogram.dsl.parser import rule_from_dict, rule_to_dict
from autogram.dsl.render import parse_rule_line, render_rule
from autogram.dsl.scalar_codec import scalar_from_json, scalar_to_json
from autogram.schema.spec import FamilySelector


@pytest.mark.parametrize(
    "text",
    [
        "[forall record] x ~band nan",
        "[forall record] x ~band inf",
        "[forall record] target := x < nan",
        "[forall record] target := x >= inf",
        "[forall record] x == nan",
        "[forall record] x == -inf",
        "[forall record] x == nan*y",
        "[forall record] x == y where kind == NaN",
        "[forall record] x == y where kind == Infinity",
        "[forall record] x == y where kind == -Infinity",
        "[forall record] x == y where kind == 1e400",
        "[forall record] x == y where kind in (1, -1e400)",
        (
            "[forall record] target := "
            "PRIORITY(flag->NaN; default=\"fallback\")"
        ),
        (
            "[forall record] target := "
            "PRIORITY(flag->\"alert\"; default=Infinity)"
        ),
    ],
)
def test_surface_parser_rejects_nonfinite_numbers(text):
    with pytest.raises(ValueError, match="finite"):
        parse_rule_line(text)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_surface_render_rejects_nonfinite_condition_and_category_scalars(value):
    rules = (
        A.Rule(
            "record",
            A.Compare(A.Ref("x"), "==", A.Ref("y")),
            condition=A.Condition("kind", "==", (value,)),
        ),
        A.Rule(
            "record",
            A.CategoryDefinition(
                "target",
                (("flag", value),),
                "fallback",
            ),
        ),
    )

    for rule in rules:
        with pytest.raises(ValueError, match="finite"):
            render_rule(rule)


def test_quoted_nonfinite_words_remain_string_scalars():
    rule = parse_rule_line(
        '[forall record] target := '
        'PRIORITY(flag->"NaN"; default="Infinity") '
        'where kind == "1e400"'
    )

    assert rule.atom.cases == (("flag", "NaN"),)
    assert rule.atom.default == "Infinity"
    assert rule.condition.values == ("1e400",)


def test_surface_parser_round_trips_scientific_notation():
    rules = (
        A.Rule(
            "record",
            A.Compare(A.Ref("x"), "==", A.Const(1e20)),
        ),
        A.Rule(
            "record",
            A.Compare(
                A.Scale(1e-20, A.Ref("x")),
                ">=",
                A.Const(-2.5e-12),
            ),
        ),
    )

    for rule in rules:
        assert parse_rule_line(rule.unparse()) == rule


def test_surface_round_trip_preserves_exact_floats_and_term_structure():
    x = A.Ref("x")
    y = A.Ref("y")
    z = A.Ref("z")
    rules = (
        A.Rule(
            "record",
            A.Compare(A.Scale(1.2345671, x), "~=", y),
        ),
        A.Rule(
            "record",
            A.Compare(A.Scale(1.2345672, x), "~=", y),
        ),
        A.Rule(
            "record",
            A.Compare(
                A.Lag(A.Scale(2.0, A.Add((x, y))), 1),
                "==",
                z,
            ),
        ),
        A.Rule(
            "record",
            A.Compare(
                A.Lag(A.Add((A.Scale(2.0, x), y)), 1),
                "==",
                z,
            ),
        ),
        A.Rule(
            "record",
            A.Compare(A.Mul(A.Add((x, y)), z), "==", A.Const(-0.0)),
        ),
        A.Rule(
            "record",
            A.Compare(A.Div(x, A.Add((y, z))), "==", A.Const(0.0)),
        ),
        A.Rule(
            "record",
            A.Compare(A.Add((A.Add((x, y)), z)), "==", A.Const(0.0)),
        ),
        A.Rule(
            "record",
            A.Compare(A.Add((x,)), "==", A.Const(0.0)),
        ),
        A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("flag"),
                A.Bound(x, "<", 1.2345671),
            ),
        ),
        A.Rule(
            "record",
            A.BandDefinition(x, 1.2345672),
        ),
    )

    rendered = [rule.unparse() for rule in rules]
    assert rendered[0] != rendered[1]
    assert rendered[2] != rendered[3]
    assert "LAG_1(2*(x + y))" in rendered[2]
    assert "LAG_1(2*x + y)" in rendered[3]

    for rule, text in zip(rules, rendered):
        parsed = parse_rule_line(text)
        assert parsed.signature() == rule.signature()


class _StubAdapter:
    """Minimal duck-typed adapter carrying just what the renderer reads (CrossCheck-shaped)."""

    binders = ("cell", "node", "link", "network")
    binder_enumerate = {"cell": "per_measured_col", "node": "per_node",
                        "link": "per_directed_link", "network": "singleton"}
    noisy_kind = "low"
    demand_kind = "high"
    ref_templates = {
        ("cell", "self"): "{col}",
        ("node", "measurement_origination"): "low_{X}_origination",
        ("node", "measurement_termination"): "low_{X}_termination",
        ("link", "o0"): "low_{X}_egress_to_{Y}",
        ("link", "o0_rev"): "low_{Y}_egress_to_{X}",
        ("link", "demand"): "high_{X}_{Y}",
    }
    family_selectors = {
        ("node", "demand_row"): FamilySelector("node", "demand_row", "high", "demand",
                                               (("source", "==", "X"), ("destination", "!=", "X"))),
        ("node", "demand_col"): FamilySelector("node", "demand_col", "high", "demand",
                                               (("destination", "==", "X"), ("source", "!=", "X"))),
        ("network", "all_demand"): FamilySelector("network", "all_demand", "high", "demand",
                                                  (("source", "!=", "@destination"),)),
        ("network", "all_measurement_origination"):
            FamilySelector("network", "all_measurement_origination", "low", "origination", ()),
    }


AD = _StubAdapter()


def test_quantifier_names_bound_variables():
    r = parse_rule_line("[forall node] measurement_origination >= 0")
    assert render_rule(r, AD) == "[forall node X] low_{X}_origination >= 0"
    r2 = parse_rule_line("[forall cell] self >= 0")
    assert render_rule(r2, AD) == "[forall cell col] {col} >= 0"


def test_directed_link_expands_both_endpoints():
    r = parse_rule_line("[forall link] o0 != o0_rev")
    assert render_rule(r, AD) == "[forall link X, Y] low_{X}_egress_to_{Y} != low_{Y}_egress_to_{X}"


def test_family_sum_shows_index_and_membership():
    # the node conservation balance -> explicit indexed sums with the ≠X membership predicate
    r = parse_rule_line(
        "[forall node] measurement_origination + SUM(demand_col) ~= "
        "measurement_termination + SUM(demand_row)")
    assert render_rule(r, AD) == (
        "[forall node X] low_{X}_origination + \u03a3_{j\u2260X} high_{j}_{X} ~= "
        "low_{X}_termination + \u03a3_{j\u2260X} high_{X}_{j}")


def test_network_family_uses_two_dummies_with_inequality():
    r = parse_rule_line("[forall network] 0 ~= MIN(all_demand)")
    assert render_rule(r, AD) == "[forall network] 0 ~= min_{j\u2260k} high_{j}_{k}"


def test_unknown_family_falls_back_to_compact():
    # a family the adapter has no selector for keeps the compact SUM(role) form
    r = parse_rule_line("[forall network] 0 ~= SUM(mystery)")
    assert "SUM(mystery)" in render_rule(r, AD)


def test_render_falls_back_to_unparse_without_adapter():
    r = parse_rule_line("[forall node] measurement_origination >= 0")
    assert render_rule(r, None) == r.unparse()


def test_parse_is_inverse_of_unparse():
    rules = [
        A.Rule("cell", A.Compare(A.Ref("self"), ">=", A.Const(0))),
        A.Rule("link", A.Compare(A.Ref("o0"), "!=", A.Ref("o0_rev"))),
        A.Rule("node", A.Compare(
            A.Add((A.Ref("measurement_origination"), A.Agg("SUM", "demand_col"))), "~=",
            A.Add((A.Ref("measurement_termination"), A.Agg("SUM", "demand_row"))))),
        A.Rule("link", A.Compare(A.Scale(-1.0, A.Ref("o0")), "<=", A.Const(0))),
        A.Rule("link", A.Compare(A.Mul(A.Ref("o0"), A.Ref("demand")), "~=", A.Ref("o0_rev"))),
        A.Rule("link", A.Compare(A.Div(A.Ref("o0"), A.Ref("demand")), "<=", A.Const(0))),
        A.Rule("network", A.Compare(A.Const(0), "~=", A.Agg("MIN", "all_demand"))),
    ]
    for r in rules:
        assert parse_rule_line(r.unparse()).unparse() == r.unparse()


def test_parse_is_inverse_for_temporal_conditional_and_definition_rules():
    rules = [
        A.Rule(
            "record",
            A.Compare(A.Diff(A.Ref("loss"), 1), ">", A.Const(0)),
            condition=A.Condition("label", "==", ("true_loss",)),
        ),
        A.Rule(
            "record",
            A.Compare(A.Ref("rate"), "==", A.RelatedAgg("raw_rate")),
        ),
        A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("alert"),
                A.Sustained(A.Bound(A.Ref("ratio"), "<", None), 10),
            ),
        ),
        A.Rule(
            "record",
            A.BooleanDefinition(
                A.Ref("trajectory"),
                A.Conjunction((
                    A.Bound(A.Ref("ratio"), "<", None),
                    A.Bound(A.Rolling(A.Ref("deficit"), 45, "SUM"), ">", 0.0),
                    A.Bound(A.Diff(A.Ref("ratio"), 45), "<=", 0.0),
                )),
            ),
        ),
        A.Rule(
            "record",
            A.CategoryDefinition(
                "label",
                (
                    ("is_true_loss", "true_loss"),
                    ("is_benign", "benign_burst"),
                ),
                "normal",
            ),
        ),
        A.Rule(
            "record",
            A.CategoryDefinition(
                "label",
                (
                    ("flag_true", "True"),
                    ("flag_comma", "a, b"),
                    ("flag_quote", 'x"y'),
                ),
                "normal; default=other",
            ),
        ),
        A.Rule(
            "record",
            A.BandDefinition(A.Ref("ratio"), None),
            condition=A.Condition(
                "",
                "all",
                (
                    A.Condition("archetype", "==", ("steady",)),
                    A.Condition("label", "==", ("normal",)),
                ),
            ),
        ),
        A.Rule(
            "record",
            A.BandDefinition(A.Ref("ratio"), None),
            condition=A.Condition(
                "label",
                "in",
                ("True", "a, b", 'x"y'),
            ),
        ),
    ]
    for rule in rules:
        assert parse_rule_line(rule.unparse()) == rule


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-01", tz="US/Eastern"),
        pd.Timedelta("1h"),
        np.datetime64("2026-01-01T00:00:00.123", "ms"),
        np.timedelta64(3, "h"),
    ],
)
def test_temporal_scalar_text_round_trip_preserves_ast_equality(value):
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            "target",
            (("flag", value),),
            "fallback",
        ),
        condition=A.Condition("when", "==", (value,)),
    )

    parsed = parse_rule_line(rule.unparse())

    assert parsed == rule
    assert parsed.signature() == rule.signature()


@pytest.mark.parametrize(
    "text,expected_offset",
    [
        ("2026-01-15 12:00:00", pd.Timedelta(hours=-5)),
        ("2026-07-15 12:00:00", pd.Timedelta(hours=-4)),
    ],
)
def test_dateutil_zone_scalar_json_round_trip_preserves_dst(
    text,
    expected_offset,
):
    value = pd.Timestamp(
        text,
        tz=dateutil_tz.gettz("America/New_York"),
    )

    payload = scalar_to_json(value, "value")
    restored = scalar_from_json(payload, "value")

    assert payload["timezone"] == {
        "kind": "dateutil.tzfile",
        "key": "America/New_York",
    }
    assert repr(restored) == repr(value)
    assert type(restored.tz) is dateutil_tz.tzfile
    assert restored.utcoffset() == expected_offset
    assert restored.dst() == value.dst()
    assert scalar_to_json(restored, "value") == payload
    assert {
        pd.Timestamp(date, tz=restored.tz).utcoffset()
        for date in ("2026-01-15 12:00", "2026-07-15 12:00")
    } == {
        pd.Timedelta(hours=-5),
        pd.Timedelta(hours=-4),
    }


def test_dateutil_zone_rule_text_and_dict_round_trip_preserves_signature():
    timezone = dateutil_tz.gettz("America/New_York")
    winter = pd.Timestamp("2026-01-15 12:00", tz=timezone)
    summer = pd.Timestamp("2026-07-15 12:00", tz=timezone)
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            "target",
            (
                ("winter", winter),
                ("summer", summer),
            ),
            "fallback",
        ),
        condition=A.Condition("when", "in", (winter, summer)),
    )
    payload = rule_to_dict(rule)

    text_restored = parse_rule_line(rule.unparse())
    dict_restored = rule_from_dict(payload)

    for restored in (text_restored, dict_restored):
        assert restored == rule
        assert restored.signature() == rule.signature()
        assert rule_to_dict(restored) == payload
        assert {
            value.utcoffset()
            for value in restored.condition.values
        } == {
            pd.Timedelta(hours=-5),
            pd.Timedelta(hours=-4),
        }


def test_scalar_codec_rejects_unreconstructable_dateutil_timezone():
    value = pd.Timestamp(
        "2026-01-15 12:00",
        tz=dateutil_tz.tzlocal(),
    )

    with pytest.raises(
        ValueError,
        match="unsupported timezone object.*tzlocal",
    ):
        scalar_to_json(value, "value")


@pytest.mark.parametrize(
    "value",
    [
        pd.NaT,
        np.datetime64("NaT", "us"),
        np.timedelta64("NaT", "ms"),
    ],
)
def test_temporal_nat_text_round_trip_preserves_type_and_semantics(value):
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            "target",
            (("flag", value),),
            "fallback",
        ),
        condition=A.Condition("when", "==", (value,)),
    )

    parsed = parse_rule_line(rule.unparse())
    condition_value = parsed.condition.values[0]
    category_value = parsed.atom.cases[0][1]

    assert type(condition_value) is type(value)
    assert type(category_value) is type(value)
    if isinstance(value, (np.datetime64, np.timedelta64)):
        assert condition_value.dtype == value.dtype
        assert category_value.dtype == value.dtype
        assert np.isnat(condition_value)
        assert np.isnat(category_value)
    assert (
        typed_condition_key(parsed.condition)
        == typed_condition_key(rule.condition)
    )
    assert A.category_definition_semantic_key(
        parsed.atom,
        typed_group_key,
    ) == A.category_definition_semantic_key(
        rule.atom,
        typed_group_key,
    )
    assert parsed.signature() == rule.signature()


def test_simple_condition_identifiers_keep_compact_rendering():
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("x"), "==", A.Ref("y")),
        condition=A.Condition("label_2", "==", ("normal",)),
    )

    assert rule.unparse().endswith(' where label_2 == "normal"')


@pytest.mark.parametrize(
    "column",
    [
        "a-b",
        "a == 1, b",
        'quote"name',
        r"path\name",
        "name where injected == true",
        "name in (1, 2)",
        "ALL(fake == 1)",
    ],
)
def test_condition_identifiers_are_quoted_and_round_trip(column):
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("x"), "==", A.Ref("y")),
        condition=A.Condition(column, "==", ('value", with \\ escape',)),
    )

    rendered = render_rule(rule)
    parsed = parse_rule_line(rendered)

    assert parsed == rule
    assert render_rule(parsed) == rendered


def test_condition_identifier_quoting_prevents_conjunction_injection():
    atom = A.Compare(A.Ref("x"), "==", A.Ref("y"))
    first = A.Rule(
        "record",
        atom,
        condition=A.Condition(
            "",
            "all",
            (
                A.Condition("a", "==", (1,)),
                A.Condition("b == 2, c", "==", (3,)),
            ),
        ),
    )
    second = A.Rule(
        "record",
        atom,
        condition=A.Condition(
            "",
            "all",
            (
                A.Condition("a == 1, b", "==", (2,)),
                A.Condition("c", "==", (3,)),
            ),
        ),
    )

    assert first.unparse() != second.unparse()
    assert parse_rule_line(first.unparse()) == first
    assert parse_rule_line(second.unparse()) == second


def test_all_conditions_round_trip_quoted_identifiers_and_typed_values():
    rule = A.Rule(
        "record",
        A.BandDefinition(A.Ref("ratio"), None),
        condition=A.Condition(
            "",
            "all",
            (
                A.Condition("regime-name", "in", (None, True, 1, 1.5, "1")),
                A.Condition('path\\"column', "==", ("a, b",)),
            ),
        ),
    )

    assert parse_rule_line(rule.unparse()) == rule


def test_typed_scalar_rendering_distinguishes_strings_from_primitives():
    string_rule = A.Rule(
        "record",
        A.CategoryDefinition(
            "label",
            (("flag", "True"),),
            "None",
        ),
    )
    primitive_rule = A.Rule(
        "record",
        A.CategoryDefinition(
            "label",
            (("flag", True),),
            None,
        ),
    )

    assert string_rule.signature() != primitive_rule.signature()
    assert parse_rule_line(string_rule.unparse()) == string_rule
    assert parse_rule_line(primitive_rule.unparse()) == primitive_rule


def test_category_identifiers_are_quoted_and_round_trip():
    rule = A.Rule(
        "record",
        A.CategoryDefinition(
            'label := "shadow"',
            (
                ("a->true, b", "alert"),
                ("name where injected == true", "warning"),
                ('quote"name', "normal"),
            ),
            "none",
        ),
    )

    rendered = rule.unparse()
    assert '"label := \\"shadow\\""' in rendered
    assert '"a->true, b"->"alert"' in rendered
    assert parse_rule_line(rendered) == rule


def test_dl_export_keeps_exact_coefficients_and_temporal_precedence():
    x = A.Ref("measurement_origination")
    y = A.Ref("measurement_termination")
    rules = (
        A.Rule("node", A.Compare(A.Scale(1.2345671, x), "==", y)),
        A.Rule("node", A.Compare(A.Scale(1.2345672, x), "==", y)),
        A.Rule(
            "node",
            A.Compare(
                A.Lag(A.Scale(2.0, A.Add((x, y))), 1),
                "==",
                A.Const(0.0),
            ),
        ),
        A.Rule(
            "node",
            A.Compare(
                A.Lag(A.Add((A.Scale(2.0, x), y)), 1),
                "==",
                A.Const(0.0),
            ),
        ),
    )

    def evaluation(rule):
        return SimpleNamespace(
            rule=rule,
            strictness="exact",
            eps=0.0,
            hold_rate=1.0,
            hold_rate_lo=1.0,
            hold_rate_hi=1.0,
            support=1.0,
            mdl_gain=1.0,
            parameters={},
        )

    result = SimpleNamespace(
        dataset=SimpleNamespace(
            name_model=SimpleNamespace(adapter=AD),
        ),
        portfolio=[evaluation(rule) for rule in rules],
        rounds_run=1,
        diagnostics=(),
    )
    text = portfolio_to_dl(result, "roundtrip")
    records = [
        line.split(" # ", 1)[0].rstrip()
        for line in text.splitlines()
        if line.startswith("[forall")
    ]

    assert len(records) == len(set(records)) == 4
    assert any("1.2345671*low_{X}_origination" in line for line in records)
    assert any("1.2345672*low_{X}_origination" in line for line in records)
    assert any(
        "B^1(2*(low_{X}_origination + low_{X}_termination))"
        in line
        for line in records
    )
    assert any(
        "B^1(2*low_{X}_origination + low_{X}_termination)"
        in line
        for line in records
    )


def test_dl_export_handles_temporal_condition_and_category_scalars():
    rules = (
        A.Rule(
            "record",
            A.Compare(A.Ref("x"), "==", A.Ref("y")),
            condition=A.Condition(
                "when",
                "==",
                (pd.Timestamp("2026-01-01", tz="UTC"),),
            ),
        ),
        A.Rule(
            "record",
            A.CategoryDefinition(
                "target",
                (("flag", np.timedelta64(1, "h")),),
                np.datetime64("NaT", "ns"),
            ),
        ),
    )

    def evaluation(rule):
        return SimpleNamespace(
            rule=rule,
            strictness="exact",
            eps=0.0,
            hold_rate=1.0,
            hold_rate_lo=1.0,
            hold_rate_hi=1.0,
            support=1.0,
            mdl_gain=1.0,
            parameters={},
        )

    result = SimpleNamespace(
        dataset=SimpleNamespace(name_model=None),
        portfolio=[evaluation(rule) for rule in rules],
        rounds_run=1,
        diagnostics=(),
    )

    text = portfolio_to_dl(result, "temporal")
    records = [
        line.split(" # ", 1)[0].rstrip()
        for line in text.splitlines()
        if line.startswith("[forall")
    ]
    parsed = [parse_rule_line(record) for record in records]

    assert len(records) == 2
    assert all("__autogram_temporal__" in record for record in records)
    assert parsed[0] == rules[0]
    assert parsed[1].signature() == rules[1].signature()
