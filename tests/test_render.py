"""Explicit (schema-aware) rendering + its inverse parser -- fast, no live model."""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.dsl.render import parse_rule_line, render_rule
from autogram.schema.spec import FamilySelector


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
