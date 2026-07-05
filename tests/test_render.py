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
