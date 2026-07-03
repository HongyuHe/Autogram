"""Item 6: multiplication and division (bounded-degree, proposer opt-in)."""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.dsl.grammar import Grammar
from autogram.dsl.parser import rule_to_dict, rule_from_dict
from autogram.dsl.typecheck import is_admissible
from autogram.discovery.propose import EnumerationProposer
from autogram.logic.solver import atom_expr, is_trivial


def _grammar(max_degree):
    return Grammar(
        binders=("node",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"node": ("a", "b", "c")},
        fam_roles={"node": ()},
        agg_kinds=("SUM",),
        scale_coeffs=(-1.0,),
        max_complexity=12,
        max_add_arity=2,
        max_degree=max_degree,
    )


def _text(max_degree):
    return "\n".join(r.unparse() for r in EnumerationProposer(_grammar(max_degree)).propose(0, (), None))


def test_linear_default_has_no_products_or_ratios():
    t = _text(1)
    assert " * " not in t   # Mul renders "(a * b)"; Scale renders "-1*a" (no spaces)
    assert " / " not in t


def test_degree2_emits_products_and_ratios():
    t = _text(2)
    assert " * " in t, "no product enumerated"
    assert " / " in t, "no ratio enumerated"


def test_degree_cap_is_respected():
    for r in EnumerationProposer(_grammar(2)).propose(0, (), None):
        assert max(r.atom.left.degree(), r.atom.right.degree()) <= 2


def test_degree3_product_is_inadmissible_at_cap2():
    ab_c = A.Mul(A.Mul(A.Ref("a"), A.Ref("b")), A.Ref("c"))   # degree 3
    rule = A.Rule("node", A.Compare(ab_c, "~=", A.Const(0.0)))
    ok, _ = is_admissible(rule, _grammar(2))
    assert not ok


def test_parser_round_trips_mul_and_div():
    for term in (A.Mul(A.Ref("a"), A.Ref("b")), A.Div(A.Ref("a"), A.Ref("b"))):
        rule = A.Rule("node", A.Compare(term, "~=", A.Ref("c")))
        assert rule_from_dict(rule_to_dict(rule)).unparse() == rule.unparse()


def test_solver_builds_mul_and_div_atoms():
    mul = A.Rule("node", A.Compare(A.Mul(A.Ref("a"), A.Ref("b")), "~=", A.Ref("c")))
    div = A.Rule("node", A.Compare(A.Div(A.Ref("a"), A.Ref("b")), "~=", A.Ref("c")))
    assert atom_expr(mul.atom, {}) is not None
    assert atom_expr(div.atom, {}) is not None
    assert is_trivial(mul) is False and is_trivial(div) is False
