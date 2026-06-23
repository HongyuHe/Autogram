"""DSL evaluation + static admissibility (design Sec. 6.5)."""

from __future__ import annotations

import numpy as np

from autogram.dsl import ast as A
from autogram.dsl.evaluate import ground, rel_residual
from autogram.dsl.grammar import default_grammar
from autogram.dsl.typecheck import is_admissible


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right), tag="t")


# --------------------------------------------------------------------- admissibility guards

def test_self_comparison_rejected():
    G = default_grammar()
    r = _rule("link", A.Ref("egress"), "~=", A.Ref("egress"))
    ok, why = is_admissible(r, G)
    assert not ok and "self-comparison" in why


def test_cross_family_separation_rejected():
    G = default_grammar()
    # egress vs ingress are different measurement families -> trivially separated.
    r = _rule("link", A.Ref("egress"), "!=", A.Ref("ingress"))
    ok, why = is_admissible(r, G)
    assert not ok and "within one measurement family" in why


def test_same_family_separation_admitted():
    G = default_grammar()
    r = _rule("link", A.Ref("egress"), "!=", A.Ref("egress_rev"))
    ok, _ = is_admissible(r, G)
    assert ok


def test_dimensionless_equality_rejected():
    G = default_grammar()
    # comparing a measured byte volume to a bare non-zero constant is meaningless.
    r = _rule("cell", A.Ref("self"), "~=", A.Const(5))
    ok, why = is_admissible(r, G)
    assert not ok and "dimensionally" in why


def test_nonnegativity_admitted():
    G = default_grammar()
    r = _rule("cell", A.Ref("self"), ">=", A.Const(0))
    ok, _ = is_admissible(r, G)
    assert ok


def test_complexity_cap_enforced():
    G = default_grammar()
    assert G.max_complexity >= 10  # must admit the Kirchhoff node-flow law (I7)


# ------------------------------------------------------------------ evaluation on real data

def test_self_demand_is_zero(abilene):
    """I2: H[X,X] grounds to an all-zero residual under the node binder."""
    r = _rule("node", A.Ref("demand_self"), "==", A.Const(0))
    g = ground(r, abilene.observed, abilene.name_model)
    res = rel_residual(g)
    assert res.size > 0
    assert float(np.nanmax(np.abs(res))) < 1e-9


def test_link_two_end_agreement_residual_small(abilene):
    """I4: e[X->Y] vs i[Y<-X] grounds to a near-zero relative residual on clean structure."""
    r = _rule("link", A.Ref("egress"), "~=", A.Ref("ingress_rev"))
    g = ground(r, abilene.observed, abilene.name_model)
    res = np.abs(rel_residual(g))
    # most points agree to within a few percent (the injected noise is ~2%).
    assert float(np.nanmedian(res)) < 0.05
