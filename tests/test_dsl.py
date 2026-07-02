"""Typed DSL grounding + static admissibility on the *induced* schema (kept functionality)."""

from __future__ import annotations

import numpy as np

from autogram.dsl import ast as A
from autogram.dsl.evaluate import ground, rel_residual
from autogram.dsl.typecheck import is_admissible


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right), tag="t")


# --------------------------------------------------------------- admissibility guards

def test_self_comparison_rejected(grammar):
    r = _rule("link", A.Ref("o1"), "~=", A.Ref("o1"))
    ok, why = is_admissible(r, grammar)
    assert not ok and "self-comparison" in why


def test_shared_leaf_rejected(grammar):
    # 2*o1 ~= o1  reduces to o1 ~= 0 -> self-referential, rejected structurally
    r = _rule("link", A.Scale(2.0, A.Ref("o1")), "~=", A.Ref("o1"))
    ok, why = is_admissible(r, grammar)
    assert not ok and "reducible" in why


def test_duplicate_leaf_on_one_side_rejected(grammar):
    # o1 + -1*o1 == 0 is algebraically reducible even though the other side is a constant.
    r = _rule("link", A.Add((A.Ref("o1"), A.Scale(-1.0, A.Ref("o1")))), "==", A.Const(0))
    ok, why = is_admissible(r, grammar)
    assert not ok and "reducible" in why


def test_distinct_pair_admitted(grammar):
    r = _rule("link", A.Ref("o1"), "~=", A.Ref("o0_rev"))
    ok, _ = is_admissible(r, grammar)
    assert ok


def test_dimensionless_equality_rejected(grammar):
    r = _rule("cell", A.Ref("self"), "~=", A.Const(5))
    ok, why = is_admissible(r, grammar)
    assert not ok and "dimensionally" in why


def test_same_family_separation_admitted(grammar):
    r = _rule("link", A.Ref("o1"), "!=", A.Ref("o1_rev"))
    ok, _ = is_admissible(r, grammar)
    assert ok


def test_cross_family_separation_rejected(grammar):
    r = _rule("link", A.Ref("o1"), "!=", A.Ref("o0"))
    ok, why = is_admissible(r, grammar)
    assert not ok and "within one measurement family" in why


# ------------------------------------------------------------ grounding on real data

def test_self_demand_grounds_to_zero(dataset):
    r = _rule("node", A.Ref("demand_self"), "==", A.Const(0))
    g = ground(r, dataset.observed, dataset.name_model)
    res = rel_residual(g)
    assert res.size > 0
    assert float(np.nanmax(np.abs(res))) < 1e-9


def test_two_end_agreement_residual_small(dataset):
    # measurement_X_to_Y == measurement_Y_from_X  ==  o1 ~= o0_rev (clean on planted structure, ~noise observed)
    r = _rule("link", A.Ref("o1"), "~=", A.Ref("o0_rev"))
    g = ground(r, dataset.observed, dataset.name_model)
    res = np.abs(rel_residual(g))
    assert float(np.nanmedian(res)) < 0.05
