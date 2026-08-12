"""Wilson interval, MDL tie-break score, and solver-backed logic metrics."""

from __future__ import annotations

import numpy as np

from autogram.dsl import ast as A
from autogram.evaluator.metrics import mdl_gain, wilson
from autogram.logic.solver import equivalent, is_tautology, subsumes


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right))


def test_wilson_interval_brackets_phat():
    lo, hi, phat = wilson(95, 100)
    assert lo < phat < hi and abs(phat - 0.95) < 1e-9


def test_mdl_gain_prefers_tight_residual_as_tiebreak():
    tight = mdl_gain(_rule("cell", A.Ref("self"), "~=", A.Const(0)), 0.01, np.full(200, 0.01))
    loose = mdl_gain(_rule("cell", A.Ref("self"), "~=", A.Const(0)), 0.5, np.full(200, 0.5))
    assert tight > loose


def test_z3_solver_checks_tautology_equivalence_subsumption():
    ge = _rule("node", A.Ref("measurement_source"), ">=", A.Const(0))
    le = _rule("node", A.Const(0), "<=", A.Ref("measurement_source"))
    eq = _rule("node", A.Ref("measurement_source"), "==", A.Const(0))
    taut = _rule("node", A.Ref("measurement_source"), ">=", A.Ref("measurement_source"))
    assert is_tautology(taut)
    assert equivalent(ge, le)
    assert subsumes(eq, ge)
    assert not subsumes(ge, eq)


def test_z3_symbol_encoding_is_injective_across_similar_role_names():
    # Roles that differ only by a non-alphanumeric character (hyphen vs underscore) must map to
    # distinct Z3 symbols; a non-injective encoding would conflate them and wrongly report the two
    # equalities as logically equivalent.
    left = _rule("node", A.Ref("a-b"), "==", A.Ref("c"))
    right = _rule("node", A.Ref("a_b"), "==", A.Ref("c"))

    assert not equivalent(left, right)
    assert not subsumes(left, right)
    assert not subsumes(right, left)


def test_z3_symbol_encoding_is_injective_for_control_and_separator_chars():
    from autogram.logic.solver import _var_name

    keys = [
        ("ref", "a\x1fb"),
        ("ref", "a", "b"),
        ("ref", "a_b"),
        ("proportional_coefficient", "x", "y"),
        ("proportional_coefficient", "x_y"),
    ]
    names = [_var_name(k) for k in keys]
    assert len(set(names)) == len(names)
