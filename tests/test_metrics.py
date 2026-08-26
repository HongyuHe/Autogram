"""Wilson interval, MDL tie-break score, and solver-backed logic metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogram.config import DiscoveryConfig
from autogram.dsl import ast as A
from autogram.evaluator.metrics import mdl_gain, wilson
from autogram.evaluator.threshold import rule_threshold
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


def _band_threshold(tolerance, eps):
    cfg = DiscoveryConfig(
        tolerance=tolerance,
        band_mode="adaptive",
        hold_rate_threshold=0.5,
        threshold_policy="per_rule",
        thr_base_complexity=1,
        thr_separation_penalty=0.25,
        thr_min_bindings=1,
        thr_ceiling=1.0,
    )
    return rule_threshold(
        cfg,
        op="~=",
        strictness="soft",
        complexity=1,
        n_bindings=1,
        eps=eps,
    )


def test_per_rule_band_penalty_uses_tiny_positive_tolerance_cap():
    assert _band_threshold(1e-12, 1e-12) == 0.75


def test_per_rule_band_penalty_handles_zero_tolerance_cap():
    assert _band_threshold(0.0, 0.0) == 0.5


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


def test_z3_symbol_encoding_supports_typed_temporal_scalars():
    from autogram.logic.solver import _var_name

    values = (
        pd.Timestamp("1970-01-01"),
        pd.Timestamp("1970-01-01", tz="UTC"),
        pd.Timedelta(0),
        np.datetime64("1970-01-01", "ns"),
        np.timedelta64(0, "ns"),
        pd.NaT,
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
    )
    names = [_var_name(("category", value)) for value in values]

    assert names == [_var_name(("category", value)) for value in values]
    assert len(names) == len(set(names))
    assert _var_name((
        "category",
        pd.Timestamp("2026-01-01", tz="UTC"),
    )) == _var_name((
        "category",
        pd.Timestamp("2025-12-31 19:00", tz="US/Eastern"),
    ))


def test_solver_temporal_categories_use_canonical_typed_identity():
    def category(value):
        return A.Rule(
            "record",
            A.CategoryDefinition(
                "target",
                (("flag", value),),
                "none",
            ),
        )

    pandas_instant = category(pd.Timestamp("2026-01-01"))
    numpy_instant = category(np.datetime64("2026-01-01", "ns"))
    pandas_duration = category(pd.Timedelta(0))
    numpy_duration = category(np.timedelta64(0, "ns"))
    missing_timestamp = category(np.datetime64("NaT", "ns"))
    missing_duration = category(np.timedelta64("NaT", "ns"))

    assert equivalent(pandas_instant, numpy_instant)
    assert equivalent(pandas_duration, numpy_duration)
    assert equivalent(missing_timestamp, missing_duration)
    assert not equivalent(pandas_instant, pandas_duration)
    assert not subsumes(pandas_instant, pandas_duration)
    assert not subsumes(pandas_duration, pandas_instant)


def test_solver_opaque_identities_preserve_exact_term_structure():
    x = A.Ref("x")
    y = A.Ref("y")
    pairs = (
        (
            _rule(
                "record",
                A.Scale(1.2345671, x),
                "~=",
                y,
            ),
            _rule(
                "record",
                A.Scale(1.2345672, x),
                "~=",
                y,
            ),
        ),
        (
            _rule(
                "record",
                A.Lag(A.Scale(2.0, A.Add((x, y))), 1),
                ">=",
                A.Const(0.0),
            ),
            _rule(
                "record",
                A.Lag(A.Add((A.Scale(2.0, x), y)), 1),
                ">=",
                A.Const(0.0),
            ),
        ),
    )

    for left, right in pairs:
        assert left.unparse() != right.unparse()
        assert not equivalent(left, right)
        assert not subsumes(left, right)
        assert not subsumes(right, left)


def test_solver_opaque_identities_preserve_commutative_term_normalization():
    x = A.Ref("x")
    y = A.Ref("y")
    left = _rule(
        "record",
        A.Lag(A.Add((x, y)), 1),
        ">=",
        A.Const(0.0),
    )
    right = _rule(
        "record",
        A.Lag(A.Add((y, x)), 1),
        ">=",
        A.Const(0.0),
    )

    assert equivalent(left, right)
    assert subsumes(left, right)
    assert subsumes(right, left)
