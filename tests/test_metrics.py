"""Name-permutation lift percentile, Wilson interval, MDL gain (the data-only metrics)."""

from __future__ import annotations

import numpy as np

from autogram.dsl import ast as A
from autogram.evaluator.metrics import mdl_gain, name_blind_null, wilson


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right))


def test_genuine_pairing_lifts(dataset):
    # two-end agreement: true pairing far below the within-kind permutation null
    r = _rule("link", A.Ref("o1"), "~=", A.Ref("o0_rev"))
    res = name_blind_null(r, dataset.observed, dataset.name_model, n_perm=12, seed=0)
    assert res.lift > 5.0
    assert res.percentile <= 0.05


def test_spurious_pairing_does_not_lift(dataset):
    # meas_X_to_Y vs meas_X_from_Y are unrelated -> no lift, high percentile
    r = _rule("link", A.Ref("o1"), "~=", A.Ref("o0"))
    res = name_blind_null(r, dataset.observed, dataset.name_model, n_perm=12, seed=0)
    assert res.percentile > 0.05


def test_wilson_interval_brackets_phat():
    lo, hi, phat = wilson(95, 100)
    assert lo < phat < hi and abs(phat - 0.95) < 1e-9


def test_mdl_gain_prefers_tight_residual():
    tight = mdl_gain(_rule("cell", A.Ref("self"), "~=", A.Const(0)), 0.01,
                     np.full(200, 0.01))
    loose = mdl_gain(_rule("cell", A.Ref("self"), "~=", A.Const(0)), 0.5,
                     np.full(200, 0.5))
    assert tight > loose
