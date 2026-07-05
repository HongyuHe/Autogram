"""Fast, offline tests for scoreboard canonicalization of near-zero sum groupings."""

from __future__ import annotations

import numpy as np

from autogram.loader.loader import Frame
from autogram.discovery.known import _canonicalize, _drop_negligible


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
