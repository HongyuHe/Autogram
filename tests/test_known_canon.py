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


def _bimodal_frame():
    """A member that is zero on 51% of the rows and large on the other 49%.

    Its MEDIAN absolute value is 0, so a central-statistic negligibility test judges it negligible
    and drops it -- crediting ``total == SUM(d1, bimodal)`` as recovered by a learned
    ``total == SUM(d1)`` that is violated on 49% of the rows.
    """
    names = ["total", "d1", "bimodal", "zero"]
    n = 100
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0                    # real cell
    mat[51:, 2] = 1000.0                  # zero on 51 rows, 1000 on 49 rows -> median 0
    mat[:, 0] = mat[:, 1] + mat[:, 2]     # the honest total
    return Frame(mat, names)


def test_bimodal_member_is_not_negligible_even_though_its_median_is_zero():
    f = _bimodal_frame()

    kept = _drop_negligible(
        frozenset({"d1", "bimodal", "zero"}), "total", f, zero_tol=1e-4,
    )

    assert "bimodal" in kept          # materially non-zero on 49% of rows
    assert "zero" not in kept         # identically zero everywhere
    assert kept == frozenset({"d1", "bimodal"})


def test_bimodal_member_keeps_two_sums_distinct():
    f = _bimodal_frame()
    known = ("ref_sum", ("total", frozenset({"d1", "bimodal"})))
    learned = ("ref_sum", ("total", frozenset({"d1"})))

    assert _canonicalize(known, f, 1e-4) != _canonicalize(learned, f, 1e-4)


def test_all_zero_member_is_still_dropped_when_the_anchor_has_zero_rows():
    # The intended use case must survive the pointwise test: a structurally-zero member stays
    # droppable even on data whose anchor is itself zero on some rows.
    names = ["total", "d1", "self"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[10:, 1] = 250.0
    mat[:, 0] = mat[:, 1]
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"d1", "self"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"d1"})


def test_member_larger_than_the_anchor_on_a_single_row_is_kept():
    # A member that is negligible almost everywhere but spikes on one row still changes the sum
    # there, so it cannot be canonicalized away.
    names = ["total", "d1", "spiky"]
    n = 200
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 5000.0
    mat[:, 2] = 1e-9
    mat[7, 2] = 4000.0
    mat[:, 0] = mat[:, 1] + mat[:, 2]
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"d1", "spiky"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"d1", "spiky"})
