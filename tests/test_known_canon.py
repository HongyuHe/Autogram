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


def test_individually_tiny_members_are_not_dropped_when_they_add_up():
    """Round-29 review: a per-member bound does not bound the sum of the members removed.

    Six hundred members each under the tolerance contributed 6% of the total between them, which is
    outside any plausible acceptance band -- yet all of them were canonicalized away.
    """
    n_members = 600
    names = ["total", "real", *[f"m{index}" for index in range(n_members)]]
    n = 30
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 940.0                    # a materially large member, so the group is never emptied
    for index in range(n_members):
        mat[:, 2 + index] = 0.1          # 1e-4 of the anchor each, 6% together
    mat[:, 0] = 1000.0
    f = Frame(mat, names)
    members = frozenset(names[1:])

    kept = _drop_negligible(members, "total", f, zero_tol=1e-4)

    # Aggregate contribution is material, so nothing may be canonicalized away -- and the result
    # must not depend on the "never empty a group" guard rescuing it.
    assert kept == members


def test_exactly_zero_members_are_still_dropped_alongside_material_ones():
    # The fallback must keep the intended use case working: a structurally-zero member is removable
    # even when other individually-tiny members are not.
    names = ["total", "real", "tiny0", "tiny1", "zero"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 900.0
    mat[:, 2] = 0.09                     # individually negligible ...
    mat[:, 3] = 0.09                     # ... but 0.18 together, over the 0.1 budget
    mat[:, 4] = 0.0
    mat[:, 0] = 1000.0
    f = Frame(mat, names)

    kept = _drop_negligible(
        frozenset({"real", "tiny0", "tiny1", "zero"}), "total", f, zero_tol=1e-4,
    )

    assert "zero" not in kept
    assert kept == frozenset({"real", "tiny0", "tiny1"})


def test_member_nonzero_only_where_the_sum_is_ungradeable_is_dropped():
    """A member that never affects a gradeable row must not split an alias pair.

    The sum is undefined wherever any member is missing, so a member that is non-zero only on those
    rows provably never changes the relation.
    """
    names = ["total", "w", "z"]
    n = 50
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[20:, 1] = np.nan                 # `w` missing on the tail -> the sum is ungradeable there
    mat[20:, 2] = 900.0                  # `z` is large only where the sum cannot be graded
    mat[:, 0] = 500.0
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"w", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"w"})


def test_canonicalize_is_idempotent():
    for f, sig in (
        (_frame(), ("ref_sum", ("orig", frozenset({"d1", "d2", "self", "dead"})))),
        (_bimodal_frame(), ("ref_sum", ("total", frozenset({"d1", "bimodal", "zero"})))),
    ):
        once = _canonicalize(sig, f, 1e-4)
        assert _canonicalize(once, f, 1e-4) == once

    names = ["total", "real", "tiny0", "tiny1", "zero"]
    n = 30
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 900.0
    mat[:, 2] = 0.09
    mat[:, 3] = 0.09
    mat[:, 0] = 1000.0
    f = Frame(mat, names)
    sig = ("ref_sum", ("total", frozenset({"real", "tiny0", "tiny1", "zero"})))
    once = _canonicalize(sig, f, 1e-4)
    assert _canonicalize(once, f, 1e-4) == once


def test_removal_may_not_widen_the_graded_population():
    """Round-30 review: a member's own missingness restricts the sum's domain.

    ``z`` is 0 on ten rows and missing on the other ninety, so ``total == SUM(real, z)`` is graded
    on ten rows only. Dropping ``z`` produced ``total == SUM(real)``, which is graded on all one
    hundred -- and fails on ninety of them. Canonicalisation may not hand the reduced relation rows
    the original never had to satisfy.
    """
    names = ["total", "real", "z"]
    n = 100
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[:, 0] = 500.0
    mat[10:, 0] = 999.0          # `total == SUM(real)` is false on the last ninety rows
    mat[:, 2] = np.nan
    mat[:10, 2] = 0.0            # `z` is defined (and zero) only on the first ten rows
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"real", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"real", "z"})
    assert _canonicalize(("ref_sum", ("total", frozenset({"real", "z"}))), f, 1e-4) != (
        "ref_sum", ("total", frozenset({"real"}))
    )


def test_all_defined_zero_member_is_still_dropped():
    # The domain-preserving requirement must not break the intended case: a member defined on every
    # row where the anchor is defined, and zero throughout, is still removable.
    names = ["total", "real", "z"]
    n = 60
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 500.0
    mat[:, 0] = 500.0
    mat[:, 2] = 0.0
    mat[40:, 0] = np.nan         # the anchor itself is missing on the tail; `z` still qualifies
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"real", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"real"})


def test_stable_row_sum_is_order_independent_and_exact():
    """Round-30 review: the aggregate bound must not depend on hash-randomised iteration order.

    Floating-point addition is not associative, so accumulating per-column magnitudes in
    ``frozenset``/``dict`` order makes the result depend on string hash randomisation -- and with it
    which members are canonicalized away, and therefore which side of the held-out boundary a
    catalogue entry lands on. The summation must be both deterministically ordered and exact.

    Tested at this seam rather than through ``_drop_negligible``: a whole-function test cannot force
    an adversarial iteration order, so it passes whether or not the summation is ordered, which is
    no test at all (round-32 review).
    """
    from autogram.discovery.known import _stable_row_sum

    columns = {
        "a": np.array([0.03630875, 1e16]),
        "b": np.array([0.04805217, 1.0]),
        "c": np.array([0.04447561, -1e16]),
        "d": np.array([0.02008216, 1.0]),
    }

    reference = _stable_row_sum(columns)
    for order in (("d", "c", "b", "a"), ("b", "a", "d", "c"), ("c", "a", "d", "b")):
        shuffled = {name: columns[name] for name in order}
        assert _stable_row_sum(shuffled).tobytes() == reference.tobytes()

    # Exact, not merely reproducible: naive accumulation of the second row loses the two ones.
    assert reference[1] == 2.0
    assert float(np.stack([columns[name] for name in ("a", "b", "c", "d")], axis=0).sum(axis=0)[1]) != 2.0


def test_member_missing_only_where_a_retained_member_is_also_missing_is_dropped():
    """Round-31 review: domain preservation must compare masks, not each member to the anchor.

    ``z`` is missing exactly where ``w`` is missing, so the sum is ungradeable on those rows either
    way and removing ``z`` changes nothing. Requiring ``z`` to be finite wherever the *anchor* is
    kept it, which split two identically-evaluated sums across the held-out boundary.
    """
    names = ["total", "w", "z"]
    n = 50
    mat = np.zeros((n, len(names)))
    mat[:, 0] = 500.0
    mat[:, 1] = 500.0
    mat[30:, 1] = np.nan          # `w` missing on the tail -> the sum is ungradeable there
    mat[:, 2] = 0.0
    mat[30:, 2] = np.nan          # `z` missing on exactly the same rows
    f = Frame(mat, names)

    kept = _drop_negligible(frozenset({"w", "z"}), "total", f, zero_tol=1e-4)

    assert kept == frozenset({"w"})


def test_stable_row_sum_reports_an_unrepresentable_total_as_infinite():
    """An aggregate beyond float64 is past any finite budget; it must not crash the run."""
    from autogram.discovery.known import _stable_row_sum

    columns = {
        f"m{index}": np.array([1.5e308, 1.0])
        for index in range(8)
    }

    total = _stable_row_sum(columns)

    assert np.isinf(total[0])
    assert total[1] == 8.0


def test_exact_relation_may_only_drop_identically_zero_members():
    """Round-33 review: an exact relation has no tolerance to spend.

    A member that is merely small still breaks ``total == SUM(...)`` on every row it is non-zero, so
    crediting a learned exact sum with recovering a known exact sum that omits it reports a law the
    data does not satisfy anywhere.
    """
    names = ["total", "a", "z"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0
    mat[:, 2] = 0.05                 # 5e-5 of the anchor: under zero_tol, but not zero
    mat[:, 0] = 1000.05
    f = Frame(mat, names)

    known = ("equality", "exact", ("ref_sum", ("total", frozenset({"a", "z"}))))
    learned = ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))
    assert _canonicalize(known, f, 1e-4) != _canonicalize(learned, f, 1e-4)

    # An approximate relation may still absorb it -- that is what its tolerance is for.
    known_approx = ("equality", "approximate", ("ref_sum", ("total", frozenset({"a", "z"}))))
    learned_approx = ("equality", "approximate", ("ref_sum", ("total", frozenset({"a"}))))
    assert _canonicalize(known_approx, f, 1e-4) == _canonicalize(learned_approx, f, 1e-4)


def test_exact_relation_still_drops_a_structurally_zero_member():
    names = ["total", "a", "z"]
    n = 40
    mat = np.zeros((n, len(names)))
    mat[:, 1] = 1000.0
    mat[:, 2] = 0.0
    mat[:, 0] = 1000.0
    f = Frame(mat, names)

    known = ("equality", "exact", ("ref_sum", ("total", frozenset({"a", "z"}))))
    learned = ("equality", "exact", ("ref_sum", ("total", frozenset({"a"}))))

    assert _canonicalize(known, f, 1e-4) == _canonicalize(learned, f, 1e-4)
