"""Deployed-mode structural seeding (Fix A) + the lift-information score term (Fix B).

These pin the two fixes that turn the realistic deployed path (a leakage-free isolated subagent
that hands over only generic narrow forms plus a widening request, NOT the row-sum / col-sum /
conservation / two-end laws) from ~4/8 strict recall back up to the full form set.

* :func:`autogram.dsl.grammar.structural_invariant_seeds` deterministically enumerates the WHOLE
  structural-equality family from the grammar's role vocabulary -- the true forms AND decoys
  alike, each still data-gated -- so the I4/I5/I6/I7/I8 forms are seeded once the grammar widens,
  rather than left to vanishingly-rare blind mutation.  It is leakage-free by construction: it
  reads only ``G`` (never the ground-truth catalogue) and emits far more forms than the catalogue
  credits.
* :func:`autogram.evaluator.evaluator.lift_info` adds a clamped log-lift bonus so a genuine
  high-lift law outranks a trivially-true low-lift sign bound that would otherwise tie it.

No live API, no oracle access, no dataset read.
"""

from __future__ import annotations

import math

import pytest

from autogram.dsl.grammar import full_ceiling, narrow_grammar, structural_invariant_seeds
from autogram.dsl.typecheck import _base_family, is_admissible
from autogram.evaluator.evaluator import lift_info


# The five testable structural target FORMS, addressed by the deterministic tag the enumerator
# stamps on each candidate (built purely from role names, sorted -- never from the catalogue).
_TARGET_TAGS = {
    "I4": "pair:egress~ingress_rev",            # link two-end agreement  e[X->Y] = i[Y<-X]
    "I5": "agg:origination~sum(demand_row)",    # origination = demand row-sum
    "I6": "agg:termination~sum(demand_col)",    # termination = demand col-sum
    "I7": "flow-cons",                          # node flow conservation
    "I8": "tot:all_orig~all_term",              # network totals balance
}


def test_ceiling_seeds_cover_all_structural_targets():
    """At the public ceiling, every testable structural FORM (I4-I8) is among the seeds."""
    seeds = structural_invariant_seeds(full_ceiling())
    tags = {r.tag for r in seeds}
    for name, tag in _TARGET_TAGS.items():
        assert tag in tags, f"{name} form ({tag}) not seeded at the ceiling"


def test_ceiling_seeds_are_decoy_rich_not_the_answer_set():
    """It emits a whole FAMILY (many decoys), not just the curated catalogue -- the anti-leakage
    property: the engine cannot be accused of being handed the five answers directly."""
    seeds = structural_invariant_seeds(full_ceiling())
    tags = {r.tag for r in seeds}
    # Far more forms than the 5 credited targets ...
    assert len(seeds) > 2 * len(_TARGET_TAGS)
    # ... including the swapped flow-conservation decoy and at least one off-target pairing.
    assert "decoy:flow-swap" in tags
    decoys = tags - set(_TARGET_TAGS.values())
    assert len(decoys) >= len(_TARGET_TAGS)


def test_all_ceiling_seeds_are_admissible():
    """Every emitted candidate type-checks against the grammar it was enumerated from."""
    G = full_ceiling()
    for r in structural_invariant_seeds(G):
        ok, why = is_admissible(r, G)
        assert ok, f"{r.unparse()} inadmissible: {why}"


def test_two_end_seeds_only_cross_family():
    """Two-end equalities pair DIFFERENT base families; same-family pairs belong to the anti
    (separation) niche and must never be emitted here as an equality."""
    for r in structural_invariant_seeds(full_ceiling()):
        if r.tag.startswith("pair:"):
            a, c = r.tag[len("pair:"):].split("~")
            assert _base_family(a) != _base_family(c), f"same-family equality leaked: {r.tag}"


def test_narrow_grammar_yields_no_target_form():
    """Under the deliberately small narrow grammar (cell+node, no family roles / link / network),
    NONE of the testable structural targets can be seeded -- only harmless node decoys -- so the
    seeds matter exactly when the proposer/floor widens the grammar, never before."""
    seeds = structural_invariant_seeds(narrow_grammar())
    tags = {r.tag for r in seeds}
    assert tags.isdisjoint(set(_TARGET_TAGS.values()))
    G = narrow_grammar()
    for r in seeds:
        ok, _why = is_admissible(r, G)
        assert ok


def test_structural_seeds_read_only_the_grammar():
    """Smoke leakage check: the enumerator is a pure function of the grammar -- two calls on the
    same grammar are identical, and it never raises trying to reach any external oracle."""
    G = full_ceiling()
    a = [r.signature() for r in structural_invariant_seeds(G)]
    b = [r.signature() for r in structural_invariant_seeds(G)]
    assert a == b and len(a) == len(set(a))    # deterministic + de-duplicated


# --------------------------------------------------------------------------- Fix B: lift_info

def test_lift_info_disequality_is_zero():
    """A disequality has no band-tightness lift, so it earns no information bonus."""
    assert lift_info("!=", 190.0) == 0.0
    assert lift_info("!=", 1e12) == 0.0


def test_lift_info_unit_lift_is_neutral():
    """A trivial bound (name-blind lift ~1, i.e. the rule holds no better than a random pairing)
    earns ~0 bonus, so it cannot out-rank a genuine law on this term."""
    assert lift_info("~=", 1.0) == pytest.approx(0.0, abs=1e-9)
    assert lift_info(">=", 1.0) == pytest.approx(0.0, abs=1e-9)


def test_lift_info_is_log_lift_and_monotone():
    assert lift_info("~=", math.e) == pytest.approx(1.0, abs=1e-9)
    assert lift_info("~=", 190.0) > lift_info("~=", 10.0) > lift_info("~=", 1.0)


def test_lift_info_clamped_both_ends():
    """The exact-zero lift sentinel (~1e12) and degenerate near-zero lifts stay bounded."""
    assert lift_info("~=", 1e12) == 6.0
    assert lift_info("~=", 1e-12) == -3.0


def test_high_lift_law_outranks_trivial_bound():
    """With any positive lift weight, a high-lift law beats a unit-lift bound on this term --
    the mechanism that stops trivial sign bounds crowding genuine laws out of the portfolio."""
    w_lift = 0.15
    genuine = w_lift * lift_info("~=", 190.0)
    trivial = w_lift * lift_info(">=", 1.0)
    assert genuine > trivial + 0.5
