"""Outer-loop grammar extension + leakage-safe search feedback (the adaptive widening fix).

These tests pin the behaviour the proposer was missing: it may now return an ``extension``
that WIDENS the grammar toward a fixed ceiling, the parser clamps that widening so a backend
can never introduce vocabulary the evaluator cannot ground, and the loop threads a
leakage-safe progress signal back so the proposer knows when to widen.

No live API and no oracle access: the LLM is stood in for by an in-process proposer that reads
only the leakage-safe ``ProposalContext`` (its ceiling + feedback), exactly as a real backend
would.
"""

from __future__ import annotations

import json

import pytest

from autogram.config import RunConfig
from autogram.dsl.grammar import default_grammar, full_ceiling, narrow_grammar
from autogram.proposer.base import (Proposal, Proposer, SearchFeedback, assert_no_leakage,
                                    build_context, extension_block, parse_grammar_extension,
                                    parse_proposal_json, render_proposal_prompt)
from autogram.search.loop import apply_extension, ceiling_grammar, learn, start_grammar


# --------------------------------------------------------------- extension parser clamping

def test_parse_grammar_extension_clamps_to_ceiling():
    G = narrow_grammar()                 # cell/node only, caps (8, 2)
    ceiling = full_ceiling()             # all vocabulary, caps (12, 3)
    ext_obj = {
        "enable_binders": ["link", "teleport", "cell"],   # link OK; teleport unknown; cell on
        "enable_fam_roles": ["all_orig"],
        "enable_ops": ["<="],
        "max_complexity": 999,                            # must clamp down to the ceiling
        "max_add_arity": 1,                               # below current (2) -> ignored
        "note": "widen",
    }
    ext = parse_grammar_extension(ext_obj, G, ceiling)
    assert ext is not None
    assert ext.enable_binders == ("link",)               # unknown + already-on dropped
    assert ext.enable_fam_roles == ("all_orig",)
    assert ext.enable_ops == ("<=",)
    assert ext.max_complexity == 12                       # clamped to the ceiling, not 999
    assert ext.max_add_arity is None                      # a cap may only ever be RAISED


def test_parse_grammar_extension_noop_at_ceiling_returns_none():
    ceiling = full_ceiling()
    # Already at the ceiling: every enable is redundant, so the widening is a pure no-op.
    ext = parse_grammar_extension(
        {"enable_binders": ["link", "network"], "max_complexity": 12}, ceiling, ceiling)
    assert ext is None


def test_parse_grammar_extension_ignores_non_dict():
    assert parse_grammar_extension(None, narrow_grammar(), full_ceiling()) is None
    assert parse_grammar_extension("widen please", narrow_grammar(), full_ceiling()) is None


# ----------------------------------------------------------- extension via the JSON contract

def test_parse_proposal_json_attaches_extension():
    G = narrow_grammar()
    ceiling = full_ceiling()
    payload = json.dumps({
        "rules": [],
        "notes": "widen first",
        "extension": {"enable_binders": ["link"], "enable_ref_roles": ["egress"]},
    })
    prop = parse_proposal_json(payload, G, ceiling)
    assert prop.extension is not None
    assert prop.extension.enable_binders == ("link",)
    assert prop.extension.enable_ref_roles == ("egress",)


def test_parse_proposal_json_two_arg_default_ceiling_caps_at_grammar():
    # The backward-compatible 2-argument call defaults the ceiling to G's own maximal
    # vocabulary, so an extension can still re-enable vocabulary but cannot raise the caps.
    G = narrow_grammar()                 # caps (8, 2)
    payload = json.dumps({
        "rules": [],
        "extension": {"enable_binders": ["link"], "max_complexity": 999},
    })
    prop = parse_proposal_json(payload, G)               # no explicit ceiling
    assert prop.extension is not None
    assert prop.extension.enable_binders == ("link",)    # widening of vocabulary is fine
    # ... but the complexity cap cannot exceed G's own (8), so no real raise survives.
    assert (prop.extension.max_complexity or G.max_complexity) <= G.max_complexity


def test_parse_proposal_json_without_extension_is_none():
    G = default_grammar()
    payload = json.dumps({"rules": [], "notes": "forms only"})
    prop = parse_proposal_json(payload, G, full_ceiling())
    assert prop.extension is None


# ------------------------------------------------------------------ disclosure + feedback

def test_extension_block_lists_available_then_empty_at_ceiling():
    avail = extension_block(narrow_grammar(), full_ceiling())
    assert "link" in avail and "network" in avail         # widening targets disclosed
    none_left = extension_block(full_ceiling(), full_ceiling())
    assert "no widening available" in none_left


def test_search_feedback_is_leakage_safe():
    fb = SearchFeedback(
        round_index=1, n_accepted=7, best_score=2.31, filled_cells=5,
        binders_covered=("cell", "node"), binders_idle=("link", "network"),
        roles_used=4, improved=False, stagnant=True,
        hint="some enabled binders have no accepted rule; widen toward them",
    )
    text = fb.as_prompt_text()                            # raises LeakageError if it leaks
    assert_no_leakage(text, "feedback")
    assert "link" in text and "stagnant" in text


# --------------------------------------------------------------- end-to-end widening loop

class _WideningProposer(Proposer):
    """A stand-in proposer that widens ``G`` toward whatever ``ctx.ceiling`` offers.

    It reads ONLY the leakage-safe context (its ceiling and feedback), proving the outer-loop
    plumbing: the ceiling reaches the backend, a proposed ``extension`` is honoured, and the
    feedback signal is delivered from the second round on.
    It carries no catalogue knowledge -- it just re-enables vocabulary the ceiling already
    permits, which is exactly the bounded widening the trusted parser allows.
    """

    name = "widening-test"

    def __init__(self) -> None:
        self.feedback_seen: list = []

    def propose(self, ctx) -> Proposal:
        self.feedback_seen.append(ctx.feedback)
        G, ceiling = ctx.grammar, ctx.ceiling
        ext = {
            "enable_binders": [b for b in ceiling.binders if b not in G.binders],
            "enable_ref_roles": [r for r in ceiling.ref_roles if r not in G.ref_roles],
            "enable_fam_roles": [r for r in ceiling.fam_roles if r not in G.fam_roles],
            "enable_ops": [o for o in ceiling.ops if o not in G.ops],
            "max_complexity": ceiling.max_complexity,
            "note": "widen toward ceiling",
        }
        payload = json.dumps({"rules": [], "notes": "widen", "extension": ext})
        return parse_proposal_json(payload, G, ceiling)


def _fast_config() -> RunConfig:
    rc = RunConfig(dataset="abilene")
    rc.grammar.start = "narrow"
    rc.grammar.rounds = 2
    rc.search.iterations = 30          # tiny: the test asserts widening, not full recall
    rc.search.islands = 2
    rc.search.bootstrap_random = 6
    rc.seed = 0
    rc.reseed()
    return rc


def test_narrow_start_widens_grammar_and_records_extension(abilene):
    rc = _fast_config()
    proposer = _WideningProposer()
    res = learn(abilene, rc, proposer)

    # The grammar started narrow but the proposer widened it to the full ceiling vocabulary.
    started = start_grammar(rc.grammar)
    ceiling = ceiling_grammar(rc.grammar)
    assert "link" not in started.binders                 # narrow start really was restricted
    assert set(res.grammar.binders) == set(ceiling.binders)
    assert "link" in res.grammar.binders and "network" in res.grammar.binders

    # The widening was recorded for the run bundle, attributed to the round it happened in.
    assert res.extensions_applied, "expected at least one recorded grammar extension"
    first = res.extensions_applied[0]
    assert first["round"] == 0
    assert "link" in first["enabled_binders"]

    # Feedback discipline: round 0 sees None, round 1 sees a real leakage-safe signal.
    assert proposer.feedback_seen[0] is None
    assert isinstance(proposer.feedback_seen[1], SearchFeedback)
    assert_no_leakage(proposer.feedback_seen[1].as_prompt_text(), "round-1 feedback")


def test_full_start_applies_no_extension(abilene):
    # The default start == ceiling, so even an eager widening proposer is a guaranteed no-op
    # and the historical behaviour is preserved.
    rc = RunConfig(dataset="abilene")
    rc.grammar.rounds = 2
    rc.search.iterations = 20
    rc.search.islands = 2
    rc.seed = 0
    rc.reseed()
    res = learn(abilene, rc, _WideningProposer())
    assert res.extensions_applied == []
    assert set(res.grammar.binders) == set(start_grammar(rc.grammar).binders)


def test_apply_extension_unions_vocabulary():
    G = narrow_grammar()
    ext = parse_grammar_extension(
        {"enable_binders": ["link"], "enable_ref_roles": ["egress", "ingress_rev"]},
        G, full_ceiling())
    G2 = apply_extension(G, ext)
    assert "link" in G2.binders
    assert "egress" in G2.ref_roles and "ingress_rev" in G2.ref_roles
    # widening is monotone: nothing the narrow grammar had is lost.
    assert set(G.binders).issubset(set(G2.binders))


def test_widening_proposer_prompt_is_leakage_free(abilene):
    # The proposer's actual prompt (vocabulary + extension block + context) must not leak.
    G = narrow_grammar()
    ctx = build_context(abilene, G, ceiling=full_ceiling())
    prompt = render_proposal_prompt(ctx)                 # raises if it leaks
    assert "AVAILABLE TO ENABLE" in prompt
    assert "I5" not in prompt and "abilene_geant_invariants" not in prompt
