"""Graceful-degradation widening floor + per-round subagent replies (the deployed-narrow fix).

These tests pin the behaviour that makes the *only practical* path -- a static, file-backed
subagent reply under a ``narrow`` start in ``deployed`` mode -- reach the full form set instead
of starving at the two narrow-grammar forms (the 25%-recall regression).

Two independent mechanisms are exercised, with NO live API and NO oracle access:

* the engine-side deterministic floor (:func:`autogram.search.loop._deterministic_widen`) widens
  ``G`` to the PUBLIC ceiling when the proposer supplies no usable extension, so a single static
  reply that pre-dates the search feedback still drives the outer loop;
* admissibility is re-checked against the *current* grammar each round, so the link/network forms
  in that static reply -- dropped while ``G`` is narrow -- are admitted once the floor widens.

A stand-in proposer returns a FIXED rule set (mimicking the committed
``subagent_response_<dataset>.json``); it reads only the leakage-safe ``ProposalContext``.
"""

from __future__ import annotations

import json
import os

from autogram.config import RunConfig
from autogram.dsl.grammar import full_ceiling, narrow_grammar
from autogram.proposer.base import Proposal, Proposer, build_context, parse_proposal_json
from autogram.proposer.subagent_backend import SubagentProposer
from autogram.search.loop import (_deterministic_widen, ceiling_grammar, learn,
                                   start_grammar)


# A static reply with three forms whose admissibility depends on the grammar width:
#   I1  cell self >= 0            -- admissible under the narrow grammar
#   I2  node demand_self == 0     -- admissible under the narrow grammar
#   I4  link egress ~= ingress_rev-- needs the link binder + egress/ingress_rev roles (ceiling)
_STATIC_REPLY = json.dumps({
    "rules": [
        {"binder": "cell", "op": ">=",
         "left": {"k": "Ref", "role": "self"}, "right": {"k": "Const", "value": 0}},
        {"binder": "node", "op": "==",
         "left": {"k": "Ref", "role": "demand_self"}, "right": {"k": "Const", "value": 0}},
        {"binder": "link", "op": "~=",
         "left": {"k": "Ref", "role": "egress"}, "right": {"k": "Ref", "role": "ingress_rev"}},
    ],
    "notes": "static structural forms (no extension)",
})


# --------------------------------------------------------------- the deterministic floor unit

def test_deterministic_widen_from_narrow_reaches_ceiling():
    narrow, ceiling = narrow_grammar(), full_ceiling()
    ext = _deterministic_widen(narrow, ceiling)
    assert ext is not None
    # it enables exactly the vocabulary the ceiling has but the narrow grammar lacks
    assert "link" in ext.enable_binders and "network" in ext.enable_binders
    assert "egress" in ext.enable_ref_roles and "ingress_rev" in ext.enable_ref_roles
    assert ext.max_complexity == ceiling.max_complexity
    assert ext.max_add_arity == ceiling.max_add_arity


def test_deterministic_widen_at_ceiling_is_none():
    ceiling = full_ceiling()
    # start == ceiling (the default "full" case) -> guaranteed no-op, so the floor never fires.
    assert _deterministic_widen(ceiling, ceiling) is None


# ----------------------------------------------------------- end-to-end graceful degradation

class _StaticReplyProposer(Proposer):
    """Mimics the file-backed subagent: returns the SAME static reply (no extension) each round.

    It re-parses that reply against ``ctx.grammar`` exactly as the real backend does, so the
    per-round admissibility filtering is faithfully reproduced.  It records how many forms were
    admissible each round, which is what proves the floor widened the space between rounds.
    """

    name = "static-test"

    def __init__(self) -> None:
        self.admitted_per_round: list = []
        self.grammars_seen: list = []

    def propose(self, ctx) -> Proposal:
        self.grammars_seen.append(ctx.grammar)
        prop = parse_proposal_json(_STATIC_REPLY, ctx.grammar, ctx.ceiling)  # no "extension"
        assert prop.extension is None
        self.admitted_per_round.append(len(prop.seeds))
        return prop


def _fast_narrow_config(auto_widen: bool = True) -> RunConfig:
    rc = RunConfig(dataset="abilene")
    rc.grammar.start = "narrow"
    rc.grammar.rounds = 2
    rc.grammar.auto_widen = auto_widen
    rc.search.iterations = 30          # tiny: the test asserts widening, not full recall
    rc.search.islands = 2
    rc.search.bootstrap_random = 6
    rc.seed = 0
    rc.reseed()
    return rc


def test_narrow_static_reply_auto_widens_and_admits_more_forms(abilene):
    rc = _fast_narrow_config(auto_widen=True)
    proposer = _StaticReplyProposer()
    res = learn(abilene, rc, proposer)

    # The grammar started narrow but the deterministic floor widened it to the full ceiling.
    assert "link" not in start_grammar(rc.grammar).binders
    assert set(res.grammar.binders) == set(ceiling_grammar(rc.grammar).binders)

    # The widening is recorded and ATTRIBUTED to the engine, not a proposer extension.
    assert res.extensions_applied, "expected the deterministic floor to record a widening"
    det = [e for e in res.extensions_applied if e.get("source") == "deterministic"]
    assert det and det[0]["round"] == 0
    assert "link" in det[0]["enabled_binders"]

    # The heart of the fix: the SAME static reply admits strictly more forms after widening.
    # Round 0 (narrow) admits only the two narrow forms; round 1 (widened) also admits the link
    # form that was dropped before.
    assert proposer.admitted_per_round[0] == 2
    assert proposer.admitted_per_round[1] > proposer.admitted_per_round[0]


def test_auto_widen_disabled_stays_narrow(abilene):
    # Gate check: with the knob off, a no-extension static reply can no longer widen, so the
    # grammar stays narrow and the run starves -- demonstrating the floor is what cures it.
    rc = _fast_narrow_config(auto_widen=False)
    proposer = _StaticReplyProposer()
    res = learn(abilene, rc, proposer)
    assert "link" not in res.grammar.binders
    assert not [e for e in res.extensions_applied if e.get("source") == "deterministic"]
    assert all(n == 2 for n in proposer.admitted_per_round)   # never grew past the narrow forms


def test_full_start_floor_is_noop(abilene):
    # start == ceiling: the floor must be a guaranteed no-op so headline runs are unchanged.
    rc = RunConfig(dataset="abilene")
    rc.grammar.rounds = 2
    rc.grammar.auto_widen = True
    rc.search.iterations = 20
    rc.search.islands = 2
    rc.seed = 0
    rc.reseed()
    res = learn(abilene, rc, _StaticReplyProposer())
    assert res.extensions_applied == []


# ------------------------------------------------------------ per-round response resolution

def test_subagent_per_round_response_file_resolution(tmp_path, abilene):
    work = str(tmp_path)
    base = os.path.join(work, "subagent_response_abilene.json")
    rnd1 = os.path.join(work, "subagent_response_abilene_round1.json")
    with open(base, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"rules": [
            {"binder": "cell", "op": ">=",
             "left": {"k": "Ref", "role": "self"}, "right": {"k": "Const", "value": 0}}],
            "notes": "BASE-REPLY"}))
    with open(rnd1, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"rules": [
            {"binder": "node", "op": "==",
             "left": {"k": "Ref", "role": "demand_self"}, "right": {"k": "Const", "value": 0}}],
            "notes": "ROUND1-REPLY"}))

    G = narrow_grammar()
    sub = SubagentProposer(work_dir=work, dataset="abilene")

    # round 0 -> base file
    p0 = sub.propose(build_context(abilene, G, ceiling=full_ceiling(), round_index=0))
    assert "BASE-REPLY" in p0.notes and "response-file" in p0.notes
    # round 1 -> the per-round file is preferred
    p1 = sub.propose(build_context(abilene, G, ceiling=full_ceiling(), round_index=1))
    assert "ROUND1-REPLY" in p1.notes and "round1" in p1.notes
    # round 2 -> no per-round file, so it falls back to the base reply (single static reply works)
    p2 = sub.propose(build_context(abilene, G, ceiling=full_ceiling(), round_index=2))
    assert "BASE-REPLY" in p2.notes
