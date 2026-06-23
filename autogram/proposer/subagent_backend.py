"""Isolated-subagent proposer backend (the second swappable LLM backend).

This backend is the one used for the **leakage-free headline validation**.  It hands an
*isolated* subagent (e.g. one spawned by a Claude Code harness) **only** the leakage-safe
prompt produced from :func:`render_proposal_prompt` -- column names/types, the current
grammar vocabulary, the node count and a few *observed* (noisy) sample rows.  The subagent
must never see ``docs/abilene_geant_invariants.md`` or any derived oracle.

Isolation is enforced/documented at four layers:

1. **Construction.**  The backend is built from the leakage-safe :class:`ProposalContext`
   only; it has no handle to the clean frame or the catalog.
2. **Prompt.**  The prompt is passed through :func:`assert_no_leakage` before it is written
   to disk, and instructs the subagent to use only the in-prompt context (no file reads).
3. **Response.**  Every returned proposal is re-scanned with :func:`assert_no_leakage`; any
   oracle token (``I1``..``I10``, the catalog filename, ``ground truth`` ...) raises.
4. **Judged from data.**  Even a leaked *form* earns the right verdict only if the evaluator
   confirms it against data and fits the right band, so recovery still reflects learning.

Three ways to supply the subagent's reply (checked in order):

* ``responder`` callback  -- ``responder(prompt:str) -> str`` (used by tests / programmatic
  spawning; the harness can pass a closure that spawns a real isolated subagent).
* response file          -- ``<work_dir>/subagent_response_<dataset>.json`` written by an
  externally-spawned subagent that only received the prompt file.
* fallback               -- empty proposal (so the engine still runs; the run report records
  that no real subagent reply was used).  Never falls back to the grammar's true-form
  templates.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from .base import (Proposal, Proposer, ProposalContext, assert_no_leakage,
                   parse_proposal_json, render_proposal_prompt)


class SubagentProposer(Proposer):
    name = "subagent"

    def __init__(self, work_dir: str = ".", dataset: str = "dataset",
                 responder: Optional[Callable[[str], str]] = None) -> None:
        self.work_dir = work_dir
        self.dataset = dataset
        self.responder = responder
        self.used_real_subagent = False     # surfaced in the run report for honesty
        os.makedirs(work_dir, exist_ok=True)

    def prompt_path(self) -> str:
        return os.path.join(self.work_dir, f"subagent_prompt_{self.dataset}.txt")

    def response_path(self) -> str:
        return os.path.join(self.work_dir, f"subagent_response_{self.dataset}.json")

    def propose(self, ctx: ProposalContext) -> Proposal:
        prompt = render_proposal_prompt(ctx)        # leakage-checked
        with open(self.prompt_path(), "w", encoding="ascii", errors="replace") as fh:
            fh.write(prompt)

        text: Optional[str] = None
        source = "fallback"
        if self.responder is not None:
            text = self.responder(prompt)
            source = "responder-callback"
        elif os.path.exists(self.response_path()):
            with open(self.response_path(), "r", encoding="utf-8") as fh:
                text = fh.read()
            source = "response-file"

        if not text:
            return Proposal(seeds=[], notes="subagent: no reply; engine continues via "
                                            "blind random search (no grammar fallback)")
        assert_no_leakage(text, "subagent response")  # defensive re-scan
        prop = parse_proposal_json(text, ctx.grammar)
        self.used_real_subagent = True
        prop.notes = f"subagent[{source}] -> {len(prop.seeds)} admissible forms"
        return prop
