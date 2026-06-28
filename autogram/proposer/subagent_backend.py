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
  externally-spawned subagent that only received the prompt file.  For multi-round (outer-loop)
  runs an externally-spawned harness may answer each round distinctly via
  ``<work_dir>/subagent_response_<dataset>_round<k>.json`` (k = 1, 2, ...); when a per-round
  file is absent the backend falls back to the base file, so a single static reply still works.
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
        self.last_notes = ""                 # proposer feedback (admitted/rejected forms)
        os.makedirs(work_dir, exist_ok=True)

    def prompt_path(self, round_idx: Optional[int] = None) -> str:
        if round_idx:
            return os.path.join(
                self.work_dir, f"subagent_prompt_{self.dataset}_round{round_idx}.txt")
        return os.path.join(self.work_dir, f"subagent_prompt_{self.dataset}.txt")

    def response_path(self, round_idx: Optional[int] = None) -> str:
        if round_idx:
            return os.path.join(
                self.work_dir, f"subagent_response_{self.dataset}_round{round_idx}.json")
        return os.path.join(self.work_dir, f"subagent_response_{self.dataset}.json")

    def propose(self, ctx: ProposalContext) -> Proposal:
        round_idx = getattr(ctx, "round_index", 0) or 0
        prompt = render_proposal_prompt(ctx)        # leakage-checked
        # Always refresh the base prompt (back-compat / single-shot harness); for round >= 1
        # also drop a per-round prompt so an external harness can answer each round distinctly.
        with open(self.prompt_path(), "w", encoding="ascii", errors="replace") as fh:
            fh.write(prompt)
        if round_idx:
            with open(self.prompt_path(round_idx), "w",
                      encoding="ascii", errors="replace") as fh:
                fh.write(prompt)

        text: Optional[str] = None
        source = "fallback"
        if self.responder is not None:
            text = self.responder(prompt)
            source = "responder-callback"
        else:
            # Prefer a per-round reply; fall back to the base static reply when absent so a
            # single committed response still drives every round (the practical default).
            rp = self.response_path(round_idx)
            if round_idx and not os.path.exists(rp):
                rp = self.response_path()
            if os.path.exists(rp):
                with open(rp, "r", encoding="utf-8") as fh:
                    text = fh.read()
                source = f"response-file:round{round_idx}" if round_idx else "response-file"

        if not text:
            self.last_notes = ("subagent: no reply; engine continues via blind random "
                               "search (no grammar fallback)")
            return Proposal(seeds=[], notes=self.last_notes)
        assert_no_leakage(text, "subagent response")  # defensive re-scan
        prop = parse_proposal_json(text, ctx.grammar, ctx.ceiling)
        self.used_real_subagent = True
        detail = f" | {prop.notes}" if prop.notes else ""
        prop.notes = f"subagent[{source}] -> {len(prop.seeds)} admissible forms{detail}"
        self.last_notes = prop.notes
        return prop
