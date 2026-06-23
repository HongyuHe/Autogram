"""Deterministic scripted proposer (the reproducible default + test baseline).

This backend uses *no* LLM.  It returns the grammar's combinatorial seed templates
(``enumerate_candidates``) as candidate forms.  Those templates are produced purely from
the role vocabulary plus a handful of planted decoys -- they are **leakage-free by
construction** (they never read the ground-truth catalog).  Genuine learning is still
required downstream: the evaluator must separate the true forms from the decoys and assign
the correct strictness (exact / soft-structural / anti) *from data*, and the search must
mutate these seeds to explore neighbours.

It is the deterministic baseline used by the tests and by the reproducible scripted run;
the headline validation run uses the isolated subagent backend instead.
"""

from __future__ import annotations

from .base import Proposal, Proposer, ProposalContext
from ..dsl.grammar import enumerate_candidates


class ScriptedProposer(Proposer):
    name = "scripted"

    def propose(self, ctx: ProposalContext) -> Proposal:
        seeds = list(enumerate_candidates(ctx.grammar))
        return Proposal(seeds=seeds, extension=None,
                        notes="combinatorial name-semantic seed templates (no LLM)")
