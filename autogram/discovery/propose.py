"""Proposals: random schema-typed generation + mutation inside the induced grammar (P4).

There is no scripted backend, no seed enumeration and no planted decoys.  The deterministic
offline baseline is *random within the induced schema* -- it carries no catalogue knowledge and
can only ever form rules the induced ontology grounds.  An optional LLM proposer plugs into the
same interface for the deployed path.

Proposals are drawn by:

* **mutation** of an existing elite (the archive's own rules), giving the search a hill-climb
  signal from its own discoveries; and
* **fresh random** admissible rules, giving it exploration.

Every candidate is re-checked with :func:`typecheck.is_admissible`, so it is always a
well-formed, terminating invariant in the induced schema.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from ..dsl import ast as A
from ..dsl.grammar import Grammar
from ..dsl.typecheck import is_admissible
from ..search.mutate import mutate_rule, random_rule


class RandomProposer:
    """Deterministic, schema-typed proposer (the offline reference baseline)."""

    def __init__(self, G: Grammar, p_mutate: float = 0.7):
        self.G = G
        self.p_mutate = p_mutate

    def propose(self, n: int, seeds: Sequence[A.Rule], rng: random.Random) -> List[A.Rule]:
        out: List[A.Rule] = []
        seen = set()
        seeds = list(seeds)
        attempts = 0
        while len(out) < n and attempts < n * 12:
            attempts += 1
            cand: Optional[A.Rule] = None
            if seeds and rng.random() < self.p_mutate:
                parent = rng.choice(seeds)
                cand = mutate_rule(parent, self.G, rng)
            if cand is None:
                cand = random_rule(self.G, rng)
            if cand is None:
                continue
            ok, _ = is_admissible(cand, self.G)
            if not ok:
                continue
            sig = cand.signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cand)
        return out


class PortfolioProposer:
    """Round-robin proposer portfolio.

    Each child proposer sees the same own-elite seeds and gets a chance to propose candidates.
    The portfolio then deduplicates and interleaves their outputs.  With an LLM child plus the
    random child, selecting the portfolio genuinely invokes the LLM seam while the random child
    preserves deterministic offline fallback and mutation-based own-elite feedback.
    """

    def __init__(self, proposers: Sequence[object]):
        self.proposers = list(proposers)
        self.G = next((getattr(p, "G") for p in self.proposers if hasattr(p, "G")), None)

    def propose(self, n: int, seeds: Sequence[A.Rule], rng: random.Random) -> List[A.Rule]:
        batches = [list(p.propose(n, seeds, rng)) for p in self.proposers]
        out: List[A.Rule] = []
        seen = set()
        more = True
        i = 0
        while len(out) < n and more:
            more = False
            for batch in batches:
                if i >= len(batch):
                    continue
                more = True
                cand = batch[i]
                if self.G is not None:
                    ok, _ = is_admissible(cand, self.G)
                    if not ok:
                        continue
                sig = cand.signature()
                if sig in seen:
                    continue
                seen.add(sig)
                out.append(cand)
                if len(out) >= n:
                    break
            i += 1
        return out


class LLMProposer:  # pragma: no cover
    """Deployed proposer: an LLM emits candidate rules (parsed to the DSL).

    Injected ``responder`` maps a leakage-safe prompt to a list of rule dicts.  Falls back to no
    proposals if absent; the loop then relies on the random proposer.
    """

    def __init__(self, G: Grammar, responder=None):
        self.G = G
        self.responder = responder

    def propose(self, n, seeds, rng) -> List[A.Rule]:
        if self.responder is None:
            return []
        import json

        from ..dsl.parser import rule_from_dict
        raw = self.responder(_proposer_prompt(self.G, seeds, n))
        out = []
        for d in json.loads(raw):
            try:
                r = rule_from_dict(d)
            except Exception:
                continue
            ok, _ = is_admissible(r, self.G)
            if ok:
                out.append(r)
        return out


def _proposer_prompt(G: Grammar, seeds, n) -> str:  # pragma: no cover
    return (f"Propose up to {n} candidate invariants as rule JSON over binders {G.binders} "
            f"with roles {G.ref_roles} and families {G.fam_roles}. Avoid tautologies.")
