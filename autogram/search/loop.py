"""The three-level nested learning loop for the P3 engine (design Sec. 10.4).

* **OUTER (grammar / LLM):** a proposer widens the grammar ``G`` and injects candidate
  *forms* (never thresholds).  Runs at low frequency (``grammar.rounds``).
* **MIDDLE (evolutionary):** a MAP-Elites quality-diversity search over ``islands`` with
  Thompson-sampled budget allocation explores ``G`` for high-scoring, *diverse* rules.
* **INNER (analytic):** the soft band epsilon is *fit in closed form per candidate by the
  evaluator* (Sec. 5.4) -- there is no search over thresholds, which is the primary
  anti-blow-up lever.

The loop never reads the ground-truth catalog; the proposer only sees a leakage-safe
context (Sec. 6.4) and the evaluator's clean-frame access is confined to the noise gate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from ..config import RunConfig
from ..loader.loader import Dataset
from ..dsl.ast import Rule
from ..dsl.grammar import Grammar, default_grammar, enumerate_candidates
from ..dsl.typecheck import is_admissible
from ..evaluator.evaluator import EvaluationResult, evaluate
from .archive import Islands
from .assemble import assemble
from .mutate import mutate_rule, random_rule
from .thompson import ThompsonAllocator

if TYPE_CHECKING:
    from ..proposer.base import Proposer


@dataclass
class RunResult:
    dataset: str
    portfolio: List[EvaluationResult]
    accepted: List[EvaluationResult]
    grammar: Grammar
    proposer_name: str
    n_evaluated: int
    n_unique: int
    island_rates: List[float] = field(default_factory=list)


def apply_extension(G: Grammar, ext) -> Grammar:
    """Union a :class:`GrammarExtension` onto ``G`` (outer-loop widening)."""
    if ext is None:
        return G
    def u(cur, add):
        return tuple(dict.fromkeys(list(cur) + list(add)))
    return Grammar(
        binders=u(G.binders, ext.enable_binders),
        ops=u(G.ops, ext.enable_ops),
        ref_roles=u(G.ref_roles, ext.enable_ref_roles),
        fam_roles=u(G.fam_roles, ext.enable_fam_roles),
        agg_kinds=u(G.agg_kinds, ext.enable_agg_kinds),
        max_complexity=max(G.max_complexity, ext.max_complexity or 0),
        max_add_arity=max(G.max_add_arity, ext.max_add_arity or 0),
        extensions=G.extensions,
    )


def learn(ds: Dataset, rc: RunConfig, proposer: "Proposer") -> RunResult:
    """Run the full nested loop and return the assembled invariant portfolio."""
    rng = random.Random(rc.seed)
    # Start from a grammar capped at the configured complexity (the proposer may widen it).
    G = default_grammar()
    G = Grammar(binders=G.binders, ops=G.ops, ref_roles=G.ref_roles,
                fam_roles=G.fam_roles, agg_kinds=G.agg_kinds,
                max_complexity=rc.grammar.max_complexity,
                max_add_arity=G.max_add_arity)

    cache: Dict[str, EvaluationResult] = {}

    def ev(rule: Rule) -> EvaluationResult:
        sig = rule.signature()
        hit = cache.get(sig)
        if hit is None:
            hit = evaluate(rule, ds, rc.eval)
            cache[sig] = hit
        return hit

    islands = Islands(rc.search.islands)
    alloc = ThompsonAllocator(rc.search.islands, rng=random.Random(rc.seed + 7))
    seed_pool: List[Rule] = []

    # ---- OUTER: grammar/LLM proposer rounds -------------------------------
    from ..proposer.base import build_context
    for _ in range(max(1, rc.grammar.rounds)):
        ctx = build_context(ds, G)
        proposal = proposer.propose(ctx)
        G = apply_extension(G, proposal.extension)
        # Seeds come from the PROPOSER.  ``enumerate_candidates`` (the combinatorial
        # name-semantic templates) is injected here ONLY if explicitly enabled -- this is
        # the leakage-fairness control: for the isolated subagent run it stays off so the
        # search is seeded purely by what the subagent proposed (plus blind random search),
        # never by the engine's own template enumerator.
        forms = list(proposal.seeds)
        if rc.search.seed_from_grammar:
            forms += enumerate_candidates(G)
        for r in forms:
            ok, _ = is_admissible(r, G)
            if ok and r.signature() not in {s.signature() for s in seed_pool}:
                seed_pool.append(r)

        # Blind random bootstrap so the middle loop always has diverse parents to mutate,
        # even when the proposal is thin (leakage-free: random_rule never reads the oracle).
        seen = {s.signature() for s in seed_pool}
        for _ in range(max(0, rc.search.bootstrap_random)):
            r = random_rule(G, rng)
            if r is not None and r.signature() not in seen:
                seed_pool.append(r)
                seen.add(r.signature())

        # seed the islands round-robin, evaluating each candidate once
        for i, r in enumerate(seed_pool):
            res = ev(r)
            islands.add(i % rc.search.islands, res)

        # ---- MIDDLE: evolutionary MAP-Elites + islands + Thompson ---------
        per_island_seeds: List[List[Rule]] = [[] for _ in range(rc.search.islands)]
        for i, r in enumerate(seed_pool):
            per_island_seeds[i % rc.search.islands].append(r)

        for it in range(rc.search.iterations):
            arm = alloc.select() if rc.search.thompson else it % rc.search.islands
            elites = islands.archives[arm].elites()
            if elites and rng.random() < rc.search.p_mutate:
                parent = rng.choice(elites).rule
                child = mutate_rule(parent, G, rng)
            elif per_island_seeds[arm] and rng.random() < 0.5:
                parent = rng.choice(per_island_seeds[arm])
                child = mutate_rule(parent, G, rng)
            else:
                child = random_rule(G, rng)
            if child is None:
                alloc.update(arm, 0.0)
                continue
            res = ev(child)
            improved = islands.add(arm, res)
            alloc.update(arm, 1.0 if improved else 0.0)
            if rc.search.migration_every and (it + 1) % rc.search.migration_every == 0:
                islands.migrate()

    # ---- assemble the final portfolio -------------------------------------
    # Assemble over the FULL accepted pool (design Sec. 10.3 mine-and-cover), not just the
    # MAP-Elites archive snapshot: the archive is the quality-diversity *driver* of the
    # evolutionary search, but a one-sided bound can out-score and evict the stronger
    # two-sided equality from a shared descriptor cell.  Assembling over all accepted rules
    # lets op-variant canonicalisation (assemble step 1b) recover the canonical equality.
    accepted = [r for r in cache.values() if r.accepted]
    portfolio = assemble(accepted, ds, rc.eval,
                         dedup_rel=rc.search.dedup_rel,
                         k_max=rc.search.assemble_k_max)
    return RunResult(
        dataset=ds.name, portfolio=portfolio, accepted=accepted, grammar=G,
        proposer_name=proposer.name, n_evaluated=len(cache),
        n_unique=len(cache), island_rates=alloc.rates(),
    )
