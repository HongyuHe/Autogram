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
from ..dsl.grammar import (Grammar, anti_invariant_seeds, default_grammar,
                           enumerate_candidates, full_ceiling, narrow_grammar,
                           structural_invariant_seeds)
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
    trace: List[dict] = field(default_factory=list)
    extensions_applied: List[dict] = field(default_factory=list)


def _trace_row(round_idx: int, phase: str, it: int, island: int, origin: str,
               parent_sig, improved: bool, res) -> dict:
    """One per-candidate telemetry record (design: poc-eval recommendation 1).

    ``res`` is ``None`` when the chosen variation produced no admissible child (so nothing
    was evaluated); the row still records the attempt for search-dynamics auditing.
    """
    row = {
        "round": round_idx,
        "phase": phase,            # "seed" | "search"
        "iter": it,                # within-round middle-loop index; -1 for the seed phase
        "island": island,
        "origin": origin,
        "parent_sig": parent_sig,
        "improved": bool(improved),
    }
    if res is None:
        row.update({
            "sig": None, "rule": None, "binder": None, "op": None, "tag": None,
            "complexity": None, "verdict": "INADMISSIBLE", "accepted": False,
            "combined_score": None, "eps": None, "kappa_hat": None, "support": None,
            "lift": None, "delta": None,
        })
        return row
    r = res.rule
    row.update({
        "sig": r.signature(),
        "rule": r.unparse(),
        "binder": r.binder,
        "op": r.atom.op,
        "tag": r.tag,
        "complexity": r.complexity(),
        "verdict": res.verdict.value,
        "accepted": bool(res.accepted),
        "combined_score": round(res.combined_score, 6),
        "eps": round(res.eps, 6),
        "kappa_hat": round(res.kappa_hat, 6),
        "support": round(res.support, 6),
        "lift": round(res.lift, 6),
        "delta": round(res.delta, 6),
    })
    return row


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


def ceiling_grammar(gc) -> Grammar:
    """The maximal grammar a proposer may widen ``G`` toward (the outer-loop boundary)."""
    return full_ceiling(gc.max_complexity_ceiling, gc.max_add_arity_ceiling)


def start_grammar(gc) -> Grammar:
    """The grammar the search STARTS from, selected by ``grammar.start``.

    ``"full"`` (default) returns the maximal vocabulary capped at ``grammar.max_complexity`` --
    identical to the historical behaviour, so with ``start == ceiling`` the outer-loop widening
    is a guaranteed no-op and existing runs are byte-for-byte unchanged.
    ``"narrow"`` returns a deliberately restricted grammar so the proposer must widen ``G``
    toward the ceiling before the link/network laws become reachable.
    """
    if getattr(gc, "start", "full") == "narrow":
        return narrow_grammar()
    G = default_grammar()
    return Grammar(binders=G.binders, ops=G.ops, ref_roles=G.ref_roles,
                   fam_roles=G.fam_roles, agg_kinds=G.agg_kinds,
                   max_complexity=gc.max_complexity, max_add_arity=G.max_add_arity)


def _rule_roles(rule: Rule) -> set:
    """Distinct ref/family roles appearing in a rule (for leakage-safe feedback only)."""
    roles: set = set()

    def walk(t) -> None:
        cn = type(t).__name__
        if cn == "Ref":
            roles.add(t.role)
        elif cn == "Agg":
            roles.add(t.family_role)
        elif cn == "Scale":
            walk(t.term)
        elif cn == "Add":
            for x in t.terms:
                walk(x)

    walk(rule.atom.left)
    walk(rule.atom.right)
    return roles


def _grammar_changed(before: Grammar, after: Grammar) -> bool:
    return (before.binders != after.binders or before.ops != after.ops
            or before.ref_roles != after.ref_roles or before.fam_roles != after.fam_roles
            or before.agg_kinds != after.agg_kinds
            or before.max_complexity != after.max_complexity
            or before.max_add_arity != after.max_add_arity)


def _extension_record(round_idx: int, ext, before: Grammar, after: Grammar,
                      source: str = "proposer") -> dict:
    """A machine-readable summary of what a widening actually added to ``G``.

    ``source`` records who drove the widening: ``"proposer"`` (an LLM/subagent extension) or
    ``"deterministic"`` (the engine-side graceful-degradation floor, see
    :func:`_deterministic_widen`).
    """
    return {
        "round": round_idx,
        "source": source,
        "note": getattr(ext, "note", "") or "",
        "enabled_binders": [b for b in after.binders if b not in before.binders],
        "enabled_ref_roles": [r for r in after.ref_roles if r not in before.ref_roles],
        "enabled_fam_roles": [r for r in after.fam_roles if r not in before.fam_roles],
        "enabled_ops": [o for o in after.ops if o not in before.ops],
        "enabled_agg_kinds": [a for a in after.agg_kinds if a not in before.agg_kinds],
        "max_complexity": after.max_complexity,
        "max_add_arity": after.max_add_arity,
    }


def _deterministic_widen(G: Grammar, ceiling: Grammar):
    """Engine-side graceful-degradation widening (Sec. 10.4 outer loop).

    When the proposer supplies no usable extension -- e.g. the *only practical* file-backed
    subagent path replays a single static reply that pre-dates the search feedback, so it can
    never carry an ``extension`` -- the outer loop would otherwise never fire and a ``narrow``
    start would starve (only the two narrow-grammar forms survive admissibility).  This widens
    ``G`` directly to the PUBLIC ``ceiling`` (the typed vocabulary derived from column *roles*,
    never the ground-truth catalog), so the mechanism is exercised end-to-end and the engine
    degrades gracefully into the full search space.

    Returns a :class:`~..proposer.base.GrammarExtension` enabling exactly the ceiling vocabulary
    not yet in ``G`` (and raising the size caps to the ceiling), or ``None`` if ``G`` is already
    at the ceiling -- which is the default ``start = "full"`` case, making this a guaranteed
    no-op there.  Leakage-safe by construction: it only enables names already disclosed to the
    proposer under 'AVAILABLE TO ENABLE'.
    """
    from ..proposer.base import GrammarExtension
    add_binders = tuple(b for b in ceiling.binders if b not in G.binders)
    add_ops = tuple(o for o in ceiling.ops if o not in G.ops)
    add_refs = tuple(r for r in ceiling.ref_roles if r not in G.ref_roles)
    add_fams = tuple(r for r in ceiling.fam_roles if r not in G.fam_roles)
    add_aggs = tuple(a for a in ceiling.agg_kinds if a not in G.agg_kinds)
    raise_cx = ceiling.max_complexity > G.max_complexity
    raise_arity = ceiling.max_add_arity > G.max_add_arity
    if not (add_binders or add_ops or add_refs or add_fams or add_aggs
            or raise_cx or raise_arity):
        return None
    return GrammarExtension(
        enable_ref_roles=add_refs, enable_fam_roles=add_fams, enable_ops=add_ops,
        enable_binders=add_binders, enable_agg_kinds=add_aggs,
        max_complexity=ceiling.max_complexity if raise_cx else None,
        max_add_arity=ceiling.max_add_arity if raise_arity else None,
        note="deterministic widening to public ceiling (proposer supplied no extension)",
    )


def _search_feedback(round_idx: int, islands: Islands, G: Grammar, prev_best: float):
    """Mine a leakage-safe :class:`SearchFeedback` from the engine's OWN elites.

    Everything here is computed from accepted elites and the enabled vocabulary -- never the
    ground-truth catalog or the clean frame -- so handing it to the proposer is legitimate
    search feedback, not oracle access.
    """
    from ..proposer.base import SearchFeedback
    elites = islands.all_elites()
    best_score = max((e.combined_score for e in elites), default=0.0)
    covered = sorted({e.rule.binder for e in elites})
    idle = tuple(b for b in G.binders if b not in set(covered))
    roles: set = set()
    for e in elites:
        roles |= _rule_roles(e.rule)
    improved = best_score > prev_best + 1e-9
    stagnant = (not improved) or bool(idle)
    if idle:
        hint = "some enabled binders have no accepted rule; widen toward them or enable a new one"
    elif not improved:
        hint = "no score improvement this round; consider widening the grammar"
    else:
        hint = "search still improving inside the current grammar"
    return SearchFeedback(
        round_index=round_idx, n_accepted=len(elites), best_score=best_score,
        filled_cells=len({e.descriptor for e in elites}),
        binders_covered=tuple(covered), binders_idle=idle, roles_used=len(roles),
        improved=improved, stagnant=stagnant, hint=hint,
    )


def learn(ds: Dataset, rc: RunConfig, proposer: "Proposer") -> RunResult:
    """Run the full nested loop and return the assembled invariant portfolio."""
    rng = random.Random(rc.seed)
    # Start from the configured starting grammar; the proposer may widen it toward the
    # ceiling across rounds (outer loop).  In the default ``start = "full"`` the start
    # already equals the ceiling, so widening is a guaranteed no-op.
    G = start_grammar(rc.grammar)
    ceiling = ceiling_grammar(rc.grammar)

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
    seed_origin: Dict[str, str] = {}          # signature -> provenance label (for the trace)
    trace: List[dict] = []
    extensions_applied: List[dict] = []
    feedback = None                           # leakage-safe progress; None on the first round
    prev_best = 0.0

    # ---- OUTER: grammar/LLM proposer rounds -------------------------------
    from ..proposer.base import build_context
    for round_idx in range(max(1, rc.grammar.rounds)):
        # Hand the proposer the widen-toward ceiling and (from round 1 on) the search
        # feedback, so it can decide whether to EXPAND the grammar or keep proposing forms
        # inside it (Sec. 10.4).  Both are leakage-safe by construction.
        ctx = build_context(ds, G, feedback=feedback, ceiling=ceiling, round_index=round_idx)
        proposal = proposer.propose(ctx)
        G_before = G
        G = apply_extension(G, proposal.extension)
        proposer_widened = proposal.extension is not None and _grammar_changed(G_before, G)
        if proposer_widened:
            extensions_applied.append(
                _extension_record(round_idx, proposal.extension, G_before, G,
                                  source="proposer"))
        # Graceful-degradation floor (Sec. 10.4): if the proposer did NOT widen the grammar
        # and G is still below the public ceiling, the engine widens to the ceiling itself so
        # the outer loop fires even on the static file-backed subagent path (the only practical
        # mode).  Leakage-safe (ceiling = public typed vocabulary) and a guaranteed no-op when
        # start == ceiling (default "full"), so headline runs are byte-for-byte unchanged.
        if rc.grammar.auto_widen and not proposer_widened:
            det = _deterministic_widen(G, ceiling)
            if det is not None:
                G_det = apply_extension(G, det)
                if _grammar_changed(G, G_det):
                    extensions_applied.append(
                        _extension_record(round_idx, det, G, G_det, source="deterministic"))
                    G = G_det
        # Seeds come from the PROPOSER.  ``enumerate_candidates`` (the combinatorial
        # name-semantic templates) is injected here ONLY if explicitly enabled -- this is
        # the leakage-fairness control: for the isolated subagent run it stays off so the
        # search is seeded purely by what the subagent proposed (plus blind random search),
        # never by the engine's own template enumerator.
        #
        # ``anti_invariant_seeds`` is injected independently (knob ``search.anti_seeds``):
        # it covers ONLY the narrow same-family separation niche (a != a_rev) that blind
        # search under-samples (one operator, restricted to same-family Ref leaves).  It is
        # leakage-free for the same reason as ``enumerate_candidates`` -- it enumerates the
        # whole separation family from G's vocabulary, not the catalog's answer, and each
        # form still has to earn an ANTI verdict from the data.  Unlike ``seed_from_grammar``
        # it does NOT inject the equality templates, so the proposer-isolation test stands.
        tagged = [(r, "proposer") for r in proposal.seeds]
        if rc.search.seed_from_grammar:
            tagged += [(r, "grammar_seed") for r in enumerate_candidates(G)]
        if rc.search.anti_seeds:
            tagged += [(r, "anti_seed") for r in anti_invariant_seeds(G)]
        # ``structural_invariant_seeds`` (knob ``search.structural_seeds``) is the equality
        # analog of the anti niche: it injects the whole aggregate-equality family (counter-
        # vs-sum, two-end, totals, balance) enumerated from G's vocabulary -- true forms and
        # decoys alike -- because that niche needs an exact multi-term pairing blind search
        # almost never hits and an isolated leakage-free proposer rarely supplies (deployed
        # eval finding).  Leakage-free for the same reason as the anti seed: it never reads the
        # catalog, is decoy-rich, and every form still earns its verdict from the data.  Under
        # a narrow start it is a no-op (the link/family roles it needs are not yet enabled), so
        # it only takes effect once the proposer/widening floor opens the grammar.
        if rc.search.structural_seeds:
            tagged += [(r, "structural_seed") for r in structural_invariant_seeds(G)]
        have = {s.signature() for s in seed_pool}
        for r, origin in tagged:
            ok, _ = is_admissible(r, G)
            sig = r.signature()
            if ok and sig not in have:
                seed_pool.append(r)
                have.add(sig)
                seed_origin[sig] = origin

        # Blind random bootstrap so the middle loop always has diverse parents to mutate,
        # even when the proposal is thin (leakage-free: random_rule never reads the oracle).
        for _ in range(max(0, rc.search.bootstrap_random)):
            r = random_rule(G, rng)
            if r is not None and r.signature() not in have:
                seed_pool.append(r)
                have.add(r.signature())
                seed_origin[r.signature()] = "random_bootstrap"

        # seed the islands round-robin, evaluating each candidate once
        for i, r in enumerate(seed_pool):
            res = ev(r)
            arm = i % rc.search.islands
            improved = islands.add(arm, res)
            trace.append(_trace_row(round_idx, "seed", -1, arm,
                                    seed_origin.get(r.signature(), "seed"),
                                    None, improved, res))

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
                origin, parent_sig = "mutation", parent.signature()
            elif per_island_seeds[arm] and rng.random() < 0.5:
                parent = rng.choice(per_island_seeds[arm])
                child = mutate_rule(parent, G, rng)
                origin, parent_sig = "seed_mutation", parent.signature()
            else:
                child = random_rule(G, rng)
                origin, parent_sig = "random", None
            if child is None:
                alloc.update(arm, 0.0)
                trace.append(_trace_row(round_idx, "search", it, arm, origin,
                                        parent_sig, False, None))
                continue
            res = ev(child)
            improved = islands.add(arm, res)
            alloc.update(arm, 1.0 if improved else 0.0)
            trace.append(_trace_row(round_idx, "search", it, arm, origin,
                                    parent_sig, improved, res))
            if rc.search.migration_every and (it + 1) % rc.search.migration_every == 0:
                islands.migrate()

        # ---- end of round: refresh the leakage-safe feedback for the NEXT round -----
        feedback = _search_feedback(round_idx, islands, G, prev_best)
        prev_best = feedback.best_score

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
        n_unique=len(cache), island_rates=alloc.rates(), trace=trace,
        extensions_applied=extensions_applied,
    )
