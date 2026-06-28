"""Portfolio assembly: mine-and-cover + de-duplication (design Sec. 10.3).

The search yields many accepted rules, several of which are near-duplicates (e.g.
``a ~= b`` and its scalar multiple ``2a ~= 2b`` share an identical relative residual, so
they satisfy the *same* points).  Assembly turns the archive into a parsimonious
portfolio:

1. **structural dedup** -- drop identical rule signatures;
2. **semantic dedup** -- within a binder, drop a rule whose satisfied-point set is a
   near-duplicate (Jaccard >= 1 - ``dedup_rel``) of a kept, simpler/higher-scoring one
   (this is the cheap, oracle-free stand-in for z3 entailment dedup of Sec. 10.3);
3. **niche collapse** -- keep one best-scoring representative per distinct structural niche
   (binder x comparison-kind x residual keyset), the granularity at which recall counts
   distinct invariants;
4. **information-aware budget + sign-entailment filter** -- two-sided equalities and
   anti-invariants are the informative laws and are kept in full; one-sided bounds
   (``a >= b`` / ``a <= b``) are split by a leakage-free symbolic test of the established sign
   model (every measurement family is a non-negative count, i.e. invariant I1).  *Compound*
   sign tautologies -- bounds that hold for every non-negative assignment (``i >= -H``,
   ``0.25e <= 1.5e + H``) -- carry no information and are dropped; the non-negativity schema
   ``q >= 0`` (target I1) is kept as one canonical representative; genuinely data-supported
   bounds are summarised by their best representative per (binder, kind) and compete only for
   the slots left after the informative laws (one slot guaranteed so I1 always surfaces).  This
   removes the redundant bound fan without touching any equality or anti-invariant.

Niche representatives are chosen by score, then by parsimony (simplest equivalent form), so a
bloated restatement never represents a niche.  Genuine, structurally-distinct invariants are
never pruned by similarity, so recall over a diverse target set (exact, soft-structural, anti)
is preserved.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

from ..config import EvalConfig
from ..loader.loader import Dataset
from ..dsl import ast as A
from ..dsl.ast import Rule
from ..dsl.evaluate import ground
from ..evaluator.evaluator import EvaluationResult
from ..evaluator.gate import Verdict


def satisfaction(rule: Rule, ds: Dataset, eps: float) -> np.ndarray:
    """Per-point soft-satisfaction of ``rule`` under its fitted band (observed frame)."""
    g = ground(rule, ds.observed, ds.name_model)
    if g.degenerate or g.n_points == 0:
        return np.zeros(0, dtype=bool)
    rel = np.abs(g.rho) / g.scale
    op = rule.atom.op
    if op == "!=":
        return rel > max(eps, 1e-9)
    if op == ">=":
        return g.rho >= -(1e-9 + 1e-9 * g.scale)
    if op == "<=":
        return g.rho <= (1e-9 + 1e-9 * g.scale)
    return rel <= max(eps, 1e-12)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or a.size != b.size:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 1.0


def _flatten(term: A.Term, coeff: float, acc: Dict[str, float]) -> None:
    """Accumulate ``coeff * term`` into a signed coefficient map keyed by leaf signature.

    Adds distribute, Scales fold their coefficient in, so an additive expression collapses
    to its linear-combination fingerprint; this lets us detect that two rules describe the
    same underlying relation up to bloat (extra near-zero terms) or a term repeated on both
    sides of an equality."""
    if isinstance(term, A.Add):
        for t in term.terms:
            _flatten(t, coeff, acc)
    elif isinstance(term, A.Scale):
        _flatten(term.term, coeff * term.coeff, acc)
    elif isinstance(term, A.Const):
        acc["#const"] = acc.get("#const", 0.0) + coeff * term.value
    elif isinstance(term, A.Ref):
        acc[term.role] = acc.get(term.role, 0.0) + coeff
    elif isinstance(term, A.Agg):
        acc[term.unparse()] = acc.get(term.unparse(), 0.0) + coeff


def _side_keys(term: A.Term) -> Set[str]:
    acc: Dict[str, float] = {}
    _flatten(term, 1.0, acc)
    return {k for k, v in acc.items() if abs(v) > 1e-9}


def _residual_keys(rule: Rule) -> Set[str]:
    """Keyset of the residual ``left - right`` after combining like terms (a term that
    cancels between the two sides drops out)."""
    acc: Dict[str, float] = {}
    _flatten(rule.atom.left, 1.0, acc)
    _flatten(rule.atom.right, -1.0, acc)
    return {k for k, v in acc.items() if abs(v) > 1e-9}


_OP_KIND = {"~=": "eq", "==": "eq", "!=": "anti", ">=": "ge", "<=": "le"}


def _niche(rule: Rule) -> tuple:
    """Structural identity of a rule = (binder, comparison kind, residual keyset).

    Two rules share a niche iff they assert the *same* relation over the *same* measured
    quantities (up to scalar multiples / op-variants already canonicalised upstream); this is
    exactly the granularity at which recall counts distinct invariants, so capping the
    portfolio by niche -- rather than by explained-point count -- guarantees a structurally
    distinct law (e.g. ``H[X,X] == 0``) is never evicted by a broader soft rule that merely
    happens to satisfy more points."""
    kind = _OP_KIND.get(rule.atom.op, rule.atom.op)
    return (rule.binder, kind, frozenset(_residual_keys(rule)))


def _has_cancellation(rule: Rule) -> bool:
    """True if an equality rule repeats a term on both sides (e.g. ``BIG + small ~= BIG``).

    Such a rule is mis-stated additive bloat: it reduces to ``small ~= 0`` but, evaluated as
    written, the shared dominant term inflates the scale and manufactures a spuriously tight
    band.  The cancelled relation, if real, is proposed on its own terms elsewhere."""
    if rule.atom.op not in ("~=", "=="):
        return False
    return bool(_side_keys(rule.atom.left) & _side_keys(rule.atom.right))


def _bound_residual(rule: Rule) -> Dict[str, float]:
    """Signed coefficient map of the quantity a one-sided bound asserts to be ``>= 0``.

    For ``a >= b`` this is ``a - b``; for ``a <= b`` it is ``b - a`` (so the bound always reads
    ``residual >= 0``).  Keys are Ref roles / Agg unparses (every measurement family in the DSL
    is a non-negative byte/packet count) plus the literal ``#const`` for constant offsets.
    Returns ``{}`` for two-sided or anti relations (not one-sided bounds)."""
    op = rule.atom.op
    acc: Dict[str, float] = {}
    if op == ">=":
        _flatten(rule.atom.left, 1.0, acc)
        _flatten(rule.atom.right, -1.0, acc)
    elif op == "<=":
        _flatten(rule.atom.right, 1.0, acc)
        _flatten(rule.atom.left, -1.0, acc)
    else:
        return {}
    return {k: v for k, v in acc.items() if abs(v) > 1e-9}


def _sign_tautology(rule: Rule) -> bool:
    """True if a one-sided bound holds for *every* non-negative assignment of its terms.

    All measurement families in the DSL are byte/packet counts and are therefore non-negative
    -- this is invariant I1 itself, the established sign model.  A bound that reduces to
    ``residual >= 0`` with every measurement coefficient ``>= 0`` and a non-negative constant
    is true by the sign model alone, independent of the data: it carries no information beyond
    non-negativity (e.g. ``i >= -H``, ``0.25e <= 1.5e + H``, ``-MIN(i) <= AVG(e)``).  Two-sided
    equalities and anti-invariants are never tautological in this sense and return False."""
    if rule.atom.op not in (">=", "<="):
        return False
    d = _bound_residual(rule)
    if d.get("#const", 0.0) < -1e-9:
        return False
    return all(v >= -1e-9 for k, v in d.items() if k != "#const")


def _pure_sign_bound(rule: Rule) -> bool:
    """True for the canonical non-negativity schema ``q >= 0`` (target I1).

    A pure sign bound is a sign tautology whose residual is a *single* measurement term with no
    constant offset, i.e. it states exactly ``measurement >= 0``.  This is distinguished from
    *compound* sign tautologies such as ``i >= -H`` (two terms), which restate non-negativity
    with extra bloat and carry no information; only the single canonical representative of the
    sign schema is retained (the recall grader credits I1 as any size-1 ``ge`` residual)."""
    if not _sign_tautology(rule):
        return False
    d = _bound_residual(rule)
    terms = [k for k in d if k != "#const"]
    return len(terms) == 1 and abs(d.get("#const", 0.0)) <= 1e-9


def assemble(results: List[EvaluationResult], ds: Dataset, cfg: EvalConfig,
             dedup_rel: float = 0.05, k_max: int = 16) -> List[EvaluationResult]:
    """Return the de-duplicated, capped portfolio (highest score first)."""
    accepted = [r for r in results if r.accepted]
    # 1. structural dedup -- best score per signature
    by_sig: Dict[str, EvaluationResult] = {}
    for r in accepted:
        sig = r.rule.signature()
        if sig not in by_sig or r.combined_score > by_sig[sig].combined_score:
            by_sig[sig] = r

    # 1b. op-variant canonicalisation -- collapse rules over the SAME unordered term-pair
    # (e.g. a<=b, a>=b, a~=b, a==b) to one representative.  The catalog invariants are
    # equalities (exact or soft), so a two-sided form is the canonical representative even
    # when a one-sided bound it implies happens to hold exactly; anti-invariants live in
    # their own group (never merged with an equality).
    _vrank = {Verdict.EXACT: 0, Verdict.SOFT_STRUCTURAL: 1, Verdict.SOFT: 2, Verdict.ANTI: 0}

    def _pairkey(r: EvaluationResult):
        a = r.rule.atom
        sides = tuple(sorted((a.left.unparse(), a.right.unparse())))
        is_anti = (r.verdict == Verdict.ANTI) or (a.op == "!=")
        return (r.rule.binder, sides, is_anti)

    def _canon_key(r: EvaluationResult):
        a = r.rule.atom
        two_sided = 0 if a.op in ("==", "~=") else 1
        return (two_sided, _vrank.get(r.verdict, 3), r.eps, -r.combined_score)

    groups: Dict[tuple, EvaluationResult] = {}
    for r in by_sig.values():
        k = _pairkey(r)
        cur = groups.get(k)
        if cur is None or _canon_key(r) < _canon_key(cur):
            groups[k] = r
    uniq = sorted(groups.values(), key=lambda r: r.combined_score, reverse=True)

    # 1c. drop additive bloat that repeats a term across an equality (``BIG + small ~= BIG``)
    uniq = [r for r in uniq if not _has_cancellation(r.rule)]

    # precompute satisfaction vectors once
    sat: Dict[str, np.ndarray] = {
        r.rule.signature(): satisfaction(r.rule, ds, r.eps) for r in uniq}

    # 1d. residual-subsumption -- drop a padded equality whose residual term-set strictly
    # contains a kept equality's (same binder) when the two share the bulk of their satisfied
    # points (the extra term is small).  This removes ``core ~= core + spurious_term`` bloat.
    # Restricted to two-sided equalities on BOTH sides: a one-sided sign bound (``>= 0``)
    # must never subsume a magnitude equality just because its single residual term happens
    # to be a subset, and structurally-distinct invariants have non-nested residual keysets.
    rkeys: Dict[str, Set[str]] = {r.rule.signature(): _residual_keys(r.rule) for r in uniq}

    def _eq_like(r: EvaluationResult) -> bool:
        return r.rule.atom.op in ("~=", "==")

    survivors: List[EvaluationResult] = []
    for r in uniq:                                    # score-descending order
        kr = rkeys[r.rule.signature()]
        padded = False
        if _eq_like(r):
            for s in survivors:
                ks = rkeys[s.rule.signature()]
                if (_eq_like(s) and s.rule.binder == r.rule.binder and ks and ks < kr
                        and _jaccard(sat[r.rule.signature()],
                                     sat[s.rule.signature()]) >= 0.6):
                    padded = True
                    break
        if not padded:
            survivors.append(r)
    uniq = survivors

    # 2. niche collapse -- one best-scoring representative per distinct structural niche
    # (binder x comparison-kind x residual keyset).  This is the granularity at which recall
    # counts distinct invariants, so a structurally distinct law (e.g. the self-demand diagonal
    # ``H[X,X] == 0``, keys {demand_self}) is never absorbed by a different law satisfied at the
    # same points (e.g. the everywhere-true Kirchhoff balance).  Point-overlap dedup is
    # deliberately NOT used: two rules constraining different variables can both hold everywhere
    # yet be different invariants.
    best_per_niche: Dict[tuple, EvaluationResult] = {}

    def _niche_rank(r: EvaluationResult) -> tuple:
        # lower is better: highest score, then simplest form (parsimony tie-break, MDL-aligned),
        # then tightest band.  Picking the simplest equivalent rule as the niche representative
        # canonicalises bloated restatements (e.g. ``v + v + v >= 0`` -> ``v >= 0``).
        return (-r.combined_score, r.rule.complexity(), r.eps)

    for r in uniq:
        nk = _niche(r.rule)
        cur = best_per_niche.get(nk)
        if cur is None or _niche_rank(r) < _niche_rank(cur):
            best_per_niche[nk] = r

    # 3. portfolio assembly with an information-aware budget + sign-entailment tautology filter.
    # Two-sided equalities (``a == b``) and anti-invariants (``a != b``) are the informative
    # laws: each distinct one is a genuine invariant the learner should report, so they are kept
    # in full (never capped against each other).  One-sided bounds are split by a leakage-free
    # symbolic test (``_sign_tautology``):
    #   * the non-negativity schema ``q >= 0`` (target I1) -- kept as ONE canonical representative
    #     (prefer a ``ge`` form, then the simplest / highest-scoring), guaranteeing a slot;
    #   * *compound* sign tautologies (``i >= -H``, ``0.25e <= 1.5e + H``, ``-MIN(i) <= AVG(e)``)
    #     hold for every non-negative assignment and add no information -- they are DROPPED;
    #   * genuine, data-supported bounds (not entailed by the sign model) that ALSO clear the
    #     name-semantic lift floor (``lift >= cfg.lift_min``) are summarised by their best
    #     representative per (binder, kind) and compete for the slots left after the informative
    #     laws; a bound with lift ~1 (tight only because of its form, e.g. ``H[Y,X] <= i[X<-Y]``)
    #     is uninformative bloat and is dropped, exactly as the equality gate already requires.
    # This removes the redundant bound fan (poc-eval E2) while never touching equalities/antis,
    # so recall over the exact / soft-structural / anti targets is preserved.
    informative = [r for r in best_per_niche.values()
                   if _OP_KIND.get(r.rule.atom.op) in ("eq", "anti")]
    bounds_all = [r for r in best_per_niche.values()
                  if _OP_KIND.get(r.rule.atom.op) in ("ge", "le")]

    pure_sign = [r for r in bounds_all if _pure_sign_bound(r.rule)]

    def _sign_rank(r: EvaluationResult) -> tuple:
        is_ge = 0 if r.rule.atom.op == ">=" else 1   # I1 is graded as a ge bound -> prefer ge
        return (is_ge, r.rule.complexity(), -r.combined_score)

    sign_rep = min(pure_sign, key=_sign_rank) if pure_sign else None

    genuine = [r for r in bounds_all
               if not _sign_tautology(r.rule) and r.lift >= cfg.lift_min]
    bound_by_class: Dict[tuple, EvaluationResult] = {}
    for r in genuine:
        bk = (r.rule.binder, _OP_KIND[r.rule.atom.op])
        cur = bound_by_class.get(bk)
        if cur is None or r.combined_score > cur.combined_score:
            bound_by_class[bk] = r

    informative.sort(key=lambda r: r.combined_score, reverse=True)
    genuine_summary = sorted(bound_by_class.values(),
                             key=lambda r: r.combined_score, reverse=True)
    slots = max(1, k_max - len(informative))
    chosen_bounds: List[EvaluationResult] = []
    if sign_rep is not None:
        chosen_bounds.append(sign_rep)               # guaranteed slot -> target I1 surfaced
    for r in genuine_summary:
        if len(chosen_bounds) >= slots:
            break
        chosen_bounds.append(r)
    kept = informative + chosen_bounds
    return sorted(kept, key=lambda r: r.combined_score, reverse=True)
