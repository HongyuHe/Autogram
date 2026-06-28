"""The grammar ``G``: the bounded hypothesis space the search explores (design Sec. 4, 6.6).

``G`` names the enabled binders, operators, single-column roles, family roles, and
aggregation kinds, plus size bounds (max complexity, max ``Add`` arity).  It is the
*search space* -- the LLM/subagent proposer's job is to *extend* it (e.g. enable a new
family role) when the inner search plateaus (Sec. 10.4), the evolutionary search's job
is to find high-scoring rules inside it, and the evaluator's job is to decide -- from
data -- which candidates are genuine invariants and at what strictness.

``enumerate_candidates`` materializes a bounded, admissible set of *hypothesis
templates* from ``G`` (true forms and plausible decoys alike).  It is purely
combinatorial over the role vocabulary and column names; it never consults the
ground-truth catalog, so it is leakage-free by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

from . import ast as A
from .ast import Add, Agg, Compare, Const, Ref, Rule, Scale
from .typecheck import _base_family, is_admissible


@dataclass
class Grammar:
    binders: Tuple[str, ...] = A.BINDERS
    ops: Tuple[str, ...] = A.OPS
    ref_roles: Tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(
            {r for rs in A.REF_ROLES.values() for r in rs})))
    fam_roles: Tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(
            {r for rs in A.FAM_ROLES.values() for r in rs})))
    agg_kinds: Tuple[str, ...] = A.AGG_KINDS
    max_complexity: int = 12
    max_add_arity: int = 3
    # extension knobs the proposer may flip on/off (start conservative)
    extensions: Tuple[str, ...] = ()


def default_grammar() -> Grammar:
    return Grammar()


def full_ceiling(max_complexity: int = 12, max_add_arity: int = 3) -> Grammar:
    """The maximal-vocabulary grammar -- the hard boundary a proposer may widen *toward*.

    Enables every binder, operator, single-column role, family role and aggregation kind the
    AST knows (the full ``ast`` vocabulary) at the given size caps.
    The outer-loop extension parser intersects every proposed widening against this ceiling
    (dropping unknown tokens) and never lets a proposed cap exceed it, so a backend can only
    re-enable vocabulary the evaluator can already ground -- the primary anti-blow-up guard on
    the LLM-driven grammar expansion (Sec. 10.4).
    """
    return Grammar(
        binders=A.BINDERS,
        ops=A.OPS,
        ref_roles=tuple(sorted({r for rs in A.REF_ROLES.values() for r in rs})),
        fam_roles=tuple(sorted({r for rs in A.FAM_ROLES.values() for r in rs})),
        agg_kinds=A.AGG_KINDS,
        max_complexity=max_complexity,
        max_add_arity=max_add_arity,
    )


def narrow_grammar(max_complexity: int = 8, max_add_arity: int = 2) -> Grammar:
    """A deliberately restricted starting grammar for ``grammar.start = "narrow"``.

    Only the cell/node binders, the approx/exact-equality and lower-bound operators, and the
    node single-column roles are enabled -- enough for non-negativity and the simplest node
    identities, but NOT the link pairing/separation roles, the family aggregations, or the
    network totals.
    Starting here forces the outer-loop proposer to *widen* the grammar (add the link binder,
    the family roles, the SUM aggregate, ...) before the middle loop can reach the link- and
    network-level laws -- exactly the adaptive grammar-expansion behaviour the proposer is
    meant to drive.
    It is leakage-free: it restricts the search space by vocabulary alone and encodes no
    catalogued answer.
    """
    return Grammar(
        binders=("cell", "node"),
        ops=("~=", "==", ">="),
        ref_roles=("self", "origination", "termination", "demand_self"),
        fam_roles=(),
        agg_kinds=("SUM",),
        max_complexity=max_complexity,
        max_add_arity=max_add_arity,
    )


def seed_templates() -> List[Rule]:
    """Hand-free structural seeds spanning the operator/role vocabulary.

    These are generic shapes (non-negativity, pairwise agreement, counter-vs-aggregate,
    conservation, network totals) plus deliberate *decoys* (wrong pairings, loose
    aggregates) so the evaluator must do real work to separate signal from noise and
    tautology.  No per-rule threshold and no oracle answer is encoded here.
    """
    seeds: List[Rule] = []

    # --- cell-level ---------------------------------------------------------
    seeds.append(Rule("cell", Compare(Ref("self"), ">=", Const(0)), tag="nonneg"))

    # --- node-level ---------------------------------------------------------
    seeds.append(Rule("node", Compare(Ref("demand_self"), "==", Const(0)), tag="self0"))
    seeds.append(Rule("node", Compare(Ref("origination"), "~=", Agg("SUM", "demand_row")),
                      tag="orig=rowsum"))
    seeds.append(Rule("node", Compare(Ref("termination"), "~=", Agg("SUM", "demand_col")),
                      tag="term=colsum"))
    # conservation: orig + ingress ~= term + egress
    seeds.append(Rule("node", Compare(
        Add((Ref("origination"), Agg("SUM", "ingress_fam"))),
        "~=",
        Add((Ref("termination"), Agg("SUM", "egress_fam"))),
    ), tag="flow-cons"))
    # decoys
    seeds.append(Rule("node", Compare(Ref("origination"), "~=", Agg("SUM", "demand_col")),
                      tag="decoy:orig=colsum"))
    seeds.append(Rule("node", Compare(Ref("origination"), "~=", Ref("termination")),
                      tag="decoy:orig=term"))

    # --- link-level ---------------------------------------------------------
    seeds.append(Rule("link", Compare(Ref("egress"), "~=", Ref("ingress_rev")),
                      tag="two-end"))
    seeds.append(Rule("link", Compare(Ref("egress"), "!=", Ref("egress_rev")),
                      tag="directionality"))
    # decoys
    seeds.append(Rule("link", Compare(Ref("egress"), "~=", Ref("egress_rev")),
                      tag="decoy:sym-egress"))
    seeds.append(Rule("link", Compare(Ref("egress"), "~=", Ref("demand")),
                      tag="decoy:egress=demand"))

    # --- network-level ------------------------------------------------------
    seeds.append(Rule("network", Compare(Agg("SUM", "all_orig"), "~=", Agg("SUM", "all_demand")),
                      tag="tot:orig=demand"))
    seeds.append(Rule("network", Compare(Agg("SUM", "all_term"), "~=", Agg("SUM", "all_demand")),
                      tag="tot:term=demand"))
    seeds.append(Rule("network", Compare(Agg("SUM", "all_orig"), "~=", Agg("SUM", "all_term")),
                      tag="tot:orig=term"))
    return seeds


def anti_invariant_seeds(G: Grammar) -> List[Rule]:
    """Same-base-family reverse-orientation separation candidates ``a != a_rev``.

    For every enabled binder, the binder's single-column roles are grouped by *base
    measurement family* (``egress`` / ``ingress`` / ``demand``, stripping a ``_rev``
    direction suffix via :func:`typecheck._base_family`) and a disequality is emitted for
    each within-family pair (e.g. a link's forward vs. reverse measurement of the same
    quantity).  This is the structural "directional-asymmetry" niche.

    Two properties make this leakage-free and non-question-begging, exactly like
    :func:`enumerate_candidates`:

    * it enumerates purely from ``G``'s role vocabulary -- never from the ground-truth
      catalog -- and emits the *whole family* of separation forms (egress, ingress and
      demand asymmetry alike), not just the one the catalog happens to credit;
    * each candidate still earns its ``ANTI`` verdict only if the evaluator confirms a
      persistent separation in the data; a pair that collapses to the noise floor is
      rejected.

    The reason it is seeded *deterministically* (rather than left to the proposer or the
    random generator) is purely a coverage argument: ``!=`` is one of five operators and is
    additionally restricted to same-family ``Ref``-vs-``Ref`` leaves, so blind search
    samples this niche very rarely -- the independent evaluation (``docs/autogram_poc_eval.md``,
    rec. 4) found anti-invariant recovery was hostage to proposer luck.  Seeding this single
    niche removes that dependence without seeding the full template enumerator (which would
    defeat the proposer-isolation test); the knob ``search.anti_seeds`` gates it.
    """
    seeds: List[Rule] = []
    seen: set = set()
    for b in G.binders:
        roles = [r for r in A.REF_ROLES.get(b, ()) if r in G.ref_roles]
        fams: Dict[str, List[str]] = {}
        for r in roles:
            fams.setdefault(_base_family(r), []).append(r)
        for fam, members in fams.items():
            for a, c in combinations(sorted(members), 2):
                rule = Rule(b, Compare(Ref(a), "!=", Ref(c)), tag=f"sep:{fam}")
                ok, _ = is_admissible(rule, G)
                if ok and rule.signature() not in seen:
                    seen.add(rule.signature())
                    seeds.append(rule)
    return seeds


def structural_invariant_seeds(G: Grammar) -> List[Rule]:
    """Combinatorial aggregate-equality candidates (counter-vs-sum, two-end, totals, balance).

    This is the equality analog of :func:`anti_invariant_seeds`: for every enabled binder it
    enumerates a *whole structural family* from ``G``'s role vocabulary -- never from the
    ground-truth catalog -- emitting the true forms and plausible decoys alike, each of which
    must still earn its verdict from the data.  Four leakage-free families are produced:

    * **counter-vs-aggregate** ``ref ~= SUM(fam)`` over every single-column role x family role
      on a binder (covers node origination=row-sum / termination=col-sum and many decoys);
    * **two-end agreement** ``a ~= b`` over single-column roles of *different* base families
      (covers a link's egress=reverse-ingress; same-family pairs are the ``anti`` niche, not an
      equality, and are skipped);
    * **network totals** ``SUM(f) ~= SUM(g)`` over distinct family roles (covers all-origination
      = all-termination network conservation and decoys);
    * **flow-conservation prior** ``orig + SUM(ingress_fam) ~= term + SUM(egress_fam)`` plus its
      swapped decoy -- a generic textbook Kirchhoff in=out balance, admitted only when the needed
      roles/aggregates are enabled; it is a domain prior, not a catalogued answer.

    The motivation is identical to the anti niche (a coverage argument, not an oracle): an exact
    two-term or aggregate pairing is a vanishingly small fraction of a complexity-12, arity-3
    space, so a leakage-free isolated proposer rarely hands these forms over and blind mutation
    almost never hits them -- the deployed eval found the row-sum / col-sum / conservation /
    two-end laws were never *seeded* nor *discovered*.  Seeding this single structural family
    (decoy-rich, data-gated) removes that dependence without enumerating the curated answer set;
    the knob ``search.structural_seeds`` gates it.
    """
    eq = "~=" if "~=" in G.ops else ("==" if "==" in G.ops else None)
    if eq is None:
        return []
    has_sum = "SUM" in G.agg_kinds
    seeds: List[Rule] = []
    seen: set = set()

    def _add(rule: Rule) -> None:
        ok, _ = is_admissible(rule, G)
        if ok and rule.signature() not in seen:
            seen.add(rule.signature())
            seeds.append(rule)

    for b in G.binders:
        refs = [r for r in A.REF_ROLES.get(b, ()) if r in G.ref_roles]
        fams = [f for f in A.FAM_ROLES.get(b, ()) if f in G.fam_roles]
        # counter-vs-aggregate: every single-column role against every family sum
        if has_sum:
            for r in refs:
                for f in fams:
                    _add(Rule(b, Compare(Ref(r), eq, Agg("SUM", f)), tag=f"agg:{r}~sum({f})"))
        # two-end agreement: cross-base-family single-column pairs only
        for a, c in combinations(sorted(refs), 2):
            if _base_family(a) == _base_family(c):
                continue
            _add(Rule(b, Compare(Ref(a), eq, Ref(c)), tag=f"pair:{a}~{c}"))
        # network totals: distinct family sums against each other
        if has_sum:
            for f, h in combinations(sorted(fams), 2):
                _add(Rule(b, Compare(Agg("SUM", f), eq, Agg("SUM", h)), tag=f"tot:{f}~{h}"))
        # flow-conservation prior (in=out balance) + swapped decoy, when the roles exist
        if has_sum:
            rs, fs = set(refs), set(fams)
            if {"origination", "termination"} <= rs and {"ingress_fam", "egress_fam"} <= fs:
                _add(Rule(b, Compare(
                    Add((Ref("origination"), Agg("SUM", "ingress_fam"))), eq,
                    Add((Ref("termination"), Agg("SUM", "egress_fam"))),
                ), tag="flow-cons"))
                _add(Rule(b, Compare(
                    Add((Ref("origination"), Agg("SUM", "egress_fam"))), eq,
                    Add((Ref("termination"), Agg("SUM", "ingress_fam"))),
                ), tag="decoy:flow-swap"))
    return seeds


def enumerate_candidates(G: Grammar, nm=None) -> List[Rule]:
    """Return all admissible seed templates under ``G`` (the starting candidate pool)."""
    out = []
    for r in seed_templates():
        ok, _ = is_admissible(r, G)
        if ok:
            out.append(r)
    return out
