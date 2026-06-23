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
from typing import List, Tuple

from . import ast as A
from .ast import Add, Agg, Compare, Const, Ref, Rule, Scale
from .typecheck import is_admissible


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


def enumerate_candidates(G: Grammar, nm=None) -> List[Rule]:
    """Return all admissible seed templates under ``G`` (the starting candidate pool)."""
    out = []
    for r in seed_templates():
        ok, _ = is_admissible(r, G)
        if ok:
            out.append(r)
    return out
