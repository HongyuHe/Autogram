"""The grammar ``G``: the bounded hypothesis space the discovery search explores.

``G`` is derived entirely from an *induced* schema -- it names the binders, operators,
single-column roles and family roles that the schema actually grounds, plus size bounds.  No
role vocabulary is hardcoded here; :func:`grammar_from_adapter` reads it out of a compiled
:class:`~autogram.schema.adapter.SchemaAdapter`, which an inducer built from column names.

The grammar is the *search space*; the proposer's job is to find high-scoring rules inside it
and the evaluator's job is to decide -- from data alone -- which candidates are genuine
invariants and at what strictness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Grammar:
    """The enabled vocabulary, keyed per binder (roles depend on the binder)."""

    binders: Tuple[str, ...]
    ops: Tuple[str, ...]
    ref_roles: Dict[str, Tuple[str, ...]]      # binder -> single-column roles
    fam_roles: Dict[str, Tuple[str, ...]]      # binder -> family roles
    agg_kinds: Tuple[str, ...] = ("SUM",)
    scale_coeffs: Tuple[float, ...] = (-1.0, 0.5, 2.0)
    max_complexity: int = 12
    max_add_arity: int = 3
    max_degree: int = 1                          # 1 = linear; >=2 enables Mul/Div (item 6)
    role_exclusions: Tuple[frozenset, ...] = ()  # role pairs that may not co-occur (item 2)
    glyphs: Dict[str, str] = field(default_factory=dict)
    # Per-binder size caps. Empty maps fall back to the global scalars above, so behaviour is
    # unchanged unless a policy (see ``default_binder_caps``) or inducer populates them.
    max_complexity_by_binder: Dict[str, int] = field(default_factory=dict)
    max_add_arity_by_binder: Dict[str, int] = field(default_factory=dict)
    max_degree_by_binder: Dict[str, int] = field(default_factory=dict)

    def refs_for(self, binder: str) -> Tuple[str, ...]:
        return tuple(self.ref_roles.get(binder, ()))

    def fams_for(self, binder: str) -> Tuple[str, ...]:
        return tuple(self.fam_roles.get(binder, ()))

    def complexity_cap(self, binder: str) -> int:
        return int(self.max_complexity_by_binder.get(binder, self.max_complexity))

    def add_arity_cap(self, binder: str) -> int:
        return int(self.max_add_arity_by_binder.get(binder, self.max_add_arity))

    def degree_cap(self, binder: str) -> int:
        return int(self.max_degree_by_binder.get(binder, self.max_degree))


def default_binder_caps(binders, ref_roles, fam_roles, max_complexity, max_add_arity, max_degree):
    """Structural per-binder size caps (enabled by default in :func:`grammar_from_adapter`).

    A binder that can *aggregate* (exposes family roles) hosts wide sums and conservation balances,
    so it keeps the full global complexity.  A non-aggregating binder only hosts small relational
    forms (non-negativity, zero, pairwise equality/separation), so its complexity is capped tighter
    -- this prunes needlessly large forms on those binders (scalability + precision) without
    dropping the small laws they actually carry.  Additive arity is bounded by the number of leaves
    a binder exposes (you cannot sum more distinct leaves than exist).  Degree mirrors the global
    cap so capability-tier widening still applies uniformly.
    """
    comp, arity, deg = {}, {}, {}
    for b in binders:
        has_family = bool(fam_roles.get(b))
        n_leaves = len(ref_roles.get(b, ())) + len(fam_roles.get(b, ()))
        comp[b] = int(max_complexity) if has_family else min(int(max_complexity), 6)
        arity[b] = min(int(max_add_arity), max(2, n_leaves))
        deg[b] = int(max_degree)
    return comp, arity, deg


def grammar_from_adapter(adapter, max_complexity: int = 12,
                         max_add_arity: int = 3,
                         scale_coeffs: Tuple[float, ...] = (-1.0, 0.5, 2.0)) -> Grammar:
    """Build the search grammar from a compiled schema adapter (the induced ontology).

    Per-binder size caps are populated by default from the binder structure (see
    :func:`default_binder_caps`); the global scalars remain the fallback for any binder not listed.
    """
    binders = tuple(adapter.binders)
    ref_roles = {b: tuple(adapter.ref_roles.get(b, ())) for b in binders}
    fam_roles = {b: tuple(adapter.fam_roles.get(b, ())) for b in binders}
    max_degree = getattr(adapter, "max_degree", 1)
    comp, arity, deg = default_binder_caps(binders, ref_roles, fam_roles,
                                           max_complexity, max_add_arity, max_degree)
    return Grammar(
        binders=binders,
        ops=tuple(adapter.ops),
        ref_roles=ref_roles,
        fam_roles=fam_roles,
        agg_kinds=tuple(adapter.agg_kinds),
        scale_coeffs=tuple(scale_coeffs),
        max_complexity=max_complexity,
        max_add_arity=max_add_arity,
        max_degree=max_degree,
        role_exclusions=getattr(adapter, "role_exclusions", ()),
        max_complexity_by_binder=comp,
        max_add_arity_by_binder=arity,
        max_degree_by_binder=deg,
    )
