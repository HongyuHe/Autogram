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
    agg_kinds: Tuple[str, ...] = ("SUM", "MIN", "MAX", "AVG")
    scale_coeffs: Tuple[float, ...] = (-1.0, 0.5, 2.0)
    max_complexity: int = 12
    max_add_arity: int = 3
    glyphs: Dict[str, str] = field(default_factory=dict)

    def refs_for(self, binder: str) -> Tuple[str, ...]:
        return tuple(self.ref_roles.get(binder, ()))

    def fams_for(self, binder: str) -> Tuple[str, ...]:
        return tuple(self.fam_roles.get(binder, ()))


def grammar_from_adapter(adapter, max_complexity: int = 12,
                         max_add_arity: int = 3,
                         scale_coeffs: Tuple[float, ...] = (-1.0, 0.5, 2.0)) -> Grammar:
    """Build the search grammar from a compiled schema adapter (the induced ontology)."""
    return Grammar(
        binders=tuple(adapter.binders),
        ops=tuple(adapter.ops),
        ref_roles={b: tuple(adapter.ref_roles.get(b, ())) for b in adapter.binders},
        fam_roles={b: tuple(adapter.fam_roles.get(b, ())) for b in adapter.binders},
        agg_kinds=tuple(adapter.agg_kinds),
        scale_coeffs=tuple(scale_coeffs),
        max_complexity=max_complexity,
        max_add_arity=max_add_arity,
    )
