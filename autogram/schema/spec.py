"""Declarative, JSON-serialisable schema description (the data an LLM/heuristic emits).

A :class:`SchemaSpec` is **data, not code**.  Every field is a primitive, a tuple of
primitives, or a tuple of small frozen dataclasses, so a spec can be serialised to/from JSON
and validated statically before it is ever used (``schema/validate.py``).  The compiler
(:func:`autogram.schema.compiler.compile_spec`) interprets these declarations with a fixed,
trusted, bounded vocabulary -- there is no ``eval`` and no arbitrary code path -- which is
what lets an *untrusted* proposer widen the schema without widening the trusted base.

The structure mirrors the four CrossCheck seams it generalises:

* :class:`ColumnPattern`  -- name -> ``(kind, direction, nodes, slots)`` parser (seam 1).
* :class:`RoleOntology`   -- binders and their legal ref/family roles (seam 2).
* :class:`RefTemplate` / :class:`FamilySelector` -- role -> concrete column grounding (seam 3).
* :class:`CellCodec`      -- how a stored cell yields its observed/clean scalar (seam 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Bounded, trusted enumeration of binding-enumeration strategies a binder may use.  The
# compiler maps each id to a fixed Python implementation; a spec can only *name* one of these,
# never supply code.
ENUMERATE_STRATEGIES = (
    "per_measured_col",     # cell binder: one binding {col: c} per measured column
    "per_node",             # node binder: one binding {X: node} per node token
    "per_directed_link",    # link binder: {X, Y} per directed link that carries a counter
    "singleton",            # network binder: a single empty binding {}
)

# Bounded predicate slots/operators for family selectors (see :class:`FamilySelector`).
PRED_SLOTS = ("src", "dst", "peer", "local")     # local == nodes[0]
PRED_OPS = ("==", "!=")
CELL_CODECS = ("dict_gt_hidden", "scalar")


@dataclass(frozen=True)
class ColumnPattern:
    """One column-name -> semantics rule (anchored regex, or a known-token split).

    ``matcher == "regex"``: ``regex`` is an anchored pattern with named groups; ``node_groups``
    names the groups (in order) that form the ``nodes`` tuple, and ``src_group`` / ``dst_group``
    / ``peer_group`` (when set) name the groups bound to those semantic slots.

    ``matcher == "split"``: a column is matched iff it starts with ``prefix``; the remainder is
    split into two node tokens using the *known node universe* (longest-known-prefix, the robust
    rule for tokens that themselves contain the separator, e.g. dotted GEANT names).  The two
    halves fill ``split_slots`` (default ``("src", "dst")``) and become ``nodes``.

    ``token_groups`` lists the groups (regex) that contribute to the inferred node universe.  For
    a split pattern, splitting is resolved *against* that universe, so it contributes nothing.
    """
    name: str
    matcher: str                      # "regex" | "split"
    kind: str
    direction: str
    # regex matcher
    regex: str = ""
    node_groups: Tuple[str, ...] = ()
    src_group: str = ""
    dst_group: str = ""
    peer_group: str = ""
    token_groups: Tuple[str, ...] = ()
    # split matcher
    prefix: str = ""
    sep: str = "_"
    split_slots: Tuple[str, str] = ("src", "dst")


@dataclass(frozen=True)
class RoleOntology:
    """The binder set and each binder's legal ref/family roles, plus operator/agg vocab.

    ``ref_roles`` / ``fam_roles`` are ``binder -> tuple(role_name)`` maps; the engine's typing
    (``dsl/typecheck.py``) admits a rule only if its roles are listed for its binder *and*
    enabled in the live grammar.  ``ref_glyphs`` / ``fam_glyphs`` give compact ASCII unparse
    symbols (falling back to the role name when absent).
    """
    binders: Tuple[str, ...]
    ref_roles: Dict[str, Tuple[str, ...]]
    fam_roles: Dict[str, Tuple[str, ...]]
    ops: Tuple[str, ...] = ("~=", "==", "!=", "<=", ">=")
    agg_kinds: Tuple[str, ...] = ("SUM", "MIN", "MAX", "AVG")
    ref_glyphs: Dict[str, str] = field(default_factory=dict)
    fam_glyphs: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RefTemplate:
    """Grounds a single ``Ref`` role to a column name via a format string over binding vars.

    ``template`` is a Python ``str.format`` pattern whose fields are the binding variables the
    binder exposes (``{X}``/``{Y}`` for node/link, ``{col}`` for cell).  The grounded name is
    accepted only if it exists in the dataset (checked by the adapter against ``nm.by_name``),
    so a template that does not correspond to a real column simply yields no binding.
    """
    binder: str
    role: str
    template: str


@dataclass(frozen=True)
class FamilySelector:
    """Grounds an ``Agg`` family role to the set of columns matching a bounded predicate.

    A column's parsed :class:`~autogram.loader.names.ColumnSemantics` qualifies iff its ``kind``
    equals ``match_kind``, its ``direction`` equals ``match_direction`` (when set), and *all*
    ``predicates`` hold.  Each predicate is ``[slot, op, rhs]`` with ``slot`` in
    :data:`PRED_SLOTS`, ``op`` in :data:`PRED_OPS`, and ``rhs`` one of:

    * ``"*"``                 -- wildcard (always satisfied; used with ``==``),
    * a binding variable name -- e.g. ``"X"`` / ``"Y"`` (compared to the binding value),
    * ``"@<slot>"``           -- another slot of the *same* column (e.g. ``"@dst"``).
    """
    binder: str
    family_role: str
    match_kind: str
    match_direction: Optional[str] = None
    predicates: Tuple[Tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class CellCodec:
    """How one stored cell yields its observed (noisy) and clean scalar.

    * ``"dict_gt_hidden"`` -- the CrossCheck format: each cell is a ``dict`` whose ``primary``
      key (default ``ground_truth``) is the observed value and whose ``clean`` key (default
      ``hidden_ground_truth``) is the clean value when present (else falls back to ``primary``).
    * ``"scalar"``         -- the cell *is* a number; observed == clean (noise, if any, is
      already baked into the stored value and there is no separate clean oracle column).
    """
    kind: str = "dict_gt_hidden"
    primary: str = "ground_truth"
    clean: str = "hidden_ground_truth"


@dataclass(frozen=True)
class SchemaSpec:
    """A complete, bounded description of a dataset schema (parser + ontology + grounding).

    ``noisy_kind`` names the column ``kind`` that carries injected noise (the ``low_*`` layer in
    CrossCheck); those columns are the ones the noise model treats as noisy and the clean codec
    reads from.  ``demand_kind`` names the demand-matrix layer (``high_*``).  ``link_marker_dir``
    is the direction string the ``per_directed_link`` strategy keys on to discover directed
    links (CrossCheck: ``"egress"``).
    """
    name: str
    patterns: Tuple[ColumnPattern, ...]
    ontology: RoleOntology
    ref_templates: Tuple[RefTemplate, ...]
    family_selectors: Tuple[FamilySelector, ...]
    binder_enumerate: Dict[str, str]            # binder -> strategy id (ENUMERATE_STRATEGIES)
    cell_codec: CellCodec = field(default_factory=CellCodec)
    noisy_kind: str = "low"
    demand_kind: str = "high"
    link_marker_dir: str = "egress"
    notes: str = ""
