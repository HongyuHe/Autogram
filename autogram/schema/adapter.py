"""The compiled, callable form of a :class:`~autogram.schema.spec.SchemaSpec`.

A :class:`SchemaAdapter` is produced by :func:`autogram.schema.compiler.compile_spec` and is
the single object the engine consults on the *generalised* path.  It reproduces, for an
arbitrary (bounded) schema, the four behaviours that are otherwise hardcoded for CrossCheck:

================  ============================================  =====================================
Adapter method    CrossCheck hardcode it generalises            Seam
================  ============================================  =====================================
``parse_column``  ``loader.names.parse_column``                 1 (name parser)
``infer_tokens``  ``loader.names.infer_nodes``                  1 (node universe)
``ref_roles`` /   static ``dsl.ast.REF_ROLES`` / ``FAM_ROLES``  2 (role ontology)
``fam_roles``
``resolve_ref`` / ``dsl.binders.resolve_ref`` /                 3 (grounding)
``resolve_family``  ``resolve_family`` / ``enumerate_bindings``
``decode_*``      ``loader.loader._cells_to_matrix``            4 (cell codec)
================  ============================================  =====================================

The adapter holds only *compiled data* (pre-compiled regexes, dict lookups, bounded strategy
ids).  It contains no user-supplied code and performs no ``eval``; every method is a fixed
interpreter over the declarations, which is what makes an untrusted, proposer-supplied spec
safe to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..loader.names import ColumnSemantics


@dataclass
class _Pattern:
    """One compiled :class:`~autogram.schema.spec.ColumnPattern`."""
    matcher: str
    kind: str
    direction: str
    rx: Optional["re.Pattern"]
    node_groups: Tuple[str, ...]
    src_group: str
    dst_group: str
    peer_group: str
    token_groups: Tuple[str, ...]
    prefix: str
    sep: str
    split_slots: Tuple[str, str]


@dataclass
class SchemaAdapter:
    """Compiled schema description; the engine's generalised parser + grounder + codec."""

    name: str
    patterns: Tuple[_Pattern, ...]
    ref_roles: Dict[str, Tuple[str, ...]]
    fam_roles: Dict[str, Tuple[str, ...]]
    binders: Tuple[str, ...]
    ops: Tuple[str, ...]
    agg_kinds: Tuple[str, ...]
    ref_templates: Dict[Tuple[str, str], str]
    family_selectors: Dict[Tuple[str, str], "object"]
    binder_enumerate: Dict[str, str]
    codec_kind: str
    codec_primary: str
    codec_clean: str
    noisy_kind: str
    demand_kind: str
    link_marker_dir: str
    ref_glyphs: Dict[str, str] = field(default_factory=dict)
    fam_glyphs: Dict[str, str] = field(default_factory=dict)

    # -- seam 1: name parsing ------------------------------------------------
    def parse_column(self, name: str, nodes) -> Optional[ColumnSemantics]:
        """Recover :class:`ColumnSemantics` from a column name (or ``None`` if no pattern)."""
        for p in self.patterns:
            if p.matcher == "regex":
                m = p.rx.match(name)
                if not m:
                    continue
                groups = m.groupdict()
                nodes_t = tuple(groups[g] for g in p.node_groups)
                return ColumnSemantics(
                    name, p.kind, p.direction, nodes_t,
                    src=groups.get(p.src_group) if p.src_group else None,
                    dst=groups.get(p.dst_group) if p.dst_group else None,
                    peer=groups.get(p.peer_group) if p.peer_group else None,
                )
            else:  # split
                if not name.startswith(p.prefix):
                    continue
                body = name[len(p.prefix):]
                sd = self._split_known(body, nodes)
                if sd is None:
                    continue
                a, b = sd
                slot_a, slot_b = p.split_slots
                kw = {slot_a: a, slot_b: b}
                return ColumnSemantics(
                    name, p.kind, p.direction, (a, b),
                    src=kw.get("src"), dst=kw.get("dst"), peer=kw.get("peer"),
                )
        return None

    @staticmethod
    def _split_known(body: str, nodes):
        for s in sorted(nodes, key=len, reverse=True):
            prefix = s + "_"
            if body.startswith(prefix):
                d = body[len(prefix):]
                if d in nodes:
                    return (s, d)
        return None

    def infer_tokens(self, columns) -> frozenset:
        """Infer the node-token universe from the columns' regex token groups."""
        toks = set()
        for c in columns:
            for p in self.patterns:
                if p.matcher != "regex" or not p.token_groups:
                    continue
                m = p.rx.match(c)
                if m:
                    g = m.groupdict()
                    for name in p.token_groups:
                        if g.get(name):
                            toks.add(g[name])
                    break
        return frozenset(toks)

    # -- seam 3: grounding ---------------------------------------------------
    def enumerate_bindings(self, binder: str, nm) -> List[dict]:
        strat = self.binder_enumerate.get(binder)
        if strat == "per_measured_col":
            return [{"col": c} for c in (list(nm.low_cols) + list(nm.high_cols))]
        if strat == "per_node":
            return [{"X": x} for x in nm.node_list()]
        if strat == "per_directed_link":
            out = []
            for x in nm.node_list():
                for c, sem in nm.by_name.items():
                    if (sem.kind == self.noisy_kind
                            and sem.direction == self.link_marker_dir
                            and sem.src == x):
                        out.append({"X": x, "Y": sem.peer})
            return out
        if strat == "singleton":
            return [{}]
        raise ValueError(f"binder {binder!r} has no enumerate strategy")

    def resolve_ref(self, role: str, binder: str, binding: dict, nm) -> Optional[str]:
        tmpl = self.ref_templates.get((binder, role))
        if tmpl is None:
            return None
        try:
            name = tmpl.format(**binding)
        except (KeyError, IndexError):
            return None
        return name if name in nm.by_name else None

    def resolve_family(self, family_role: str, binder: str, binding: dict, nm) -> tuple:
        sel = self.family_selectors.get((binder, family_role))
        if sel is None:
            return ()
        out = []
        for c, sem in nm.by_name.items():
            if sem.kind != sel.match_kind:
                continue
            if sel.match_direction is not None and sem.direction != sel.match_direction:
                continue
            if all(self._pred_ok(pred, sem, binding) for pred in sel.predicates):
                out.append(c)
        return tuple(sorted(out))

    @staticmethod
    def _slot(sem: ColumnSemantics, slot: str):
        if slot == "src":
            return sem.src
        if slot == "dst":
            return sem.dst
        if slot == "peer":
            return sem.peer
        if slot == "local":
            return sem.nodes[0] if sem.nodes else None
        return None

    def _pred_ok(self, pred, sem: ColumnSemantics, binding: dict) -> bool:
        slot, op, rhs = pred
        left = self._slot(sem, slot)
        if rhs == "*":
            right = left  # wildcard: trivially equal
        elif rhs.startswith("@"):
            right = self._slot(sem, rhs[1:])
        elif rhs in binding:
            right = binding[rhs]
        else:
            right = rhs
        return (left == right) if op == "==" else (left != right)

    # -- seam 4: cell codec --------------------------------------------------
    def decode_observed(self, v) -> float:
        if self.codec_kind == "scalar":
            return float("nan") if v is None else float(v)
        x = v.get(self.codec_primary)
        return float("nan") if x is None else float(x)

    # -- glyphs (interpretability; role names are reused so engine unparse is unchanged) --
    def ref_glyph(self, role: str) -> str:
        return self.ref_glyphs.get(role, role)

    def fam_glyph(self, role: str) -> str:
        return self.fam_glyphs.get(role, role)
