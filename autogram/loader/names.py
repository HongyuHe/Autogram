"""Name-semantics parser ``pi`` and locality-family construction (design Sec. 3.1, 4.2).

A *deployed* learner only sees observable column *names* and *values*; it is given
no oracle map from column name to (type, locality, direction).  Proposing that map
is part of learning.  This module implements a deterministic parser that recovers
the structured semantics from the CrossCheck naming convention:

    low_<NODE>_origination
    low_<NODE>_termination
    low_<NODE>_egress_to_<PEER>
    low_<NODE>_ingress_from_<PEER>
    high_<SRC>_<DST>

Node tokens never contain ``_`` in either dataset (Abilene plain names, GEANT dotted
names such as ``at1.at``), so demand columns ``high_<SRC>_<DST>`` are split by matching
the longest known node token -- exactly the strategy of the existing validation harness.

The parser is the *legitimately available* signal handed to the proposer backends.
It reads only column names, never the ground-truth invariant catalog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Per-column structured semantics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSemantics:
    """Structured meaning recovered from a single column name."""

    name: str
    kind: str                      # 'low' | 'high'
    direction: str                 # origination|termination|egress|ingress|demand
    nodes: tuple                   # locality tokens this column touches (ordered)
    src: Optional[str] = None      # demand source / low local node
    dst: Optional[str] = None      # demand destination
    peer: Optional[str] = None     # egress/ingress neighbour

    @property
    def role(self) -> str:
        """Coarse dimensional role used by the DSL type checker.

        All measured quantities here are byte volumes, so they share one
        dimension group ('volume').  The role string distinguishes the
        measurement layer (low counter vs high demand) for interpretability and
        for the name-blind null, but does not block cross-layer comparison
        (I5/I6 legitimately compare a low counter to a sum of high demands).
        """
        return "volume"


_ORIG = re.compile(r"^low_(?P<node>.+)_origination$")
_TERM = re.compile(r"^low_(?P<node>.+)_termination$")
_EGR = re.compile(r"^low_(?P<node>.+)_egress_to_(?P<peer>.+)$")
_ING = re.compile(r"^low_(?P<node>.+)_ingress_from_(?P<peer>.+)$")


def parse_column(name: str, nodes: frozenset) -> Optional[ColumnSemantics]:
    """Parse one column name into :class:`ColumnSemantics`.

    Returns ``None`` for metadata columns (anything not ``low_*``/``high_*``).
    """
    if name.startswith("low_"):
        m = _ORIG.match(name)
        if m:
            n = m.group("node")
            return ColumnSemantics(name, "low", "origination", (n,), src=n)
        m = _TERM.match(name)
        if m:
            n = m.group("node")
            return ColumnSemantics(name, "low", "termination", (n,), dst=n)
        m = _EGR.match(name)
        if m:
            n, p = m.group("node"), m.group("peer")
            return ColumnSemantics(name, "low", "egress", (n, p), src=n, peer=p)
        m = _ING.match(name)
        if m:
            n, p = m.group("node"), m.group("peer")
            return ColumnSemantics(name, "low", "ingress", (n, p), dst=n, peer=p)
        return None
    if name.startswith("high_"):
        body = name[len("high_"):]
        sd = _split_demand(body, nodes)
        if sd is not None:
            s, d = sd
            return ColumnSemantics(name, "high", "demand", (s, d), src=s, dst=d)
        return None
    return None


def _split_demand(body: str, nodes: frozenset):
    """Split ``<SRC>_<DST>`` by matching a known node prefix (longest first)."""
    for s in sorted(nodes, key=len, reverse=True):
        prefix = s + "_"
        if body.startswith(prefix):
            d = body[len(prefix):]
            if d in nodes:
                return (s, d)
    return None


# ---------------------------------------------------------------------------
# Node-set inference and the whole-dataset name model
# ---------------------------------------------------------------------------

def infer_nodes(columns) -> frozenset:
    """Infer the node-token set from the union of all column conventions.

    Robust to nodes that lack an origination/termination counter: tokens are
    gathered from origination/termination *and* the local side of egress/ingress.
    Demand columns are resolved afterwards against this set.
    """
    nodes = set()
    for c in columns:
        m = _ORIG.match(c) or _TERM.match(c)
        if m:
            nodes.add(m.group("node"))
            continue
        m = _EGR.match(c) or _ING.match(c)
        if m:
            nodes.add(m.group("node"))
    return frozenset(nodes)


@dataclass
class NameModel:
    """Holds parsed semantics for every column and resolves locality families."""

    nodes: frozenset
    by_name: dict = field(default_factory=dict)          # name -> ColumnSemantics
    low_cols: tuple = ()
    high_cols: tuple = ()

    @classmethod
    def from_columns(cls, columns) -> "NameModel":
        nodes = infer_nodes(columns)
        by_name = {}
        low_cols, high_cols = [], []
        for c in columns:
            sem = parse_column(c, nodes)
            if sem is None:
                continue
            by_name[c] = sem
            if sem.kind == "low":
                low_cols.append(c)
            else:
                high_cols.append(c)
        return cls(nodes=nodes, by_name=by_name,
                   low_cols=tuple(low_cols), high_cols=tuple(high_cols))

    # -- locality families ---------------------------------------------------
    def resolve_family(self, token: str, type_: str, direction: str) -> tuple:
        """Resolve ``Fam(token, type, dir)`` to a tuple of concrete column names.

        ``token`` is a node, or ``'*'`` for a network-wide aggregate.  Empty
        results are legal (the DSL ``SUM`` of an empty family is 0).
        """
        out = []
        if type_ == "high":
            for c, sem in self.by_name.items():
                if sem.kind != "high":
                    continue
                if direction == "src" and sem.src == token and sem.dst != token:
                    out.append(c)
                elif direction == "dst" and sem.dst == token and sem.src != token:
                    out.append(c)
                elif direction == "all" and token == "*" and sem.src != sem.dst:
                    out.append(c)
        elif type_ == "low":
            for c, sem in self.by_name.items():
                if sem.kind != "low":
                    continue
                local = sem.nodes[0] if sem.nodes else None
                if sem.direction == direction and (token == "*" or token == local):
                    out.append(c)
        return tuple(sorted(out))

    def node_list(self) -> list:
        return sorted(self.nodes)
