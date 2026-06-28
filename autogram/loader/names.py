"""Column-name semantics and the dataset name model (grounding infrastructure).

A deployed learner only sees observable column *names* and *values*; it is given no oracle
map from a column name to its (kind, locality, direction).  Recovering that map is part of
learning, and it is done by an *induced* schema (:mod:`autogram.discovery.induce` ->
:class:`autogram.schema.adapter.SchemaAdapter`).

This module is schema-agnostic infrastructure:

* :class:`ColumnSemantics` is the structured meaning of one column (whatever the adapter
  parsed it into).
* :class:`NameModel` holds the parsed semantics for every column and the inferred token
  universe, and carries the adapter that grounding consults.

There is no dataset-specific parser here; :meth:`NameModel.from_columns_with_adapter` always
parses through an induced adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ColumnSemantics:
    """Structured meaning recovered from a single column name by the induced schema."""

    name: str
    kind: str                      # measurement layer / entity-kind token
    direction: str                 # relation/direction token (e.g. egress, demand, ...)
    nodes: tuple                   # entity tokens this column touches (ordered)
    src: Optional[str] = None      # relation source / local entity
    dst: Optional[str] = None      # relation destination
    peer: Optional[str] = None     # neighbour entity

    @property
    def role(self) -> str:
        """Coarse dimensional role used by the type checker.

        All measured quantities share one comparable dimension here, so the role string is a
        single ``"volume"`` group; the ``kind``/``direction`` carry the interpretable
        structure (and seed the name-permutation null), but do not block cross-kind
        comparison.
        """
        return "volume"


@dataclass
class NameModel:
    """Holds parsed semantics for every column and resolves locality tokens."""

    nodes: frozenset
    by_name: dict = field(default_factory=dict)          # name -> ColumnSemantics
    low_cols: tuple = ()                                  # columns of the noisy_kind layer
    high_cols: tuple = ()                                 # columns of the other layer(s)
    adapter: object = None                                # compiled SchemaAdapter

    @classmethod
    def from_columns_with_adapter(cls, columns, adapter) -> "NameModel":
        """Build a name model from a compiled :class:`SchemaAdapter` (the induced path).

        Parsing and token inference are delegated to the adapter; the ``low``/``high`` split
        keys on the adapter's declared ``noisy_kind`` so an arbitrary schema's primary-vs-other
        columns land in buckets the engine understands.
        """
        nodes = adapter.infer_tokens(columns)
        by_name = {}
        low_cols, high_cols = [], []
        for c in columns:
            sem = adapter.parse_column(c, nodes)
            if sem is None:
                continue
            by_name[c] = sem
            if sem.kind == adapter.noisy_kind:
                low_cols.append(c)
            else:
                high_cols.append(c)
        return cls(nodes=nodes, by_name=by_name, low_cols=tuple(low_cols),
                   high_cols=tuple(high_cols), adapter=adapter)

    def node_list(self) -> list:
        return sorted(self.nodes)
