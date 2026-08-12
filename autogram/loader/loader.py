"""Dataset frames and dataset assembly (read-only, schema-general).

A :class:`Frame` is a dense numeric column store with optional row context and related frames. A :class:`Dataset` bundles that frame, the parsed :class:`NameModel`, timestamps, grouping metadata, and conditions.

There is no separate hidden "clean" oracle on the discovery path: ``Dataset.clean`` aliases
``observed`` so the data-only evaluator literally cannot read injected ground truth.  Datasets
are built either directly from a numeric matrix (:func:`build_dataset`, used by the synthetic
generator) or from an in-memory DataFrame whose cells are decoded by the schema adapter codec
(:func:`load_dataframe`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .names import NameModel


class TermCache:
    def __init__(
        self,
        max_entries: int = 4_096,
        max_bytes: int = 64 * 1024 * 1024,
    ):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.total_bytes = 0
        self._data = OrderedDict()

    @staticmethod
    def _size(value) -> int:
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, tuple):
            # Term evaluation caches ``(value, overflow_mask)`` pairs; charging only the first
            # member would under-count the cache and let it grow past ``max_bytes``.
            return sum(TermCache._size(member) for member in value)
        return 0

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def __setitem__(self, key, value) -> None:
        if key in self._data:
            self.total_bytes -= self._size(self._data.pop(key))
        self._data[key] = value
        self.total_bytes += self._size(value)
        while (
            len(self._data) > self.max_entries
            or self.total_bytes > self.max_bytes
        ):
            _old_key, old_value = self._data.popitem(last=False)
            self.total_bytes -= self._size(old_value)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
        self.total_bytes = 0


class Frame:
    """A column store: a dense ``(N, d)`` float matrix with a name index.

    Provides O(1) single-column access and vectorized multi-column sums, which the evaluator
    uses for cached family aggregation.
    """

    __slots__ = (
        "matrix",
        "name_to_idx",
        "names",
        "row_context",
        "relations",
        "related_cache",
        "related_index_cache",
        "term_cache",
    )

    def __init__(self, matrix: np.ndarray, names, row_context=None, relations=None):
        self.matrix = matrix
        self.names = list(names)
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.row_context = dict(row_context or {})
        self.relations = dict(relations or {})
        self.related_cache = {}
        self.related_index_cache = {}
        self.term_cache = TermCache()

    @property
    def n_rows(self) -> int:
        return self.matrix.shape[0]

    def has(self, name: str) -> bool:
        return name in self.name_to_idx

    def col(self, name: str) -> np.ndarray:
        return self.matrix[:, self.name_to_idx[name]]

    def sum_cols(self, names) -> np.ndarray:
        """Vectorized sum over a set of columns; empty set -> zeros."""
        idx = [self.name_to_idx[n] for n in names if n in self.name_to_idx]
        if not idx:
            return np.zeros(self.matrix.shape[0], dtype=float)
        return self.matrix[:, idx].sum(axis=1)

    def slice_rows(self, rows) -> "Frame":
        """A view-like Frame over a subset of rows (used for cross-split / temporal blocks)."""
        context = {name: np.asarray(values)[rows] for name, values in self.row_context.items()}
        return Frame(self.matrix[rows], self.names, row_context=context, relations=self.relations)


@dataclass
class Dataset:
    name: str
    name_model: NameModel
    observed: Frame
    timestamps: np.ndarray
    n_snapshots: int
    time_index: str = ""
    group_keys: tuple[str, ...] = ()
    row_context: dict[str, np.ndarray] = field(default_factory=dict)
    relations: dict[str, object] = field(default_factory=dict)

    @property
    def clean(self) -> Frame:
        """No hidden oracle on the discovery path: clean == observed."""
        return self.observed

    @property
    def columns(self):
        return self.observed.names

    def observable_summary(self) -> dict:
        """Leakage-safe summary handed to proposers (names/types only, no values)."""
        nm = self.name_model
        return {
            "dataset": self.name,
            "n_snapshots": self.n_snapshots,
            "n_columns": len(self.columns),
            "nodes": nm.node_list(),
            "n_low": len(nm.low_cols),
            "n_high": len(nm.high_cols),
            "time_index": self.time_index,
            "group_keys": list(self.group_keys),
        }


def build_dataset(columns, matrix: np.ndarray, adapter, name: str,
                  timestamps=None) -> Dataset:
    """Build a :class:`Dataset` directly from a numeric ``(N, d)`` matrix and an adapter.

    The columns are parsed through the induced ``adapter``; only columns the adapter recognises
    are kept (re-ordered to the engine's low-then-high convention).

    When the adapter declares a time index (and optionally grouping columns), the row context is
    populated the same way the DataFrame path does -- otherwise temporal terms would ground to zero
    points because ``_ordered_groups`` cannot find the declared time column in ``row_context``.
    """
    matrix = np.asarray(matrix, dtype=float)
    nm = NameModel.from_columns_with_adapter(list(columns), adapter)
    ordered = list(nm.low_cols) + list(nm.high_cols)
    idx = [list(columns).index(c) for c in ordered]
    if timestamps is None:
        timestamps = np.arange(matrix.shape[0])
    timestamps = np.asarray(timestamps)
    time_index = getattr(adapter, "time_index", "") or ""
    group_keys = tuple(getattr(adapter, "group_keys", ()) or ())
    row_context: dict = {}
    if time_index:
        row_context[time_index] = timestamps
    source_columns = list(columns)
    for key in group_keys:
        if key in source_columns:
            row_context[key] = matrix[:, source_columns.index(key)]
    observed = Frame(
        matrix[:, idx] if idx else np.empty((matrix.shape[0], 0)),
        ordered,
        row_context=row_context,
    )
    return Dataset(name=name, name_model=nm, observed=observed,
                   timestamps=timestamps, n_snapshots=matrix.shape[0],
                   time_index=time_index,
                   group_keys=group_keys,
                   row_context=row_context)


def _cells_to_matrix_adapter(df, cols, nm: NameModel) -> np.ndarray:
    """Decode DataFrame cells via the schema adapter codec (observed values only)."""
    adapter = nm.adapter
    n = len(df)
    mat = np.empty((n, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        vals = df[c].values
        col = mat[:, j]
        for i in range(n):
            x = adapter.decode_observed(vals[i])
            col[i] = np.nan if x is None else float(x)
    return mat


def load_dataframe(df, adapter, name: str, timestamps=None) -> Dataset:
    """Build a :class:`Dataset` from an in-memory DataFrame via a compiled adapter codec."""
    columns = list(df.columns)
    nm = NameModel.from_columns_with_adapter(columns, adapter)
    metadata = {
        adapter.time_index,
        *adapter.group_keys,
        *adapter.condition_columns,
        *adapter.metadata_columns,
    } - {""}
    ordered = [
        c for c in (list(nm.low_cols) + list(nm.high_cols))
        if c not in metadata
        or pd.api.types.is_bool_dtype(df[c])
    ]
    row_context = {
        c: df[c].to_numpy(copy=True)
        for c in metadata
        if c in df.columns
    }
    relations = {}
    try:
        from .gtib import AUTOGRAM_PROFILE_ATTR
        relations = dict(df.attrs.get(AUTOGRAM_PROFILE_ATTR, {}).get("related_frames", {}))
    except (AttributeError, TypeError):
        relations = {}
    observed = Frame(
        _cells_to_matrix_adapter(df, ordered, nm),
        ordered,
        row_context=row_context,
        relations=relations,
    )
    if timestamps is None:
        timestamps = (df["timestamp"].values if "timestamp" in df.columns
                     else np.arange(len(df)))
    return Dataset(name=name, name_model=nm, observed=observed,
                   timestamps=np.asarray(timestamps), n_snapshots=len(df),
                   time_index=adapter.time_index,
                   group_keys=tuple(adapter.group_keys),
                   row_context=row_context,
                   relations=relations)
