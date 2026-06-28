"""Dataset frames and dataset assembly (read-only, schema-general).

A :class:`Frame` is a dense column store (a ``(N, d)`` float matrix with a name index).  A
:class:`Dataset` bundles the observed frame, the parsed :class:`NameModel` (carrying the
induced schema adapter) and per-row timestamps.

There is no separate hidden "clean" oracle on the discovery path: ``Dataset.clean`` aliases
``observed`` so the data-only evaluator literally cannot read injected ground truth.  Datasets
are built either directly from a numeric matrix (:func:`build_dataset`, used by the synthetic
generator) or from an in-memory DataFrame whose cells are decoded by the schema adapter codec
(:func:`load_dataframe`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .names import NameModel


class Frame:
    """A column store: a dense ``(N, d)`` float matrix with a name index.

    Provides O(1) single-column access and vectorized multi-column sums, which the evaluator
    uses for cached family aggregation.
    """

    __slots__ = ("matrix", "name_to_idx", "names")

    def __init__(self, matrix: np.ndarray, names):
        self.matrix = matrix
        self.names = list(names)
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}

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
        return Frame(self.matrix[rows], self.names)


@dataclass
class Dataset:
    name: str
    name_model: NameModel
    observed: Frame
    timestamps: np.ndarray
    n_snapshots: int

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
        }


def build_dataset(columns, matrix: np.ndarray, adapter, name: str,
                  timestamps=None) -> Dataset:
    """Build a :class:`Dataset` directly from a numeric ``(N, d)`` matrix and an adapter.

    The columns are parsed through the induced ``adapter``; only columns the adapter recognises
    are kept (re-ordered to the engine's low-then-high convention).
    """
    matrix = np.asarray(matrix, dtype=float)
    nm = NameModel.from_columns_with_adapter(list(columns), adapter)
    ordered = list(nm.low_cols) + list(nm.high_cols)
    idx = [list(columns).index(c) for c in ordered]
    observed = Frame(matrix[:, idx] if idx else np.empty((matrix.shape[0], 0)), ordered)
    if timestamps is None:
        timestamps = np.arange(matrix.shape[0])
    return Dataset(name=name, name_model=nm, observed=observed,
                   timestamps=np.asarray(timestamps), n_snapshots=matrix.shape[0])


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
    ordered = list(nm.low_cols) + list(nm.high_cols)
    observed = Frame(_cells_to_matrix_adapter(df, ordered, nm), ordered)
    if timestamps is None:
        timestamps = (df["timestamp"].values if "timestamp" in df.columns
                      else np.arange(len(df)))
    return Dataset(name=name, name_model=nm, observed=observed,
                   timestamps=np.asarray(timestamps), n_snapshots=len(df))
