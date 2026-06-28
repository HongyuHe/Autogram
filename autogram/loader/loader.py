"""Dataset loading: build clean/observed numeric frames from the pickled samples.

Each cell of the CrossCheck sample DataFrames is a dict with ``hidden_ground_truth``
(clean) and ``ground_truth`` (noisy).  ``low_*`` cells carry both; ``high_*`` demand
cells carry only ``ground_truth`` (clean may be ``None``).

We build two frames:

* ``observed`` -- what a *deployed* learner sees: ``ground_truth`` for every column
  (noisy ``low_*`` + ``high_*``).  The proposer and the search operate on this.
* ``clean`` -- an *oracle* used only by the noise model / evaluator gate: clean
  ``hidden_ground_truth`` for ``low_*`` and ``ground_truth`` for ``high_*`` (no clean
  counterpart exists, so high demands are treated as noise-free for residual-bias
  detection, per design Sec. 5.2/5.5).

All access here is read-only; the pickles are never written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .names import NameModel


class Frame:
    """A column store: a dense ``(N, d)`` float matrix with a name index.

    Provides O(1) single-column access and vectorized multi-column sums, which
    the evaluator uses for cached locality-family aggregation (design Sec. 10.2).
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


@dataclass
class Dataset:
    name: str
    name_model: NameModel
    observed: Frame
    clean: Frame
    timestamps: np.ndarray
    n_snapshots: int

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


def _cells_to_matrix(df: pd.DataFrame, cols, primary: str, fallback: str) -> np.ndarray:
    n = len(df)
    mat = np.empty((n, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        vals = df[c].values
        col = mat[:, j]
        for i in range(n):
            v = vals[i]
            x = v.get(primary)
            if x is None:
                x = v.get(fallback)
            col[i] = np.nan if x is None else float(x)
    return mat


def load_dataset(path: str, name: str | None = None) -> Dataset:
    """Load one ``*.pkl`` sample into a :class:`Dataset` (read-only)."""
    df = pd.read_pickle(path)
    columns = list(df.columns)
    nm = NameModel.from_columns(columns)
    low = list(nm.low_cols)
    high = list(nm.high_cols)
    ordered = low + high

    # observed = noisy ground_truth everywhere
    obs_low = _cells_to_matrix(df, low, "ground_truth", "ground_truth")
    obs_high = _cells_to_matrix(df, high, "ground_truth", "ground_truth")
    observed = Frame(np.hstack([obs_low, obs_high]), ordered)

    # clean = hidden_ground_truth for low, ground_truth for high (no clean high)
    cln_low = _cells_to_matrix(df, low, "hidden_ground_truth", "ground_truth")
    clean = Frame(np.hstack([cln_low, obs_high]), ordered)

    ts = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))
    if name is None:
        name = "abilene" if "ATLAng" in nm.nodes else "geant"
    return Dataset(name=name, name_model=nm, observed=observed, clean=clean,
                   timestamps=np.asarray(ts), n_snapshots=len(df))


def _cells_to_matrix_adapter(df: pd.DataFrame, cols, nm: NameModel, which: str) -> np.ndarray:
    """Decode cells via a :class:`SchemaAdapter` codec (generalised path).

    ``which`` is ``"observed"`` or ``"clean"``; the per-column ``kind`` lets the codec
    decide which field is noisy (only ``noisy_kind`` columns get a clean counterpart).
    """
    adapter = nm.adapter
    n = len(df)
    mat = np.empty((n, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        kind = nm.by_name[c].kind
        vals = df[c].values
        col = mat[:, j]
        for i in range(n):
            v = vals[i]
            x = adapter.decode_observed(v) if which == "observed" \
                else adapter.decode_clean(v, kind)
            col[i] = np.nan if x is None else float(x)
    return mat


def load_dataframe(df: pd.DataFrame, adapter, name: str,
                   timestamps=None) -> Dataset:
    """Build a :class:`Dataset` from an in-memory frame via a compiled adapter.

    This is the schema-general counterpart of :func:`load_dataset`: the column
    semantics, low/high split, and cell decoding are all driven by ``adapter``
    (data, not hardcoded CrossCheck templates).  ``load_dataset`` is left byte-for-byte
    unchanged so the CrossCheck hot path never routes through here.
    """
    columns = list(df.columns)
    nm = NameModel.from_columns_with_adapter(columns, adapter)
    low = list(nm.low_cols)
    high = list(nm.high_cols)
    ordered = low + high

    obs_low = _cells_to_matrix_adapter(df, low, nm, "observed")
    obs_high = _cells_to_matrix_adapter(df, high, nm, "observed")
    observed = Frame(np.hstack([obs_low, obs_high]), ordered)

    cln_low = _cells_to_matrix_adapter(df, low, nm, "clean")
    clean = Frame(np.hstack([cln_low, obs_high]), ordered)

    if timestamps is None:
        timestamps = (df["timestamp"].values if "timestamp" in df.columns
                      else np.arange(len(df)))
    return Dataset(name=name, name_model=nm, observed=observed, clean=clean,
                   timestamps=np.asarray(timestamps), n_snapshots=len(df))
