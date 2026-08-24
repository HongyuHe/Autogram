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
        if isinstance(value, list):
            return sum(TermCache._size(member) for member in value)
        return 0

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def __setitem__(self, key, value) -> None:
        value_size = self._size(value)
        if value_size > self.max_bytes:
            return
        if key in self._data:
            self.total_bytes -= self._size(self._data.pop(key))
        self._data[key] = value
        self.total_bytes += value_size
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
        "condition_cache",
        "temporal_cache",
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
        self.condition_cache = TermCache(
            max_entries=4_096,
            max_bytes=32 * 1024 * 1024,
        )
        self.temporal_cache = TermCache(
            max_entries=4_096,
            max_bytes=32 * 1024 * 1024,
        )

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


def _matrix_column_accessor(matrix, n_columns: int):
    """Return ``(n_rows, column_at)`` without promoting identity columns.

    NumPy arrays are homogeneous, so an already-float array may already have lost integer
    precision before reaching the loader.  Preserve an array's existing dtype, preserve pandas
    columns independently, and materialize other row-oriented inputs as objects so this function
    does not itself promote mixed integer/float rows before copying their identities.
    """
    if isinstance(matrix, pd.DataFrame):
        shape = matrix.shape

        def column_at(index):
            return matrix.iloc[:, index].to_numpy(copy=True)
    else:
        source = (
            np.asarray(matrix)
            if isinstance(matrix, np.ndarray)
            else np.asarray(matrix, dtype=object)
        )
        shape = source.shape

        def column_at(index):
            return np.array(source[:, index], copy=True)

    if len(shape) != 2:
        raise ValueError(
            f"dataset matrix must be two-dimensional, got shape {shape}"
        )
    if shape[1] != n_columns:
        raise ValueError(
            "dataset matrix column count does not match supplied columns "
            f"({shape[1]} != {n_columns})"
        )
    return int(shape[0]), column_at


def _is_missing_scalar(value) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _grounded_boolean_columns(nm: NameModel) -> dict[str, tuple[str, ...]]:
    """Resolve binder-scoped Boolean roles to their concrete runtime columns."""
    adapter = nm.adapter
    grounded = {}
    for binder in getattr(adapter, "binders", ()):
        roles = tuple(getattr(adapter, "boolean_roles", {}).get(binder, ()))
        columns = []
        seen = set()
        for binding in adapter.enumerate_bindings(binder, nm):
            for role in roles:
                column = adapter.resolve_ref(role, binder, binding, nm)
                if column is not None and column not in seen:
                    seen.add(column)
                    columns.append(column)
        grounded[binder] = tuple(columns)
    return grounded


def _declared_boolean_columns(
    nm: NameModel,
    grounded: dict[str, tuple[str, ...]] | None = None,
) -> set:
    """Resolve schema-declared Boolean conditions and ref roles to columns."""
    adapter = nm.adapter
    declared = {
        column
        for column, values in (
            getattr(adapter, "condition_columns", {}) or {}
        ).items()
        if values
        and all(isinstance(value, (bool, np.bool_)) for value in values)
    }
    for columns in (
        grounded
        if grounded is not None
        else _grounded_boolean_columns(nm)
    ).values():
        declared.update(columns)
    return declared


def _is_boolean_column(values, *, declared: bool = False) -> bool:
    """Recognize native Booleans without conflating them with 0/1 or text.

    A schema declaration retains an empty or all-missing Boolean column, but
    never overrides contradictory non-Boolean scalar contents.
    """
    if pd.api.types.is_bool_dtype(values):
        return True
    found = False
    for value in np.asarray(values, dtype=object).reshape(-1):
        if _is_missing_scalar(value):
            continue
        if not isinstance(value, (bool, np.bool_)):
            return False
        found = True
    return found or declared


def _column_to_float(values) -> np.ndarray:
    """Convert one measurement column, mapping nullable scalars to NaN."""
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        source = np.asarray(values, dtype=object)
        converted = np.empty(source.shape, dtype=float)
        for index in np.ndindex(source.shape):
            value = source[index]
            converted[index] = (
                np.nan if _is_missing_scalar(value) else float(value)
            )
        return converted


def build_dataset(columns, matrix: np.ndarray, adapter, name: str,
                  timestamps=None) -> Dataset:
    """Build a :class:`Dataset` directly from a matrix-like table and an adapter.

    The columns are parsed through the induced ``adapter``; only columns the adapter recognises
    are kept as float measurements (re-ordered to the engine's low-then-high convention).
    Declared grouping identities are copied from the source columns without float promotion.

    When the adapter declares a time index (and optionally grouping columns), the row context is
    populated the same way the DataFrame path does -- otherwise temporal terms would ground to zero
    points because ``_ordered_groups`` cannot find the declared time column in ``row_context``.
    """
    source_columns = list(columns)
    n_rows, column_at = _matrix_column_accessor(matrix, len(source_columns))
    nm = NameModel.from_columns_with_adapter(source_columns, adapter)
    positions = {}
    for index, column in enumerate(source_columns):
        positions.setdefault(column, index)
    source_cache = {}

    def source_column(column):
        if column not in source_cache:
            source_cache[column] = column_at(positions[column])
        return source_cache[column]

    time_index = getattr(adapter, "time_index", "") or ""
    group_keys = tuple(getattr(adapter, "group_keys", ()) or ())
    condition_columns = tuple(
        getattr(adapter, "condition_columns", {}) or ()
    )
    metadata_columns = tuple(
        getattr(adapter, "metadata_columns", ()) or ()
    )
    grounded_boolean_columns = _grounded_boolean_columns(nm)
    category_case_columns = tuple(
        column
        for binder in getattr(adapter, "binders", ())
        for column in grounded_boolean_columns.get(binder, ())
    )
    context_order = tuple(dict.fromkeys((
        time_index,
        *group_keys,
        *condition_columns,
        *category_case_columns,
        *metadata_columns,
    )))
    context_columns = set(context_order) - {""}
    declared_boolean_columns = _declared_boolean_columns(
        nm,
        grounded_boolean_columns,
    )
    ordered = [
        column
        for column in (list(nm.low_cols) + list(nm.high_cols))
        if (
            column not in context_columns
            or _is_boolean_column(
                source_column(column),
                declared=column in declared_boolean_columns,
            )
        )
    ]
    observed_matrix = np.empty((n_rows, len(ordered)), dtype=float)
    for target, column in enumerate(ordered):
        try:
            observed_matrix[:, target] = _column_to_float(
                source_column(column)
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"measurement column {column!r} cannot be converted to float"
            ) from error
    if timestamps is None:
        timestamps = (
            source_column(time_index)
            if time_index and time_index in positions
            else np.arange(n_rows)
        )
    timestamps = np.asarray(timestamps)
    row_context: dict = {}
    if time_index:
        row_context[time_index] = timestamps
    for key in context_order:
        if key and key != time_index and key in positions:
            row_context[key] = source_column(key)
    observed = Frame(
        observed_matrix,
        ordered,
        row_context=row_context,
    )
    return Dataset(name=name, name_model=nm, observed=observed,
                   timestamps=timestamps, n_snapshots=n_rows,
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
    grounded_boolean_columns = _grounded_boolean_columns(nm)
    category_case_columns = {
        column
        for columns in grounded_boolean_columns.values()
        for column in columns
    }
    declared_boolean_columns = _declared_boolean_columns(
        nm,
        grounded_boolean_columns,
    )
    metadata = {
        adapter.time_index,
        *adapter.group_keys,
        *adapter.condition_columns,
        *adapter.metadata_columns,
        *category_case_columns,
    } - {""}
    ordered = [
        c for c in (list(nm.low_cols) + list(nm.high_cols))
        if c not in metadata
        or _is_boolean_column(
            df[c],
            declared=c in declared_boolean_columns,
        )
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
