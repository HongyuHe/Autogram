"""Preparation helpers for long/tidy GTIB telemetry tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import math

import numpy as np
import pandas as pd

from ..config import DiscoveryConfig


AUTOGRAM_PROFILE_ATTR = "autogram_profile"

_IDENTITY_COLUMNS = {"timestamp", "consumer_id", "shard_id", "minute_index"}
# Mirrors `schema/compiler._MAX_CONDITION_DOMAIN`. Inference must not propose a condition the
# compiler will reject, or an ordinary CSV becomes unloadable.
_MAX_CONDITION_DOMAIN = 64
_COUNTERS = {
    "collector_input_counted": "input_increment",
    "presenter_output_counted": "output_increment",
}
_BOUNDARY_COLUMNS = ("backlog_bytes", "cum_lost_bytes")

# Flag columns that are semantically Boolean but may arrive as bool, int, float, or string after a
# CSV/parquet round-trip. They are coerced to a real nullable Boolean at ingestion so downstream
# reductions (`.any()`), condition matching, Boolean-role detection, and null generation all agree;
# a naive pandas ``.any()`` on an object column treats the non-empty string ``"false"`` as truthy.
# The set spans every declared GTIB Boolean field across the raw, derived, and event tables.
_FLAG_COLUMNS = (
    "reset_flag",
    "missing_flag",
    "is_true_loss",
    "is_benign_burst",
    "is_artifact",
    "static_alert",
    "oracle_alert",
    "traj_alert",
)
_TRUE_TOKENS = frozenset({"true", "t", "1", "1.0", "yes", "y"})
_FALSE_TOKENS = frozenset({"false", "f", "0", "0.0", "no", "n"})


def _coerce_boolean_flag(series: pd.Series) -> pd.Series:
    """Coerce a semantically-Boolean flag column to a nullable Boolean dtype.

    Accepts native bool, numeric {0,1}, and case-insensitive true/false string tokens; any
    unrecognised, empty, or explicitly-missing value (``""``, ``"nan"``, ``"none"``, ``NaN``, or an
    out-of-range numeric) becomes ``pd.NA`` rather than being silently forced to a Boolean.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    def _one(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                return pd.NA
            if abs(numeric) < 1e-9:
                return False
            if abs(numeric - 1.0) < 1e-9:
                return True
            return pd.NA
        token = str(value).strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        return pd.NA

    return series.map(_one).astype("boolean")


def _coerce_flag_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame
    copied = False
    for column in _FLAG_COLUMNS:
        if column in out.columns and not pd.api.types.is_bool_dtype(out[column]):
            if not copied:
                out = out.copy()
                copied = True
            out[column] = _coerce_boolean_flag(out[column])
    return out



def profile_dataframe(
    frame: pd.DataFrame,
    *,
    time_index: str | None = None,
    group_keys: Iterable[str] = (),
    condition_columns: Iterable[str] = (),
    families: dict[str, Iterable[str]] | None = None,
    related_frames: dict[str, pd.DataFrame] | None = None,
    temporal_windows: Iterable[int] = (),
    max_lag: int = 0,
    related_aggregates: dict[str, dict] | None = None,
    run_lengths: Iterable[int] = (),
    advanced: bool = False,
    max_conjunction_terms: int = 3,
    metadata_columns: Iterable[str] = (),
    band_enabled: bool = False,
    max_degree: int = 0,
    proportional: bool = False,
    agg_kinds: Iterable[str] = (),
) -> pd.DataFrame:
    """Attach layout metadata without changing the table's observed values."""

    out = frame.copy()
    groups = [str(c) for c in group_keys if c in out.columns]
    conditions = [str(c) for c in condition_columns if c in out.columns]
    family_map = {
        str(name): [str(c) for c in columns if c in out.columns]
        for name, columns in (families or {}).items()
    }
    profile = {
        "time_index": time_index if time_index in out.columns else "",
        "group_keys": groups,
        "condition_columns": conditions,
        "families": family_map,
        "related_frames": dict(related_frames or {}),
        "temporal_windows": sorted({int(window) for window in temporal_windows}),
        "max_lag": int(max_lag),
        "related_aggregates": dict(related_aggregates or {}),
        "run_lengths": sorted({int(window) for window in run_lengths}),
        "advanced": bool(advanced),
        "max_conjunction_terms": int(max_conjunction_terms),
        "metadata_columns": [
            str(column)
            for column in metadata_columns
            if column in out.columns
        ],
        "band_enabled": bool(band_enabled),
        "max_degree": int(max_degree),
        "proportional": bool(proportional),
        "agg_kinds": [str(kind) for kind in agg_kinds],
    }
    out.attrs[AUTOGRAM_PROFILE_ATTR] = profile
    return out


def _min_condition_value_rows(n_rows: int, cfg: DiscoveryConfig | None = None) -> int:
    """Rows one condition value must cover before the column is worth proposing as a condition.

    Mirrors the evaluator's condition-support floor (``DiscoveryConfig.min_condition_points`` and
    ``min_condition_fraction``) rather than inventing a second, unrelated constant: a value that
    cannot clear that floor can never produce an accepted conditioned rule, so inferring it only
    multiplies the conditioned search space with candidates that are rejected by construction.
    Ingestion runs before any evaluation config is chosen, so the defaults apply unless a caller
    supplies the configuration the run will actually use.
    """
    floor = cfg or DiscoveryConfig()
    return max(
        int(floor.min_condition_points),
        int(math.ceil(float(floor.min_condition_fraction) * max(0, int(n_rows)))),
    )


def _is_regime_column(counts: pd.Series, n_rows: int, cfg: DiscoveryConfig | None = None) -> bool:
    """Does a column's value distribution look like a *regime label* rather than an identifier?

    The bar is read off the evaluator's own condition-support floor rather than a new constant:

    * Some value must clear the floor, so the column can actually yield an ACCEPTED conditioned
      rule.  A domain whose every stratum is too thin to be graded is pure search-space inflation --
      that is what 50 distinct values over 100 rows is, under the compiler's 64-value ceiling and
      not near-unique per row, yet an identifier repeated twice that generated a quarter of a
      million conditions.
    * The values that clear the floor must cover most of the table, so the column *describes* the
      data rather than labelling a corner of it.
    * The values that cannot clear the floor must not be numerous.  ``{common: 50, id-0..id-49: 1}``
      passes both tests above -- one fat stratum covering half the rows -- yet its fifty singleton
      values are an identifier tail, and expanding them projects over a million conditioned
      variants.  The bound is again the floor: at most ``n_rows // floor`` strata could each clear
      it, so a column carrying more thin values than that is not a regime label.

    Deliberately NOT ``counts.min() >= floor``, and deliberately not a bound on the domain size
    alone.  A genuine regime label often carries a rare value -- ``{normal: 80, alert: 39,
    unknown: 1}`` -- or several small ones beside a dominant stratum -- ``{normal: 50, r0..r4: 10}``
    -- and a thin stratum is no reason to discard a column whose other values are well populated.
    The thin strata's own conditioned rules are still rejected downstream by the support floor,
    which is where that decision belongs.
    """
    if counts.empty or n_rows <= 0:
        return False
    floor = _min_condition_value_rows(n_rows, cfg)
    eligible = counts[counts >= floor]
    if eligible.empty:
        return False
    if int(eligible.sum()) * 2 < n_rows:
        return False
    n_thin = int(counts.size) - int(eligible.size)
    return n_thin <= max(1, n_rows // max(1, floor))


def _typed_value_counts(series: pd.Series) -> pd.Series:
    """Value counts under Autogram's typed categorical identity."""
    from ..dsl.evaluate import typed_group_key

    counts: dict[tuple, int] = {}
    for value in series.to_numpy(dtype=object):
        key = typed_group_key(value)
        if key[0] == "missing":
            continue
        counts[key] = counts.get(key, 0) + 1
    return pd.Series(counts, dtype=np.int64)


def infer_tabular_profile(
    frame: pd.DataFrame,
    discovery_cfg: DiscoveryConfig | None = None,
) -> pd.DataFrame:
    """Attach conservative generic metadata inferred from common tabular conventions."""

    time_index = "timestamp" if "timestamp" in frame.columns else None
    groups = []
    for column in frame.columns:
        if column == time_index or not (
            column.endswith("_id") or column in {"consumer_id", "shard_id"}
        ):
            continue
        cardinality = int(_typed_value_counts(frame[column]).size)
        if 1 < cardinality and cardinality <= max(1, len(frame) // 3):
            groups.append(column)
    conditions = []
    for c in frame.columns:
        if c in _IDENTITY_COLUMNS:
            continue
        if not (
            pd.api.types.is_bool_dtype(frame[c])
            or isinstance(frame[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(frame[c])
            or pd.api.types.is_string_dtype(frame[c])
        ):
            continue
        # A condition has to name a *regime*, so its values must be able to carry evidence. The bar
        # is the evaluator's own condition-support floor, and the shape test lives in
        # `_is_regime_column`; a bounded-domain check alone is not enough.
        counts = _typed_value_counts(frame[c])
        distinct = int(counts.size)
        if distinct < 1 or distinct > _MAX_CONDITION_DOMAIN:
            continue
        if not _is_regime_column(counts, len(frame), discovery_cfg):
            continue
        conditions.append(c)
    return profile_dataframe(
        frame,
        time_index=time_index,
        group_keys=groups,
        condition_columns=conditions,
    )


def _infer_rate_window_seconds(derived: pd.DataFrame) -> int:
    """Infer the timestamp step represented by one increment of ``minute_index``."""
    from ..dsl.evaluate import _datetime_ns, typed_group_key

    from ..dsl.evaluate import typed_group_key

    times = _datetime_ns(derived["timestamp"].to_numpy())
    minutes = derived["minute_index"].to_numpy(dtype=np.int64)
    consumers = derived["consumer_id"].to_numpy(dtype=object)
    groups = {}
    for index, consumer in enumerate(consumers):
        groups.setdefault(typed_group_key(consumer), []).append(index)
    steps = set()
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: minutes[index])
        group_steps = set()
        for left, right in zip(ordered, ordered[1:]):
            delta_index = int(minutes[right]) - int(minutes[left])
            if delta_index == 0:
                continue
            delta_time = int(times[right]) - int(times[left])
            if delta_time % delta_index:
                raise ValueError(
                    "GTIB timestamps are not an integral cadence of minute_index"
                )
            step = delta_time // delta_index
            if step <= 0:
                raise ValueError(
                    "GTIB timestamps must increase with minute_index"
                )
            group_steps.add(step)
        if len(ordered) > 1 and not group_steps:
            raise ValueError(
                "GTIB consumer has no positive timestamp cadence"
            )
        if len(group_steps) > 1:
            raise ValueError(
                "GTIB consumer has inconsistent minute_index cadence"
            )
        steps.update(group_steps)
    if not steps:
        return 60
    if len(steps) != 1:
        raise ValueError(
            "GTIB timestamps imply inconsistent minute_index cadence"
        )
    nanoseconds = steps.pop()
    if nanoseconds % 1_000_000_000:
        raise ValueError(
            "GTIB rate cadence must be an integral number of seconds"
        )
    return nanoseconds // 1_000_000_000


def prepare_gtib(
    derived: pd.DataFrame,
    raw: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare GTIB derived rows and optionally materialize their raw-grain identities."""

    required = {"timestamp", "consumer_id", "minute_index"}
    missing = sorted(required - set(derived.columns))
    if missing:
        raise ValueError(f"GTIB derived table is missing required columns: {missing}")

    from ..dsl.evaluate import _datetime_ns, typed_group_key

    out = derived.reset_index(drop=True).copy()
    out["timestamp"] = _datetime_ns(
        out["timestamp"].to_numpy()
    ).view("datetime64[ns]")
    if out["timestamp"].isna().any():
        raise ValueError(
            "GTIB derived identities require nonmissing timestamps"
        )
    seen_identities = set()
    raw_minutes = out["minute_index"].to_numpy(dtype=object)
    for consumer, minute in zip(
        out["consumer_id"].to_numpy(dtype=object),
        raw_minutes,
    ):
        if (
            consumer is None
            or bool(pd.isna(consumer))
            or not isinstance(minute, (int, np.integer))
            or isinstance(minute, (bool, np.bool_))
            or int(minute) < 0
        ):
            raise ValueError(
                "GTIB derived identities require a nonmissing consumer_id "
                "and nonnegative integer minute_index"
            )
        identity = (typed_group_key(consumer), int(minute))
        if identity in seen_identities:
            raise ValueError(
                "GTIB derived table has duplicate consumer/minute row "
                f"for {consumer!r}, minute {minute}"
            )
        seen_identities.add(identity)
    out = _coerce_flag_columns(out)
    condition_columns = [
        c for c in out.columns
        if c not in _IDENTITY_COLUMNS
        and (
            pd.api.types.is_bool_dtype(out[c])
            or isinstance(out[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(out[c])
            or pd.api.types.is_string_dtype(out[c])
        )
    ]
    rate_window_seconds = _infer_rate_window_seconds(out)
    families: dict[str, list[str]] = {}
    related: dict[str, pd.DataFrame] = {}
    if raw is not None:
        raw = _coerce_flag_columns(raw)
        materialized, families = _materialize_raw(
            out,
            raw,
            rate_window_seconds=rate_window_seconds,
        )
        for name, values in materialized.items():
            out[name] = values
        related["raw"] = raw.copy()
    if events is not None:
        related["events"] = events.copy()
    related_aggregates = {}
    if raw is not None:
        common = {
            "relation": "raw",
            "parent_keys": ("consumer_id",),
            "child_keys": ("consumer_id",),
            "partition_keys": ("consumer_id", "shard_id"),
            "parent_time": "timestamp",
            "child_time": "timestamp",
            "window_seconds": rate_window_seconds,
        }
        related_aggregates = {
            "raw_input_rate": {
                **common,
                "column": "collector_input_counted",
                "mode": "sum_delta",
                "reset_column": "reset_flag",
                "validity_columns": (
                    "collector_input_counted",
                    "presenter_output_counted",
                ),
            },
            "raw_output_rate": {
                **common,
                "column": "presenter_output_counted",
                "mode": "sum_delta",
                "reset_column": "reset_flag",
                "validity_columns": (
                    "collector_input_counted",
                    "presenter_output_counted",
                ),
            },
        }
        if "backlog_bytes" in raw.columns:
            related_aggregates["raw_backlog"] = {
                **common,
                "column": "backlog_bytes",
                "mode": "sum_last",
            }
        if "cum_lost_bytes" in raw.columns:
            related_aggregates["raw_cum_lost"] = {
                **common,
                "column": "cum_lost_bytes",
                "mode": "sum_last",
            }
    if events is not None and not events.empty:
        event_types = tuple(str(value) for value in events["type"].dropna().unique())
        span_common = {
            "relation": "events",
            "column": "type",
            "mode": "span_any",
            "parent_keys": ("consumer_id",),
            "child_keys": ("consumer_id",),
            "partition_keys": (),
            "parent_time": "timestamp",
            "child_time": "span_start",
            "window_seconds": rate_window_seconds,
            "span_start": "span_start",
            "span_end": "span_end",
            "filter_column": "type",
        }
        related_aggregates.update({
            "event_true_loss": {
                **span_common,
                "filter_values": tuple(
                    value for value in event_types
                    if value.startswith("true_loss")
                ),
            },
            "event_benign_burst": {
                **span_common,
                "filter_values": ("benign_burst",),
            },
            "event_artifact": {
                **span_common,
                "filter_values": ("artifact",),
            },
        })

    return profile_dataframe(
        out,
        time_index="timestamp",
        group_keys=("consumer_id",),
        condition_columns=condition_columns,
        families=families,
        related_frames=related,
        temporal_windows=(10, 45, 60),
        max_lag=60,
        related_aggregates=related_aggregates,
        run_lengths=(10,),
        advanced=True,
        max_conjunction_terms=3,
        metadata_columns=("minute_index",),
        band_enabled=True,
        max_degree=2,
        proportional=True,
        agg_kinds=("SUM",),
    )


def prepare_gtib_files(
    derived_path: str | Path,
    raw_path: str | Path | None = None,
    events_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read GTIB CSV files and return the profiled, materialized derived table."""

    derived_path = Path(derived_path)
    if raw_path is None:
        candidate = derived_path.with_name("timeseries_raw.csv")
        raw_path = candidate if candidate.exists() else None
    if events_path is None:
        candidate = derived_path.with_name("events.csv")
        events_path = candidate if candidate.exists() else None
    derived = pd.read_csv(derived_path)
    raw = pd.read_csv(raw_path) if raw_path is not None else None
    events = pd.read_csv(events_path) if events_path is not None else None
    return prepare_gtib(derived, raw, events)


def prepare_gtib_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Profile the raw shard-grain table for grouped temporal discovery."""

    required = {
        "timestamp",
        "consumer_id",
        "shard_id",
        "collector_input_counted",
        "presenter_output_counted",
        "reset_flag",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"GTIB raw table is missing required columns: {missing}")
    from ..dsl.evaluate import _datetime_ns

    frame = raw.copy()
    frame["timestamp"] = _datetime_ns(
        frame["timestamp"].to_numpy()
    ).view("datetime64[ns]")
    frame = _coerce_flag_columns(frame)
    conditions = [
        column
        for column in ("missing_flag", "reset_flag")
        if column in frame.columns
    ]
    return profile_dataframe(
        frame,
        time_index="timestamp",
        group_keys=("consumer_id", "shard_id"),
        condition_columns=conditions,
        temporal_windows=(1,),
        max_lag=1,
    )


def _materialize_raw(
    derived: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    rate_window_seconds: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    # Local import avoids making loader initialisation depend on the DSL module. Identity-sensitive
    # grouping must use the same recursive rule as streaming related joins: pandas groupby merges
    # `True` with `1`, while string coercion also merges `1` with `"1"`.
    from ..dsl.evaluate import (
        _datetime_ns,
        typed_group_key,
        typed_sort_key,
    )

    def typed_display_map(raw_by_key: dict) -> dict:
        """Readable, deterministic, injective labels for typed identities.

        Ordinary string IDs retain their old spelling. Only colliding renderings are qualified, so
        existing GTIB column names and known catalogues remain byte-for-byte stable.
        """
        ordered = sorted(raw_by_key, key=typed_sort_key)
        plain = {key: str(raw_by_key[key]) for key in ordered}
        counts: dict[str, int] = {}
        for text in plain.values():
            counts[text] = counts.get(text, 0) + 1
        result = {}
        # Reserve every ordinary unambiguous spelling first. Otherwise an integer `1` may claim
        # the qualified name `int:1` before the literal string shard `"int:1"` is visited, forcing
        # the ordinary string to move even though its spelling was already unique.
        used = {
            plain[key]
            for key in ordered
            if counts[plain[key]] == 1
        }
        for key in ordered:
            if counts[plain[key]] == 1:
                result[key] = plain[key]
        for key in ordered:
            if key in result:
                continue
            raw_value = raw_by_key[key]
            base = f"{type(raw_value).__name__}:{raw_value!r}"
            candidate = base
            suffix = 2
            while candidate in used:
                candidate = f"{base}#{suffix}"
                suffix += 1
            used.add(candidate)
            result[key] = candidate
        return result

    def unique_group_names(group_order, preferred) -> dict:
        counts: dict[str, int] = {}
        for name in preferred.values():
            counts[name] = counts.get(name, 0) + 1
        result = {
            key: preferred[key]
            for key in group_order
            if counts[preferred[key]] == 1
        }
        # Reserve all unambiguous literal/preferred names before assigning suffixes to collisions.
        # Otherwise a duplicate `c1__s` may claim `c1__s#2` before the real `c1__s#2` shard is
        # visited, forcing the ordinary shard to move.
        used = set(result.values())
        for key in group_order:
            if key in result:
                continue
            base = preferred[key]
            candidate = base
            suffix = 2
            while candidate in used:
                candidate = f"{base}#{suffix}"
                suffix += 1
            used.add(candidate)
            result[key] = candidate
        return result

    required = {
        "timestamp",
        "consumer_id",
        "shard_id",
        "collector_input_counted",
        "presenter_output_counted",
        "reset_flag",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"GTIB raw table is missing required columns: {missing}")

    child = raw.copy()
    child["timestamp"] = _datetime_ns(
        child["timestamp"].to_numpy()
    ).view("datetime64[ns]")
    parent_lookup = {}
    parent_consumers = derived["consumer_id"].to_numpy(dtype=object)
    parent_minutes = derived["minute_index"].to_numpy(dtype=int)
    for position, (consumer, minute) in enumerate(
        zip(parent_consumers, parent_minutes)
    ):
        key = (typed_group_key(consumer), int(minute))
        if key in parent_lookup:
            raise ValueError(
                "GTIB derived table has duplicate consumer/minute row "
                f"for {consumer!r}, minute {minute}"
            )
        parent_lookup[key] = position
    # Which parent rows belong to each consumer, so a shard can mark its own consumer's minutes as
    # "no reading yet" without disturbing the structural zeros on other consumers' rows. Derived
    # from `parent_lookup` so the two can never index differently.
    _own: dict[tuple, list] = {}
    for (consumer, _minute), index in parent_lookup.items():
        _own.setdefault(consumer, []).append(index)
    own_rows: dict[tuple, np.ndarray] = {
        consumer: np.asarray(sorted(indices), dtype=int)
        for consumer, indices in _own.items()
    }
    # The streaming join accepts a parent row only when every partition is covered AND at least one
    # partition contributed a usable value (`complete & any_valid` in `dsl/evaluate.py`). Coverage
    # is expressible per shard column; "at least one usable contributor" is not, so it is collected
    # here and applied in a second pass. Without it a consumer whose only shard resets would read as
    # a total of 0 in the fast path while the join calls the row ungradeable.
    consumer_covered: dict[tuple, set] = {}
    consumer_any_valid: dict[tuple, set] = {}
    minute_ns = (
        int(rate_window_seconds)
        if rate_window_seconds is not None
        else _infer_rate_window_seconds(derived)
    ) * 1_000_000_000
    parent_timestamps = _datetime_ns(
        derived["timestamp"].to_numpy()
    )
    parent_start = {}
    parent_raw_by_key = {}
    inconsistent = []
    nat_ns = np.iinfo(np.int64).min
    for consumer, minute, timestamp_ns in zip(
        parent_consumers,
        parent_minutes,
        parent_timestamps,
    ):
        key = typed_group_key(consumer)
        parent_raw_by_key.setdefault(key, consumer)
        if timestamp_ns == nat_ns:
            raise ValueError(
                f"GTIB derived table has NaT timestamp for consumer {consumer!r}"
            )
        origin_ns = int(timestamp_ns) - int(minute) * minute_ns
        previous = parent_start.get(key)
        if previous is not None and previous != origin_ns:
            inconsistent.append(consumer)
        else:
            parent_start[key] = origin_ns
    if inconsistent:
        raise ValueError(
            "GTIB derived timestamps and minute_index values imply "
            f"inconsistent origins for consumers {sorted(map(repr, inconsistent))}"
        )

    materialized: dict[str, np.ndarray] = {}
    reserved_columns = {str(column) for column in derived.columns}
    allocated_columns = set()

    def allocate_column_name(preferred: str) -> str:
        """A generated column may neither alias another shard nor overwrite source data."""
        candidate = preferred
        suffix = 2
        while candidate in reserved_columns or candidate in allocated_columns:
            candidate = f"{preferred}#{suffix}"
            suffix += 1
        allocated_columns.add(candidate)
        return candidate

    families = {
        "shard_input_increment": [],
        "shard_output_increment": [],
        "shard_backlog_bytes": [],
        "shard_cum_lost_bytes": [],
    }

    child_consumers = child["consumer_id"].to_numpy(dtype=object)
    child_shards = child["shard_id"].to_numpy(dtype=object)
    groups: dict[tuple, list[int]] = {}
    raw_consumer_by_key = dict(parent_raw_by_key)
    raw_shard_by_key = {}
    shard_owners: dict[tuple, set] = {}
    for position, (consumer, shard) in enumerate(
        zip(child_consumers, child_shards)
    ):
        consumer_key = typed_group_key(consumer)
        shard_key = typed_group_key(shard)
        raw_consumer_by_key.setdefault(consumer_key, consumer)
        raw_shard_by_key.setdefault(shard_key, shard)
        groups.setdefault((consumer_key, shard_key), []).append(position)
        shard_owners.setdefault(shard_key, set()).add(consumer_key)

    group_order = sorted(
        groups,
        key=lambda pair: tuple(typed_sort_key(part) for part in pair),
    )
    consumer_names = typed_display_map(raw_consumer_by_key)
    shard_names = typed_display_map(raw_shard_by_key)
    preferred_group_names = {
        (consumer_key, shard_key): (
            f"{consumer_names[consumer_key]}__{shard_names[shard_key]}"
            if len(shard_owners[shard_key]) > 1
            else shard_names[shard_key]
        )
        for consumer_key, shard_key in group_order
    }
    safe_group_names = unique_group_names(group_order, preferred_group_names)
    active_groups = set()

    for consumer_key, shard_key in group_order:
        group = child.iloc[groups[(consumer_key, shard_key)]]
        group = group.sort_values("timestamp", kind="stable").copy()
        if consumer_key not in parent_start:
            raise ValueError(
                "GTIB raw table references consumer absent from derived table: "
                f"{raw_consumer_by_key[consumer_key]!r}"
            )
        # `minute_index` is relative to the derived series' exact origin, which need not be aligned
        # to a wall-clock minute. Flooring shifts every materialized window while the streaming join
        # keeps the true `[parent_time, parent_time + 60s)` interval, making the two paths disagree.
        consumer_start_ns = parent_start[consumer_key]
        group_times = _datetime_ns(
            group["timestamp"].to_numpy()
        )
        present = group_times != nat_ns
        if not np.any(present):
            continue
        active_groups.add((consumer_key, shard_key))
        group = group.loc[present].copy()
        group_times = group_times[present]
        group["_minute_index"] = np.asarray(
            [
                (int(timestamp_ns) - consumer_start_ns) // minute_ns
                for timestamp_ns in group_times
            ],
            dtype=np.int64,
        )

        boundaries = group.groupby("_minute_index", sort=True, observed=True).tail(1)
        minute_ids = boundaries["_minute_index"].to_numpy(dtype=int)
        reset_by_minute = (
            group.groupby("_minute_index", sort=True, observed=True)["reset_flag"]
            .any()
            .reindex(minute_ids, fill_value=False)
            .to_numpy(dtype=bool)
        )

        counter_boundaries: dict[str, np.ndarray] = {}
        for counter in _COUNTERS:
            # Non-finite readings are missing data; forward fill only carries ``NaN``, so leaving an
            # infinity in place would treat it as a real reading. The streaming path applies the
            # same rule, and the two implementations of one cross-grain law must agree.
            numeric = pd.to_numeric(group[counter], errors="coerce")
            filled = numeric.mask(
                ~np.isfinite(numeric.to_numpy(dtype=float))
            ).ffill()
            counter_boundaries[counter] = (
                filled.groupby(group["_minute_index"], sort=True)
                .last()
                .reindex(minute_ids)
                .to_numpy(dtype=float)
            )
        adjacent = np.zeros(minute_ids.size, dtype=bool)
        if minute_ids.size > 1:
            adjacent[1:] = np.diff(minute_ids) == 1
        eligible = adjacent & ~reset_by_minute
        increments = {}
        for counter, values in counter_boundaries.items():
            previous = np.concatenate((values[:1], values[:-1]))
            delta = np.full(values.shape, np.nan, dtype=float)
            rows = np.flatnonzero(eligible)
            with np.errstate(over="ignore", invalid="ignore"):
                delta[rows] = values[rows] - previous[rows]
            overflow = (
                ~np.isfinite(delta)
                & np.isfinite(values)
                & np.isfinite(previous)
                & eligible
            )
            if np.any(overflow):
                raw_consumer = raw_consumer_by_key[consumer_key]
                raw_shard = raw_shard_by_key[shard_key]
                minute = int(minute_ids[int(np.flatnonzero(overflow)[0])])
                raise ValueError(
                    "GTIB counter subtraction overflowed float64 for "
                    f"consumer {raw_consumer!r}, shard {raw_shard!r}, "
                    f"counter {counter!r}, minute {minute}"
                )
            increments[counter] = delta
        valid = eligible.copy()
        for values in increments.values():
            valid &= np.isfinite(values) & (values >= 0.0)
        # The streaming join separates *coverage* from *validity*: a minute with no adjacent prior
        # boundary has no measurable increment at all (`coverage` in `dsl/evaluate.py`), whereas a
        # covered minute whose increment is unusable (a reset, a negative step) still counts as a
        # deliberate zero contribution. The materialised path has to draw the same line, or the two
        # implementations of one law disagree on exactly the awkward rows.
        covered = adjacent.copy()

        safe_shard = safe_group_names[(consumer_key, shard_key)]
        for counter, suffix in _COUNTERS.items():
            name = allocate_column_name(f"{safe_shard}_{suffix}")
            # Three-way, mirroring the streaming join. Other consumers' rows stay 0 (this shard
            # contributes nothing there). This consumer's minutes start NaN -- no coverage -- so a
            # minute the shard never reported, or one with no adjacent prior boundary, leaves the
            # cross-grain total ungradeable rather than silently counting as zero. A covered minute
            # whose increment is unusable is a deliberate 0, because the emitted per-minute value
            # excludes that shard too.
            values = np.zeros(len(derived), dtype=float)
            own = own_rows.get(consumer_key)
            if own is not None and own.size:
                values[own] = np.nan
            for minute, value, is_valid, is_covered in zip(
                minute_ids, increments[counter], valid, covered
            ):
                index = parent_lookup.get((consumer_key, int(minute)))
                if index is None or not is_covered:
                    continue
                values[index] = float(value) if is_valid else 0.0
                if is_valid:
                    consumer_any_valid.setdefault(consumer_key, set()).add(index)
                consumer_covered.setdefault(consumer_key, set()).add(index)
            materialized[name] = values
            families[f"shard_{suffix}"].append(name)

        for boundary_col in _BOUNDARY_COLUMNS:
            if boundary_col not in boundaries.columns:
                continue
            name = allocate_column_name(f"{safe_shard}_{boundary_col}")
            # Rows belonging to OTHER consumers stay 0: this shard genuinely contributes nothing
            # there, which is what makes the family sum per row a sum over that consumer's shards.
            # This consumer's own rows start as NaN -- "no reading" -- so a minute the shard never
            # reported, or reported non-finite, makes the cross-grain total ungradeable instead of
            # silently counting as zero. That matches the streaming path in `dsl/evaluate.py`,
            # which requires every partition to contribute before a total is accepted.
            values = np.zeros(len(derived), dtype=float)
            own = own_rows.get(consumer_key)
            if own is not None and own.size:
                values[own] = np.nan
            for minute, value in zip(
                minute_ids,
                pd.to_numeric(
                    boundaries[boundary_col],
                    errors="coerce",
                ).to_numpy(dtype=float),
            ):
                index = parent_lookup.get((consumer_key, int(minute)))
                if index is not None and np.isfinite(value):
                    values[index] = float(value)
            materialized[name] = values
            families[f"shard_{boundary_col}"].append(name)

    # Second pass for the join's "at least one usable contributor" rule: a minute every shard
    # covered but none could contribute carries no information, so the increment columns report it
    # as ungradeable instead of as a total of zero.
    increment_names = [
        name
        for family, columns in families.items()
        if family.endswith("_increment")
        for name in columns
    ]
    if increment_names:
        for consumer, covered_rows in consumer_covered.items():
            barren = covered_rows - consumer_any_valid.get(consumer, set())
            if not barren:
                continue
            rows = np.asarray(sorted(barren), dtype=int)
            for name in increment_names:
                column = materialized.get(name)
                if column is None:
                    continue
                own = own_rows.get(consumer)
                if own is None:
                    continue
                # Only blank the rows that belong to this consumer; other consumers' structural
                # zeros must stay untouched.
                column[np.intersect1d(rows, own, assume_unique=False)] = np.nan

    # A parent consumer with no raw partition is ungradeable for EVERY related family. Leaving the
    # generated shard columns at their cross-consumer structural zero would fabricate a total of
    # zero, while the streaming join correctly returns NaN because it finds no child partitions.
    raw_consumers = {
        consumer_key
        for consumer_key, _shard_key in active_groups
    }
    for consumer_key in set(own_rows) - raw_consumers:
        rows = own_rows[consumer_key]
        for column in materialized.values():
            column[rows] = np.nan

    families = {name: columns for name, columns in families.items() if columns}
    return materialized, families
