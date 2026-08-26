"""Invariants that follow from the generation process.

This module is the executable companion to the "Invariants" section of the
README. It defines the relations that *must* hold given how the data is
generated, and checks them on a produced dataset. Two tiers:

* **hard** -- structural guarantees of the model (byte conservation, monotone
  counters, non-negative backlog/rates). A failure here is a generator bug.
* **soft** -- statistical expectations of *normal operation* (smoothed ratio sits
  in the healthy band; benign bursts do not lose bytes; true-loss spans do lose
  bytes). These should hold but are checked with tolerances because the data is
  deliberately noisy.

``check_all`` returns a list of :class:`InvariantResult`; ``--validate`` in the
CLI prints them and fails the run if any *hard* invariant is violated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import EmulatorConfig


@dataclass
class InvariantResult:
    name: str
    tier: str            # "hard" | "soft"
    passed: bool
    detail: str


def _hard(name: str, ok: bool, detail: str) -> InvariantResult:
    return InvariantResult(name, "hard", bool(ok), detail)


def _soft(name: str, ok: bool, detail: str) -> InvariantResult:
    return InvariantResult(name, "soft", bool(ok), detail)


def _identity_missing(value: Any) -> bool:
    """Whether a scalar or any nested tuple component is missing."""

    if isinstance(value, tuple):
        return any(_identity_missing(component) for component in value)
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _typed_identity_key(value: Any) -> tuple[Any, Any] | None:
    """Return a hashable recursive identity that keeps scalar types distinct."""

    if _identity_missing(value):
        return None
    if isinstance(value, tuple):
        components = tuple(_typed_identity_key(item) for item in value)
        if any(component is None for component in components):
            return None
        return (tuple, components)
    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)
    elif isinstance(value, np.timedelta64):
        value = pd.Timedelta(value)
    elif isinstance(value, np.generic):
        value = value.item()
    if _identity_missing(value):
        return None
    key = (type(value), value)
    try:
        hash(key)
    except (TypeError, ValueError):
        return None
    return key


def _identity_equal(actual: Any, expected: Any) -> bool:
    """Compare one generated identity with typed, missing-safe semantics."""

    actual_key = _typed_identity_key(actual)
    expected_key = _typed_identity_key(expected)
    if actual_key is None or expected_key is None:
        return False
    try:
        equal = actual_key == expected_key
    except (TypeError, ValueError):
        return False
    return isinstance(equal, (bool, np.bool_)) and bool(equal)


def _identity_mask(values: np.ndarray, expected: Any) -> np.ndarray:
    """Rows whose typed identity equals ``expected``."""

    return np.fromiter(
        (_identity_equal(value, expected) for value in values),
        dtype=bool,
        count=len(values),
    )


def _exact_nan(value: Any) -> bool:
    """Whether ``value`` is specifically a floating-point NaN."""

    return (
        isinstance(value, (float, np.floating))
        and bool(np.isnan(value))
    )


def _raw_payload_equal(
    actual: Any,
    expected: Any,
    *,
    boolean: bool,
) -> bool:
    """Compare emitted raw payload values without coercing flags or missingness."""

    if boolean:
        return (
            isinstance(actual, (bool, np.bool_))
            and isinstance(expected, (bool, np.bool_))
            and bool(actual) == bool(expected)
        )

    actual_nan = _exact_nan(actual)
    expected_nan = _exact_nan(expected)
    if actual_nan or expected_nan:
        return actual_nan and expected_nan
    if (
        actual is None
        or actual is pd.NA
        or expected is None
        or expected is pd.NA
        or isinstance(actual, (bool, np.bool_))
        or isinstance(expected, (bool, np.bool_))
    ):
        return False
    try:
        equal = actual == expected
    except (TypeError, ValueError):
        return False
    return isinstance(equal, (bool, np.bool_)) and bool(equal)


def _independent_minute_counters(
    counter: np.ndarray,
    reset_flag: np.ndarray,
    spm: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently audit the shared trustworthy-boundary contract."""

    values = np.asarray(counter)
    resets = np.asarray(reset_flag, dtype=bool)
    n_shards, n_steps = values.shape
    n_minutes = n_steps // spm
    deltas = np.full((n_shards, n_minutes), np.nan, dtype=float)
    valid = np.zeros((n_shards, n_minutes), dtype=bool)

    for shard in range(n_shards):
        lifetime_value = np.nan
        previous_boundary = np.nan
        previous_trustworthy = False
        for minute in range(n_minutes):
            start = minute * spm
            stop = start + spm
            reset_in_minute = False
            boundary_trustworthy = False

            for step in range(start, stop):
                if resets[shard, step]:
                    lifetime_value = 0.0
                    reset_in_minute = True
                    boundary_trustworthy = True
                value = values[shard, step]
                if not np.isnan(value):
                    lifetime_value = value
                    if np.isfinite(value):
                        boundary_trustworthy = True

            comparison_boundary = (
                lifetime_value
                if minute == 0
                else previous_boundary
            )
            with np.errstate(invalid="ignore"):
                delta = lifetime_value - comparison_boundary
            deltas[shard, minute] = delta
            valid[shard, minute] = (
                minute > 0
                and previous_trustworthy
                and boundary_trustworthy
                and not reset_in_minute
                and np.isfinite(delta)
                and delta >= 0.0
            )
            previous_boundary = lifetime_value
            previous_trustworthy = boundary_trustworthy

    return deltas, valid


def _generated_derived_grid(
    cfg: EmulatorConfig,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, str]:
    """Validate the exact per-consumer identity grid emitted by the generator."""

    expected_consumers = int(cfg.scale.n_consumers)
    expected_rows = int(
        cfg.n_raw_steps // cfg.raw_steps_per_minute
    )
    expected_total_rows = expected_consumers * expected_rows
    start_ns = int(pd.Timestamp(cfg.time.start_timestamp).value)
    cadence_ns = int(cfg.time.rate_window_seconds) * 1_000_000_000
    expected_timestamps = (
        start_ns
        + np.arange(expected_rows, dtype=np.int64) * cadence_ns
    )
    issues: list[str] = []
    valid_records: list[dict[str, Any]] = []
    total_rows = 0
    seen_consumers: dict[tuple[Any, Any], int] = {}
    seen_row_identities: dict[
        tuple[tuple[Any, Any], int],
        tuple[int, int],
    ] = {}
    invalid_row_identities = 0
    duplicate_row_identities = 0
    first_duplicate_row: tuple[
        tuple[int, int],
        tuple[int, int],
        Any,
        int,
    ] | None = None

    if len(records) != expected_consumers:
        issues.append(
            f"consumer records={len(records)} (expected {expected_consumers})"
        )

    for record_index, rec in enumerate(records):
        frame = rec.get("frame") if isinstance(rec, dict) else None
        consumer = rec.get("consumer") if isinstance(rec, dict) else None
        consumer_id = getattr(consumer, "consumer_id", None)
        archetype = getattr(consumer, "archetype", None)
        consumer_key = _typed_identity_key(consumer_id)
        record_ok = True
        if consumer_key is None:
            issues.append(
                f"record {record_index} has missing or invalid consumer_id "
                f"{consumer_id!r}"
            )
            record_ok = False
        else:
            previous = seen_consumers.get(consumer_key)
            if previous is not None:
                issues.append(
                    f"record {record_index} has duplicate consumer_id "
                    f"{consumer_id!r} (first seen in record {previous})"
                )
                record_ok = False
            else:
                seen_consumers[consumer_key] = record_index

        if not isinstance(frame, pd.DataFrame):
            issues.append(f"record {record_index} has no derived DataFrame")
            continue

        total_rows += len(frame)
        if len(frame) != expected_rows:
            issues.append(
                f"record {record_index} rows={len(frame)} "
                f"(expected {expected_rows})"
            )
            record_ok = False

        missing_columns = [
            column
            for column in ("timestamp", "consumer_id", "minute_index")
            if column not in frame.columns
        ]
        if missing_columns:
            issues.append(
                f"record {record_index} missing identity columns "
                f"{missing_columns}"
            )
            record_ok = False
        if "archetype" not in frame.columns:
            issues.append(
                f"record {record_index} missing required archetype column"
            )
            record_ok = False

        if (
            "consumer_id" in frame.columns
            and "minute_index" in frame.columns
        ):
            row_consumers = frame["consumer_id"].to_numpy(dtype=object)
            row_minutes = frame["minute_index"].to_numpy(dtype=object)
            for row_position, (row_consumer, minute) in enumerate(
                zip(row_consumers, row_minutes)
            ):
                row_consumer_key = _typed_identity_key(row_consumer)
                minute_ok = (
                    isinstance(minute, (int, np.integer))
                    and not isinstance(minute, (bool, np.bool_))
                    and int(minute) >= 0
                )
                if row_consumer_key is None or not minute_ok:
                    invalid_row_identities += 1
                    continue
                row_identity = (row_consumer_key, int(minute))
                previous_row = seen_row_identities.get(row_identity)
                if previous_row is not None:
                    duplicate_row_identities += 1
                    if first_duplicate_row is None:
                        first_duplicate_row = (
                            previous_row,
                            (record_index, row_position),
                            row_consumer,
                            int(minute),
                        )
                else:
                    seen_row_identities[row_identity] = (
                        record_index,
                        row_position,
                    )

        if len(frame) == expected_rows and "minute_index" in frame.columns:
            minute_values = frame["minute_index"].to_numpy(dtype=object)
            minute_ok = all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and int(value) == position
                for position, value in enumerate(minute_values)
            )
            if not minute_ok:
                issues.append(
                    f"record {record_index} minute_index is not exactly "
                    f"0..{expected_rows - 1} in row order"
                )
                record_ok = False

        if len(frame) == expected_rows and "timestamp" in frame.columns:
            try:
                timestamps = pd.to_datetime(
                    frame["timestamp"].to_numpy(dtype=object),
                    errors="coerce",
                    utc=True,
                )
                timestamp_ns = np.asarray(
                    timestamps.as_unit("ns").asi8,
                    dtype=np.int64,
                )
                timestamp_ok = np.array_equal(
                    timestamp_ns,
                    expected_timestamps,
                )
            except (AttributeError, OverflowError, TypeError, ValueError):
                timestamp_ok = False
            if not timestamp_ok:
                issues.append(
                    f"record {record_index} timestamps do not equal "
                    "start + minute_index * rate_window_seconds"
                )
                record_ok = False

        if len(frame) == expected_rows and "consumer_id" in frame.columns:
            consumer_ok = consumer_key is not None and all(
                _identity_equal(value, consumer_id)
                for value in frame["consumer_id"].to_numpy(dtype=object)
            )
            if not consumer_ok:
                issues.append(
                    f"record {record_index} consumer_id rows do not match "
                    f"{consumer_id!r}"
                )
                record_ok = False

        if "archetype" in frame.columns:
            archetype_mismatches = sum(
                not _identity_equal(value, archetype)
                for value in frame["archetype"].to_numpy(dtype=object)
            )
            if archetype_mismatches:
                issues.append(
                    f"record {record_index} has {archetype_mismatches} "
                    "archetype rows that do not match "
                    f"{archetype!r}"
                )
                record_ok = False

        if record_ok:
            valid_records.append(rec)

    if total_rows != expected_total_rows:
        issues.append(
            f"derived rows={total_rows} (expected {expected_total_rows})"
        )
    if len(seen_consumers) != expected_consumers:
        issues.append(
            f"unique consumer identities={len(seen_consumers)} "
            f"(expected {expected_consumers})"
        )
    if invalid_row_identities:
        issues.append(
            f"{invalid_row_identities} global derived rows have a missing or "
            "invalid (consumer_id, minute_index) identity"
        )
    if duplicate_row_identities and first_duplicate_row is not None:
        first, duplicate, consumer_id, minute = first_duplicate_row
        issues.append(
            f"{duplicate_row_identities} duplicate global "
            "(consumer_id, minute_index) identities; first duplicate "
            f"{consumer_id!r}, minute {minute} occurs at records/rows "
            f"{first} and {duplicate}"
        )
    if len(seen_row_identities) != expected_total_rows:
        issues.append(
            f"unique global derived row identities="
            f"{len(seen_row_identities)} (expected {expected_total_rows})"
        )

    if issues:
        shown = "; ".join(issues[:8])
        if len(issues) > 8:
            shown += f"; plus {len(issues) - 8} more"
        return valid_records, False, shown
    return (
        valid_records,
        True,
        f"{expected_consumers} typed consumer identities each emit "
        f"{expected_rows} ordered rows on the configured cadence, with "
        "globally unique (consumer_id, minute_index) identities and the "
        "record archetype on every row.",
    )


def _generated_raw_grid(
    cfg: EmulatorConfig,
    records: list[dict[str, Any]],
    raw: pd.DataFrame | None,
) -> tuple[bool, str]:
    """Validate the exact typed shard grid emitted by the generator."""

    try:
        return _generated_raw_grid_impl(cfg, records, raw)
    except Exception as exc:
        return (
            False,
            "raw grid validation rejected malformed input without raising: "
            f"{type(exc).__name__}: {exc}",
        )


def _generated_raw_grid_impl(
    cfg: EmulatorConfig,
    records: list[dict[str, Any]],
    raw: pd.DataFrame | None,
) -> tuple[bool, str]:
    expected_consumers = int(cfg.scale.n_consumers)
    expected_steps = int(cfg.n_raw_steps)
    start_ns = int(pd.Timestamp(cfg.time.start_timestamp).value)
    cadence_ns = int(cfg.time.raw_scrape_seconds) * 1_000_000_000
    payload_specs = [
        (
            "collector_input_counted",
            "obs",
            "input_counted",
            False,
        ),
        (
            "presenter_output_counted",
            "obs",
            "output_counted",
            False,
        ),
        ("missing_flag", "obs", "missing_flag", True),
        ("reset_flag", "obs", "reset_flag", True),
    ]
    if cfg.output.include_hidden_state:
        payload_specs.extend([
            ("backlog_bytes", "phys", "backlog", False),
            (
                "cum_lost_bytes",
                "phys",
                "cum_true_loss",
                False,
            ),
        ])
    expected_columns = (
        "timestamp",
        "consumer_id",
        "shard_id",
        *(spec[0] for spec in payload_specs),
    )
    issues: list[str] = []
    expected_pairs: list[
        tuple[
            Any,
            Any,
            tuple[Any, Any] | None,
            tuple[Any, Any] | None,
        ]
    ] = []
    expected_consumer_keys: set[tuple[Any, Any]] = set()
    seen_consumers: dict[tuple[Any, Any], int] = {}
    expected_payload_blocks: dict[str, list[np.ndarray]] = {
        column: []
        for column, _, _, _ in payload_specs
    }
    payload_sources_ok = True

    if len(records) != expected_consumers:
        issues.append(
            f"consumer records={len(records)} (expected {expected_consumers})"
        )

    for record_index, rec in enumerate(records):
        consumer = rec.get("consumer") if isinstance(rec, dict) else None
        consumer_id = getattr(consumer, "consumer_id", None)
        consumer_key = _typed_identity_key(consumer_id)
        if consumer_key is None:
            issues.append(
                f"record {record_index} has missing or invalid raw consumer_id "
                f"{consumer_id!r}"
            )
        else:
            previous = seen_consumers.get(consumer_key)
            if previous is not None:
                issues.append(
                    f"record {record_index} has duplicate raw consumer_id "
                    f"{consumer_id!r} (first seen in record {previous})"
                )
            else:
                seen_consumers[consumer_key] = record_index
                expected_consumer_keys.add(consumer_key)

        shard_ids = getattr(consumer, "shard_ids", None)
        if isinstance(shard_ids, (str, bytes)) or shard_ids is None:
            issues.append(
                f"record {record_index} has invalid shard_ids "
                f"{shard_ids!r}"
            )
            continue
        try:
            shards = list(shard_ids)
        except (TypeError, ValueError):
            issues.append(
                f"record {record_index} has non-iterable shard_ids "
                f"{shard_ids!r}"
            )
            continue

        if not (
            int(cfg.scale.shards_min)
            <= len(shards)
            <= int(cfg.scale.shards_max)
        ):
            issues.append(
                f"record {record_index} shards={len(shards)} "
                f"(expected {cfg.scale.shards_min}.."
                f"{cfg.scale.shards_max})"
            )

        expected_shape = (len(shards), expected_steps)
        for column, owner_name, attribute, _ in payload_specs:
            owner = (
                rec.get(owner_name)
                if isinstance(rec, dict)
                else None
            )
            values = getattr(owner, attribute, None)
            if np.shape(values) != expected_shape:
                issues.append(
                    f"record {record_index} source {owner_name}."
                    f"{attribute} shape={np.shape(values)} (expected "
                    f"{expected_shape})"
                )
                payload_sources_ok = False
                continue
            expected_payload_blocks[column].append(
                np.asarray(values).reshape(-1)
            )

        seen_shards: dict[tuple[Any, Any], int] = {}
        for shard_index, shard_id in enumerate(shards):
            shard_key = _typed_identity_key(shard_id)
            if shard_key is None:
                issues.append(
                    f"record {record_index} shard {shard_index} has missing "
                    f"or invalid shard_id {shard_id!r}"
                )
            else:
                previous = seen_shards.get(shard_key)
                if previous is not None:
                    issues.append(
                        f"record {record_index} has duplicate typed shard_id "
                        f"{shard_id!r} at shard positions {previous} and "
                        f"{shard_index}"
                    )
                else:
                    seen_shards[shard_key] = shard_index
            expected_pairs.append(
                (
                    consumer_id,
                    shard_id,
                    consumer_key,
                    shard_key,
                )
            )

    expected_total_rows = len(expected_pairs) * expected_steps
    expected_pair_keys = {
        (consumer_key, shard_key)
        for _, _, consumer_key, shard_key in expected_pairs
        if consumer_key is not None and shard_key is not None
    }
    if len(expected_pair_keys) != len(expected_pairs):
        issues.append(
            f"unique expected typed consumer/shard identities="
            f"{len(expected_pair_keys)} (expected {len(expected_pairs)})"
        )

    if raw is None:
        if issues:
            return False, _raw_grid_detail(issues)
        return (
            True,
            f"{len(expected_consumer_keys)} typed consumers and "
            f"{len(expected_pairs)} unique typed consumer/shard identities "
            f"define {expected_total_rows} ordered raw rows on the configured "
            "scrape cadence.",
        )

    if not isinstance(raw, pd.DataFrame):
        issues.append(
            f"raw table has type {type(raw).__name__}, expected DataFrame"
        )
        return False, _raw_grid_detail(issues)

    if not cfg.output.write_raw and raw.empty:
        if issues:
            return False, _raw_grid_detail(issues)
        return (
            True,
            "raw output is disabled; the typed consumer/shard source grid is "
            "valid.",
        )

    if len(raw) != expected_total_rows:
        issues.append(
            f"raw rows={len(raw)} (expected {expected_total_rows})"
        )

    missing_identity_columns = [
        column
        for column in ("timestamp", "consumer_id", "shard_id")
        if column not in raw.columns
    ]
    missing_payload_columns = [
        column
        for column, _, _, _ in payload_specs
        if column not in raw.columns
    ]
    if missing_identity_columns:
        issues.append(
            "raw table missing identity columns "
            f"{missing_identity_columns}"
        )
    if missing_payload_columns:
        issues.append(
            "raw table missing payload columns "
            f"{missing_payload_columns}"
        )
    if tuple(raw.columns) != expected_columns:
        issues.append(
            f"raw schema columns={list(raw.columns)!r} "
            f"(expected {list(expected_columns)!r})"
        )
    if not raw.columns.is_unique:
        issues.append("raw schema contains duplicate column names")
    if (
        missing_identity_columns
        or missing_payload_columns
        or not raw.columns.is_unique
    ):
        return False, _raw_grid_detail(issues)

    timestamp_values = raw["timestamp"].to_numpy(dtype=object)
    consumer_values = raw["consumer_id"].to_numpy(dtype=object)
    shard_values = raw["shard_id"].to_numpy(dtype=object)
    try:
        timestamps = pd.to_datetime(
            timestamp_values,
            errors="coerce",
            utc=True,
            format="mixed",
        )
        timestamp_ns = np.asarray(
            timestamps.as_unit("ns").asi8,
            dtype=np.int64,
        )
    except (AttributeError, OverflowError, TypeError, ValueError):
        timestamp_ns = np.full(
            len(raw),
            np.iinfo(np.int64).min,
            dtype=np.int64,
        )

    nat_ns = np.iinfo(np.int64).min
    invalid_identities = 0
    timestamp_mismatches = 0
    alignment_mismatches = 0
    first_alignment_mismatch: tuple[int, Any, Any, Any, Any] | None = None
    seen_global: dict[
        tuple[int, tuple[Any, Any], tuple[Any, Any]],
        int,
    ] = {}
    duplicate_global = 0
    first_duplicate: tuple[int, int, Any, Any, Any] | None = None
    observed_consumer_keys: set[tuple[Any, Any]] = set()
    observed_pair_keys: set[
        tuple[tuple[Any, Any], tuple[Any, Any]]
    ] = set()

    for position, (timestamp, consumer_id, shard_id) in enumerate(
        zip(timestamp_ns, consumer_values, shard_values)
    ):
        consumer_key = _typed_identity_key(consumer_id)
        shard_key = _typed_identity_key(shard_id)
        identity_ok = (
            int(timestamp) != nat_ns
            and consumer_key is not None
            and shard_key is not None
        )
        if not identity_ok:
            invalid_identities += 1
        if consumer_key is not None:
            observed_consumer_keys.add(consumer_key)
        if consumer_key is not None and shard_key is not None:
            observed_pair_keys.add((consumer_key, shard_key))
        if identity_ok:
            identity = (int(timestamp), consumer_key, shard_key)
            previous = seen_global.get(identity)
            if previous is not None:
                duplicate_global += 1
                if first_duplicate is None:
                    first_duplicate = (
                        previous,
                        position,
                        timestamp_values[position],
                        consumer_id,
                        shard_id,
                    )
            else:
                seen_global[identity] = position

        if position >= expected_total_rows or expected_steps <= 0:
            continue
        block = position // expected_steps
        step = position % expected_steps
        expected_consumer, expected_shard, _, _ = expected_pairs[block]
        expected_timestamp = start_ns + step * cadence_ns
        if int(timestamp) != expected_timestamp:
            timestamp_mismatches += 1
        if (
            not _identity_equal(consumer_id, expected_consumer)
            or not _identity_equal(shard_id, expected_shard)
        ):
            alignment_mismatches += 1
            if first_alignment_mismatch is None:
                first_alignment_mismatch = (
                    position,
                    consumer_id,
                    shard_id,
                    expected_consumer,
                    expected_shard,
                )

    missing_consumers = expected_consumer_keys - observed_consumer_keys
    extra_consumers = observed_consumer_keys - expected_consumer_keys
    if missing_consumers or extra_consumers:
        issues.append(
            "raw consumer identities differ from records: "
            f"missing={len(missing_consumers)}, extra={len(extra_consumers)}"
        )
    missing_pairs = expected_pair_keys - observed_pair_keys
    extra_pairs = observed_pair_keys - expected_pair_keys
    if missing_pairs or extra_pairs:
        issues.append(
            "raw shard identities differ from records: "
            f"missing={len(missing_pairs)}, extra={len(extra_pairs)}"
        )
    if invalid_identities:
        issues.append(
            f"{invalid_identities} raw rows have a missing or invalid typed "
            "(timestamp, consumer_id, shard_id) identity"
        )
    if duplicate_global and first_duplicate is not None:
        first, duplicate, timestamp, consumer_id, shard_id = first_duplicate
        issues.append(
            f"{duplicate_global} duplicate global "
            "(timestamp, consumer_id, shard_id) identities; first duplicate "
            f"{timestamp!r}, {consumer_id!r}, {shard_id!r} occurs at rows "
            f"{first} and {duplicate}"
        )
    if len(seen_global) != expected_total_rows:
        issues.append(
            f"unique global raw row identities={len(seen_global)} "
            f"(expected {expected_total_rows})"
        )
    if timestamp_mismatches:
        issues.append(
            f"{timestamp_mismatches} raw timestamps do not equal start + "
            "step * raw_scrape_seconds in shard-major row order"
        )
    if alignment_mismatches and first_alignment_mismatch is not None:
        (
            position,
            consumer_id,
            shard_id,
            expected_consumer,
            expected_shard,
        ) = first_alignment_mismatch
        issues.append(
            f"{alignment_mismatches} raw rows violate consumer/shard row "
            f"order or identity alignment; first at row {position}: "
            f"got ({consumer_id!r}, {shard_id!r}), expected "
            f"({expected_consumer!r}, {expected_shard!r})"
        )

    if (
        payload_sources_ok
        and len(raw) == expected_total_rows
    ):
        boolean_columns = {
            column
            for column, _, _, boolean in payload_specs
            if boolean
        }
        for column, _, _, _ in payload_specs:
            blocks = expected_payload_blocks[column]
            if not blocks:
                expected_values = np.array([], dtype=object)
            else:
                expected_values = np.concatenate(blocks).astype(
                    object,
                    copy=False,
                )
            actual_values = raw[column].to_numpy(dtype=object)
            equal = np.fromiter(
                (
                    _raw_payload_equal(
                        actual,
                        expected,
                        boolean=column in boolean_columns,
                    )
                    for actual, expected in zip(
                        actual_values,
                        expected_values,
                    )
                ),
                dtype=bool,
                count=len(actual_values),
            )
            mismatch_positions = np.flatnonzero(~equal)
            if mismatch_positions.size:
                first = int(mismatch_positions[0])
                issues.append(
                    f"raw payload column {column!r} differs from generator "
                    f"records at {mismatch_positions.size} rows; first at "
                    f"row {first}: got {actual_values[first]!r}, expected "
                    f"{expected_values[first]!r}"
                )

    if issues:
        return False, _raw_grid_detail(issues)
    return (
        True,
        f"{len(expected_consumer_keys)} typed consumers and "
        f"{len(expected_pairs)} unique typed consumer/shard identities emit "
        f"{expected_total_rows} globally unique rows in exact shard-major "
        "order on the configured scrape cadence, with the complete raw schema "
        "and payload exactly matching generator records.",
    )


def _raw_grid_detail(issues: list[str]) -> str:
    shown = "; ".join(issues[:10])
    if len(issues) > 10:
        shown += f"; plus {len(issues) - 10} more"
    return shown


def check_all(cfg: EmulatorConfig, records: list[dict[str, Any]],
              events: pd.DataFrame,
              raw: pd.DataFrame | None = None) -> list[InvariantResult]:
    """Run every invariant, including the emitted raw table when supplied."""

    results: list[InvariantResult] = []
    raw_grid_ok, raw_grid_detail = _generated_raw_grid(
        cfg,
        records,
        raw,
    )
    results.append(_hard(
        "raw_identity_grid",
        raw_grid_ok,
        raw_grid_detail,
    ))
    derived_records, grid_ok, grid_detail = _generated_derived_grid(
        cfg,
        records,
    )
    results.append(_hard(
        "derived_identity_grid",
        grid_ok,
        grid_detail,
    ))

    # -- Hard: every physical state value is finite --------------------------
    physical_fields = (
        "cum_input",
        "cum_output_physical",
        "cum_true_loss",
        "backlog",
    )
    nonfinite_by_field = {field: 0 for field in physical_fields}
    for rec in records:
        phys = rec["phys"]
        for field in physical_fields:
            values = np.asarray(getattr(phys, field))
            nonfinite_by_field[field] += int(np.count_nonzero(
                ~np.isfinite(values)
            ))
    physical_nonfinite = sum(nonfinite_by_field.values())
    nonfinite_detail = ", ".join(
        f"{field}={count}"
        for field, count in nonfinite_by_field.items()
        if count
    ) or "none"
    results.append(_hard(
        "physical_state_is_finite",
        physical_nonfinite == 0,
        f"{physical_nonfinite} non-finite physical state values "
        f"({nonfinite_detail}).",
    ))

    # -- Hard: every physical byte state is non-negative --------------------
    minimum_by_field = {}
    for field in physical_fields:
        minima = []
        for rec in records:
            values = np.asarray(getattr(rec["phys"], field))
            if values.size and np.all(np.isfinite(values)):
                minima.append(float(np.min(values)))
        minimum_by_field[field] = (
            min(minima)
            if minima
            else float("nan")
        )
    physical_non_negative = (
        physical_nonfinite == 0
        and all(
            minimum >= -1e-9
            for minimum in minimum_by_field.values()
        )
    )
    results.append(_hard(
        "physical_state_is_non_negative",
        physical_non_negative,
        "minimum physical values: " + ", ".join(
            f"{field}={minimum:.3g}"
            for field, minimum in minimum_by_field.items()
        ),
    ))

    # -- Hard: physical byte conservation  I == O + Q + L  --------------------
    max_resid = 0.0
    max_relative = 0.0
    conservation_values_finite = physical_nonfinite == 0
    for rec in records:
        phys = rec["phys"]
        residual = np.asarray(phys.conservation_residual())
        cumulative_input = np.asarray(phys.cum_input)
        if (
            not residual.size
            or residual.shape != cumulative_input.shape
            or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(cumulative_input))
        ):
            conservation_values_finite = False
            continue
        max_resid = max(max_resid, float(np.max(residual)))
        relative = residual / np.maximum(
            np.abs(cumulative_input),
            1.0,
        )
        max_relative = max(
            max_relative,
            float(np.max(relative)),
        )
    if not conservation_values_finite:
        max_relative = float("inf")
    results.append(_hard(
        "physical_byte_conservation",
        conservation_values_finite and max_relative < 1e-6,
        f"max |I-(O+Q+L)| = {max_resid:.3g} bytes "
        f"(pointwise relative to input {max_relative:.2e}); "
        "input == output + backlog + true_loss must hold exactly."))

    # -- Hard: backlog non-negative ------------------------------------------
    backlogs = [
        np.asarray(rec["phys"].backlog)
        for rec in records
    ]
    backlog_values_finite = (
        bool(backlogs)
        and all(np.all(np.isfinite(v)) for v in backlogs)
    )
    min_backlog = (
        min(float(np.min(v)) for v in backlogs)
        if backlog_values_finite
        else float("nan")
    )
    results.append(_hard(
        "backlog_non_negative",
        backlog_values_finite and min_backlog >= -1e-9,
        f"min backlog = {min_backlog:.3g} bytes; the queue can never hold negative bytes."))

    # -- Hard: cumulative true loss non-decreasing ---------------------------
    worst = 0.0
    true_loss_values_finite = True
    for rec in records:
        cumulative_loss = np.asarray(rec["phys"].cum_true_loss)
        if not np.all(np.isfinite(cumulative_loss)):
            true_loss_values_finite = False
            continue
        d = np.diff(cumulative_loss, axis=1)
        worst = min(worst, float(d.min()) if d.size else 0.0)
    results.append(_hard(
        "true_loss_monotone_non_decreasing",
        true_loss_values_finite and worst >= -1e-6,
        f"min step in cumulative true loss = {worst:.3g}; loss can only accumulate."))

    # -- Hard: observed counters monotone within reset-free runs -------------
    bad = _counter_monotone_violations(records)
    results.append(_hard(
        "observed_counters_monotone_between_resets", bad == 0,
        f"{bad} negative counter steps outside a flagged reset (should be 0)."))

    # -- Hard: derived rates non-negative ------------------------------------
    neg_rates = 0
    for rec in derived_records:
        f = rec["frame"]
        for col in ("input_rate_bytes_per_min", "output_rate_bytes_per_min"):
            v = f[col].to_numpy()
            neg_rates += int(np.nansum(v < -1e-6))
    results.append(_hard(
        "derived_rates_non_negative", neg_rates == 0,
        f"{neg_rates} negative derived rate values (should be 0)."))

    masked_counter_not_nan = 0
    present_counter_nonfinite = 0
    counter_infinite = 0
    counter_shape_mismatches = 0
    derived_infinite = 0
    for rec in records:
        obs = rec["obs"]
        absent = np.asarray(obs.missing_flag) | ~np.asarray(
            obs.active_flag
        )
        for counter in (obs.input_counted, obs.output_counted):
            values = np.asarray(counter)
            if values.shape != absent.shape:
                counter_shape_mismatches += 1
                continue
            try:
                is_nan = np.isnan(values)
                is_finite = np.isfinite(values)
                is_infinite = np.isinf(values)
            except TypeError:
                counter_shape_mismatches += 1
                continue
            masked_counter_not_nan += int(np.count_nonzero(
                absent & ~is_nan
            ))
            present_counter_nonfinite += int(np.count_nonzero(
                ~absent & ~is_finite
            ))
            counter_infinite += int(np.count_nonzero(is_infinite))
    for rec in derived_records:
        numeric = rec["frame"].select_dtypes(include=[np.number])
        derived_infinite += int(np.count_nonzero(
            np.isinf(numeric.to_numpy(dtype=float))
        ))
    results.append(_hard(
        "reported_telemetry_is_finite",
        masked_counter_not_nan == 0
        and present_counter_nonfinite == 0
        and counter_infinite == 0
        and counter_shape_mismatches == 0
        and derived_infinite == 0,
        f"{masked_counter_not_nan} missing/inactive counter values were "
        f"not NaN, {present_counter_nonfinite} present counter values were "
        f"non-finite, {counter_infinite} counter values were infinite, "
        f"{counter_shape_mismatches} counter shapes/types were invalid, and "
        f"{derived_infinite} derived values were infinite.",
    ))

    # -- Hard: every emitted derived identity is reproducible ----------------
    derived_bad = 0
    static_bad = 0
    trajectory_bad = 0
    label_bad = 0

    def strict_boolean(series: pd.Series) -> tuple[np.ndarray, int]:
        raw = series.to_numpy(dtype=object)
        valid = np.fromiter(
            (
                isinstance(value, (bool, np.bool_))
                for value in raw
            ),
            dtype=bool,
            count=raw.size,
        )
        output = np.zeros(raw.size, dtype=bool)
        output[valid] = np.asarray(raw[valid], dtype=bool)
        return output, int(np.count_nonzero(~valid))

    for rec in derived_records:
        frame = rec["frame"]
        obs = rec["obs"]
        phys = rec["phys"]
        spm = cfg.raw_steps_per_minute
        win = cfg.minutes_per_smoothing_window
        minute_values = [
            _independent_minute_counters(
                counter,
                obs.reset_flag,
                spm,
            )
            for counter in (
                obs.input_counted,
                obs.output_counted,
            )
        ]
        common_valid = minute_values[0][1] & minute_values[1][1]
        rates = []
        for delta, _valid in minute_values:
            usable = np.where(common_valid, delta, np.nan)
            summed = np.nansum(usable, axis=0)
            rates.append(np.where(
                common_valid.any(axis=0),
                summed,
                np.nan,
            ))
        input_rate, output_rate = rates
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                input_rate > 0.0,
                output_rate / input_rate,
                np.nan,
            )
        ratio_1h = (
            pd.Series(output_rate)
            .rolling(win, min_periods=win)
            .sum()
            / pd.Series(input_rate)
            .rolling(win, min_periods=win)
            .sum()
        ).to_numpy()
        backlog = phys.backlog.reshape(
            phys.backlog.shape[0], -1, spm,
        )[:, :, -1].sum(axis=0)
        loss = phys.cum_true_loss.reshape(
            phys.cum_true_loss.shape[0], -1, spm,
        )[:, :, -1].sum(axis=0)
        expected_columns = {
            "input_rate_bytes_per_min": input_rate,
            "output_rate_bytes_per_min": output_rate,
            "completeness_ratio": ratio,
            "completeness_ratio_1h": ratio_1h,
            "backlog_bytes": backlog,
            "cum_lost_bytes": loss,
        }
        for column, expected_values in expected_columns.items():
            if not np.allclose(
                frame[column].to_numpy(dtype=float),
                expected_values,
                rtol=1e-12,
                atol=1e-9,
                equal_nan=True,
            ):
                derived_bad += 1
        below = np.where(
            np.isfinite(ratio_1h),
            ratio_1h < cfg.alerting.alert_threshold,
            False,
        )
        expected_static = np.zeros(len(frame), dtype=bool)
        run = 0
        for index, active in enumerate(below):
            run = run + 1 if active else 0
            expected_static[index] = (
                run >= cfg.alerting.alert_duration_minutes
            )
        static_actual, invalid = strict_boolean(
            frame["static_alert"]
        )
        static_bad += invalid + int(np.count_nonzero(
            static_actual != expected_static
        ))
        deficit = input_rate - output_rate
        low = np.where(
            np.isfinite(ratio_1h),
            ratio_1h < cfg.alerting.good_data_threshold,
            False,
        )
        deficit_sum = pd.Series(deficit).rolling(
            45,
            min_periods=45,
        ).sum().to_numpy()
        slope = pd.Series(ratio_1h).diff(45).to_numpy()
        expected_trajectory = (
            low
            & np.where(
                np.isfinite(deficit_sum),
                deficit_sum > 0.0,
                False,
            )
            & np.where(
                np.isfinite(slope),
                slope <= 0.0,
                True,
            )
        )
        trajectory_actual, invalid = strict_boolean(
            frame["traj_alert"]
        )
        trajectory_bad += invalid + int(np.count_nonzero(
            trajectory_actual != expected_trajectory
        ))
        timestamps = pd.to_datetime(frame["timestamp"])
        true_loss = np.zeros(len(frame), dtype=bool)
        benign = np.zeros(len(frame), dtype=bool)
        artifact = np.zeros(len(frame), dtype=bool)
        consumer_events = (
            events.loc[_identity_mask(
                events["consumer_id"].to_numpy(dtype=object),
                rec["consumer"].consumer_id,
            )]
            if not events.empty
            else events
        )
        for _, event in consumer_events.iterrows():
            interval_end = timestamps + pd.to_timedelta(
                cfg.time.rate_window_seconds,
                unit="s",
            )
            active = (
                (timestamps < pd.Timestamp(event["span_end"]))
                & (
                    interval_end
                    > pd.Timestamp(event["span_start"])
                )
            ).to_numpy(dtype=bool)
            event_type = str(event["type"])
            if event_type.startswith("true_loss"):
                true_loss |= active
            elif event_type == "benign_burst":
                benign |= active
            elif event_type == "artifact":
                artifact |= active
        for column, expected_mask in (
            ("is_true_loss", true_loss),
            ("is_benign_burst", benign),
            ("is_artifact", artifact),
        ):
            actual, invalid = strict_boolean(frame[column])
            label_bad += invalid + int(np.count_nonzero(
                actual != expected_mask
            ))
        expected_label = np.full(len(frame), "normal", dtype=object)
        expected_label[artifact] = "artifact"
        expected_label[benign] = "benign_burst"
        expected_label[true_loss] = "true_loss"
        label_bad += int(np.count_nonzero(
            frame["label"].to_numpy(dtype=object) != expected_label
        ))
        oracle, invalid = strict_boolean(frame["oracle_alert"])
        label_bad += invalid + int(np.count_nonzero(
            oracle != true_loss
        ))
        label_bad += int(np.count_nonzero(~_identity_mask(
            frame["consumer_id"].to_numpy(dtype=object),
            rec["consumer"].consumer_id,
        )))
    results.extend((
        _hard(
            "derived_signal_identities",
            derived_bad == 0,
            f"{derived_bad} emitted derived columns disagree with independent "
            "re-derivation.",
        ),
        _hard(
            "static_alert_definition",
            static_bad == 0,
            f"{static_bad} static-alert rows disagree with the sustained rule.",
        ),
        _hard(
            "trajectory_alert_definition",
            trajectory_bad == 0,
            f"{trajectory_bad} trajectory-alert rows disagree with the target rule.",
        ),
        _hard(
            "label_priority_definition",
            label_bad == 0,
            f"{label_bad} label/oracle rows disagree with priority identities.",
        ),
    ))

    # -- Soft: normal-operation ratio sits in the healthy band ---------------
    # Scoped to STEADY consumers and the *instantaneous* 1 min ratio: calm
    # healthy operation must look healthy. (The 1 h smoothed ratio is NOT used
    # here because its trailing window has memory -- a normal minute within an
    # hour after a true-loss event legitimately shows a depressed smoothed ratio;
    # that is real smoothing behaviour, not a generator fault.) Bursty tenants are
    # allowed to dip -- that dip IS the reported false-positive bug.
    normal_ratio = _collect_steady(
        derived_records,
        label="normal",
        col="completeness_ratio",
    )
    if normal_ratio.size:
        med = float(np.nanmedian(normal_ratio))
        lo = cfg.pipeline.healthy_ratio_mean - 0.03
        hi = 1.05
        results.append(_soft(
            "steady_normal_ratio_in_healthy_band", lo <= med <= hi,
            f"median steady-normal 1min ratio = {med:.4f}; expected ~[{lo:.3f}, {hi:.3f}] "
            f"around healthy_ratio_mean={cfg.pipeline.healthy_ratio_mean}."))
        frac_below = float(np.nanmean(normal_ratio < cfg.alerting.alert_threshold))
        results.append(_soft(
            "steady_normal_minutes_rarely_below_threshold", frac_below < 0.10,
            f"{frac_below:.1%} of steady-consumer normal minutes have 1min ratio < "
            f"{cfg.alerting.alert_threshold} (should be small; these would be spurious)."))
    else:
        results.append(_soft("steady_normal_ratio_in_healthy_band", True, "no steady-normal minutes."))

    # -- Soft: benign bursts lose no bytes; true loss accumulates ------------
    benign_ok, benign_detail = _event_loss_behaviour(
        cfg,
        derived_records,
        events,
        benign=True,
    )
    results.append(_soft("benign_events_do_not_lose_bytes", benign_ok, benign_detail))
    loss_ok, loss_detail = _event_loss_behaviour(
        cfg,
        derived_records,
        events,
        benign=False,
    )
    results.append(_soft("true_loss_events_accumulate_deficit", loss_ok, loss_detail))

    return results


def _counter_monotone_violations(records: list[dict[str, Any]]) -> int:
    bad = 0
    for rec in records:
        obs = rec["obs"]
        for counter in (obs.input_counted, obs.output_counted):
            n_shards, n = counter.shape
            for sh in range(n_shards):
                row = counter[sh]
                reset = obs.reset_flag[sh]
                prev = np.nan
                for t in range(n):
                    v = row[t]
                    if reset[t]:
                        prev = np.nan
                    if np.isnan(v):
                        continue
                    if not np.isnan(prev) and v < prev - 1.0:
                        bad += 1
                    prev = v
    return bad


def _collect(records: list[dict[str, Any]], mask_col: str, mask_val: str, col: str) -> np.ndarray:
    chunks = []
    for rec in records:
        f = rec["frame"]
        if mask_col in f:
            chunks.append(f.loc[f[mask_col] == mask_val, col].to_numpy())
    return np.concatenate(chunks) if chunks else np.array([])


def _collect_steady(records: list[dict[str, Any]], label: str, col: str) -> np.ndarray:
    """Collect a column over ``label`` minutes, steady-archetype consumers only."""

    chunks = []
    for rec in records:
        if rec["consumer"].archetype != "steady":
            continue
        f = rec["frame"]
        chunks.append(f.loc[f["label"] == label, col].to_numpy())
    return np.concatenate(chunks) if chunks else np.array([])


def _event_loss_behaviour(cfg: EmulatorConfig, records: list[dict[str, Any]], events: pd.DataFrame,
                          benign: bool) -> tuple[bool, str]:
    """Check that benign events add ~0 true-loss and true-loss events add >0."""

    if events.empty:
        return True, "no events to check."
    by_consumer = {
        consumer_key: rec
        for rec in records
        if (
            consumer_key := _typed_identity_key(
                rec["consumer"].consumer_id
            )
        ) is not None
    }
    event_consumers = events["consumer_id"].to_numpy(dtype=object)
    checked = 0
    violations = 0
    overlap_skipped = 0
    checked_by_type: dict[str, int] = {}
    matching_types = {
        str(value)
        for value in events["type"].unique()
        if (value in ("benign_burst", "artifact")) == benign
    }
    for _, ev in events.iterrows():
        is_benign = ev["type"] in ("benign_burst", "artifact")
        if is_benign != benign:
            continue
        event_consumer_key = _typed_identity_key(ev["consumer_id"])
        rec = (
            by_consumer.get(event_consumer_key)
            if event_consumer_key is not None
            else None
        )
        if rec is None:
            continue
        if benign:
            overlapping_loss = events[
                _identity_mask(event_consumers, ev["consumer_id"])
                & ~events["type"].isin(("benign_burst", "artifact"))
                & (events["span_start"] < ev["span_end"])
                & (events["span_end"] > ev["span_start"])
            ]
            if not overlapping_loss.empty:
                overlap_skipped += 1
                continue
        frame = rec["frame"]
        seg = frame[(frame["timestamp"] >= ev["span_start"]) & (frame["timestamp"] < ev["span_end"])]
        if len(seg) >= 2:
            delta_loss = float(
                seg["cum_lost_bytes"].iloc[-1]
                - seg["cum_lost_bytes"].iloc[0]
            )
            input_bytes = float(np.nansum(
                seg["input_rate_bytes_per_min"].to_numpy()
            ))
        else:
            start = np.datetime64(
                cfg.time.start_timestamp.replace("Z", "")
            )
            dt = int(cfg.time.raw_scrape_seconds)
            first = int(
                (np.datetime64(ev["span_start"]) - start)
                / np.timedelta64(dt, "s")
            )
            stop = int(
                (np.datetime64(ev["span_end"]) - start)
                / np.timedelta64(dt, "s")
            )
            first = max(0, first)
            stop = min(cfg.n_raw_steps, stop)
            if stop <= first:
                continue
            prior = max(0, first - 1)
            current = max(prior, stop - 1)
            phys = rec["phys"]
            delta_loss = float(
                np.sum(phys.cum_true_loss[:, current])
                - np.sum(phys.cum_true_loss[:, prior])
            )
            input_bytes = float(
                np.sum(phys.cum_input[:, current])
                - np.sum(phys.cum_input[:, prior])
            )
        checked += 1
        checked_by_type[str(ev["type"])] = (
            checked_by_type.get(str(ev["type"]), 0) + 1
        )
        if benign:
            if delta_loss > max(1.0, 1e-6 * input_bytes):
                violations += 1
        else:
            if delta_loss <= 0.0:
                violations += 1
    if checked == 0:
        return True, "no matching events with enough coverage."
    uncovered = sorted(
        event_type
        for event_type in matching_types
        if checked_by_type.get(event_type, 0) == 0
    )
    ok = violations == 0 and not uncovered
    kind = "benign" if benign else "true-loss"
    expect = "add ~0 lost bytes" if benign else "accumulate lost bytes"
    suffix = (
        f"; skipped {overlap_skipped} precedence-probe overlaps"
        if overlap_skipped else ""
    )
    coverage = ", ".join(
        f"{event_type}:{checked_by_type.get(event_type, 0)}"
        for event_type in sorted(matching_types)
    )
    if uncovered:
        suffix += f"; uncovered subtypes={uncovered}"
    return (
        ok,
        f"{violations}/{checked} {kind} events violated the expectation "
        f"that they {expect}; coverage={coverage}{suffix}.",
    )
