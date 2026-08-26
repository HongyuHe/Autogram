"""Reset-aware state transitions for cumulative counter boundaries.

Autogram and the standalone GTIB generator each carry this helper. The
packages cannot depend on each other, so regression tests exercise both copies
against the same cases.
"""

from __future__ import annotations

import math

import numpy as np


def is_reset_marker(value) -> bool:
    """Return whether a scalar is an explicit truthy reset marker."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return math.isfinite(numeric) and abs(numeric) > 1e-9
    return str(value).strip().lower() in {
        "true",
        "1",
        "1.0",
        "yes",
        "t",
    }


def scan_counter_state(
    values: np.ndarray,
    reset_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan ordered counter samples.

    ``values`` has shape ``(n_samples, n_counters)``.  Non-finite samples are
    missing and retain the current lifetime value.  A reset clears every
    counter to zero before the sample on that row is applied, even when that
    sample is missing.

    Returns ``(effective_values, observed)`` where ``effective_values`` is the
    state after each row and ``observed`` marks finite samples.
    """

    raw = np.asarray(values, dtype=float)
    resets = np.asarray(reset_flags, dtype=bool)
    if raw.ndim != 2:
        raise ValueError("values must have shape (n_samples, n_counters)")
    if resets.shape != (raw.shape[0],):
        raise ValueError("reset_flags must have shape (n_samples,)")

    observed = np.isfinite(raw)
    effective = np.empty_like(raw, dtype=float)
    state = np.full(raw.shape[1], np.nan, dtype=float)
    for index in range(raw.shape[0]):
        if resets[index]:
            state.fill(0.0)
        present = observed[index]
        state[present] = raw[index, present]
        effective[index] = state
    return effective, observed


def trustworthy_boundaries(
    observed_in_bucket: np.ndarray,
    reset_in_bucket: np.ndarray,
) -> np.ndarray:
    """A boundary is trustworthy when its bucket observed a value or reset."""

    observed = np.asarray(observed_in_bucket, dtype=bool)
    resets = np.asarray(reset_in_bucket, dtype=bool)
    return observed | np.broadcast_to(resets, observed.shape)


def counter_boundary_deltas(
    current: np.ndarray,
    previous: np.ndarray,
    current_trustworthy: np.ndarray,
    previous_trustworthy: np.ndarray,
    reset_in_bucket: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute safe deltas between adjacent trustworthy boundaries.

    A bucket containing a reset never contributes a delta.  The reset still
    establishes the zero-based lifetime state used by the next trustworthy
    bucket.  Arithmetic is skipped for ineligible boundaries so extreme values
    across gaps or reset buckets cannot create a spurious overflow.
    """

    current_values = np.asarray(current, dtype=float)
    previous_values = np.asarray(previous, dtype=float)
    current_trust = np.asarray(current_trustworthy, dtype=bool)
    previous_trust = np.asarray(previous_trustworthy, dtype=bool)
    if not (
        current_values.shape
        == previous_values.shape
        == current_trust.shape
        == previous_trust.shape
    ):
        raise ValueError("counter boundary arrays must have identical shapes")

    resets = np.broadcast_to(
        np.asarray(reset_in_bucket, dtype=bool),
        current_values.shape,
    )
    eligible = current_trust & previous_trust & ~resets
    delta = np.full(current_values.shape, np.nan, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        np.subtract(
            current_values,
            previous_values,
            out=delta,
            where=eligible,
        )
    finite_operands = (
        np.isfinite(current_values)
        & np.isfinite(previous_values)
    )
    overflow = eligible & finite_operands & ~np.isfinite(delta)
    valid = eligible & np.isfinite(delta) & (delta >= 0.0)
    return delta, valid, overflow
