"""Data loading and name semantics (read-only)."""

from .gtib import (
    AUTOGRAM_PROFILE_ATTR,
    infer_tabular_profile,
    prepare_gtib,
    prepare_gtib_files,
    prepare_gtib_raw,
)

__all__ = [
    "AUTOGRAM_PROFILE_ATTR",
    "infer_tabular_profile",
    "prepare_gtib",
    "prepare_gtib_files",
    "prepare_gtib_raw",
]
