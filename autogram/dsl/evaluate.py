"""Vectorized grounding of DSL rules.

Grounding a :class:`~autogram.dsl.ast.Rule` against a data :class:`~autogram.loader.loader.Frame`
produces, for every (snapshot t, binding b), a left value ``L`` and a right value ``R``.  From
these we derive the *raw residual* ``rho = L - R`` and a *typed scale* ``s = max(|L|, |R|)`` (the
relative-error denominator; a small floor avoids division by zero).

This module is deliberately thin: it only turns a rule + frame into a residual population.  The
data-only evaluator (:mod:`autogram.discovery.evaluate`) consumes that population -- it fits the
tolerance band, reads the operating coverage, and runs the acceptance tests.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..loader.loader import Frame
from ..loader.names import NameModel
from . import ast as A
from . import binders as B


@dataclass
class Grounded:
    """A rule's residual population, computed once and reused across the evaluator."""
    rho: np.ndarray          # (n_points,) raw residuals  L - R
    scale: np.ndarray        # (n_points,) typed scale     max(|L|,|R|)
    left: np.ndarray         # (n_points,) left values
    right: np.ndarray        # (n_points,) right values
    n_bindings: int          # bindings that grounded successfully (in scope)
    n_candidates: int        # bindings attempted (for support denominator)
    degenerate: bool         # True if no in-scope bindings
    row_indices: np.ndarray  # source row for each pooled point
    condition_support: float = 1.0
    # Rows the rule can actually GRADE under its condition, and that count as a fraction of the
    # attempted population. ``condition_support`` above measures only the rows the condition
    # SELECTS, which overstates the evidence whenever operands are non-finite. Both are recorded
    # before subsampling so the coreset lever -- a statistical estimator for the hold rate -- can
    # never decide a support question.
    graded_points: int = 0
    graded_condition_support: float = 0.0
    # Rows on which the candidate's OWN arithmetic exceeded float64 and produced an infinity. These
    # are not missing data: dropping them silently would shrink the population the rule claims to
    # describe while leaving its reported support untouched, so they are counted here and the
    # evaluator refuses the candidate outright (see ``DiscoveryConfig.max_overflow_fraction``).
    overflow_points: int = 0
    overflow_fraction: float = 0.0
    # Rows the rule was offered under its condition, across every grounded binding. This is the
    # denominator every overflow fraction is measured against, including the one a fitted parameter
    # introduces after grounding, so the separate checks compose into one bound.
    attempted_points: int = 0
    # Pre-subsample operands and rows. A coreset is a statistical estimator for the hold rate and
    # must never decide a universal claim, so a guard over the full graded population (notably the
    # finite-arithmetic guard on post-fit arithmetic) reads these instead.
    full_left: np.ndarray | None = None
    full_right: np.ndarray | None = None
    full_row_indices: np.ndarray | None = None
    # Source rows where candidate-created arithmetic overflowed before the finite population was
    # formed. They remain excluded from numeric fitting, but their group identities must still
    # reach the per-group gate as failures.
    overflow_row_indices: np.ndarray | None = None
    # True only for an atomic sign bound ``x OP 0`` that holds tolerance-free on every grounded row,
    # computed on the FULL population BEFORE any subsampling so a sampled-out violation can never
    # spuriously mark the bound exact.
    raw_exact_sign: bool = False

    @property
    def n_points(self) -> int:
        return self.rho.shape[0]

    @property
    def support(self) -> float:
        """Fraction of attempted bindings x rows the rule actually GRADED (Sec. 10.1).

        Reported support must describe the population the rule was scored on, not the population it
        was offered: rows whose operands are undefined, and rows the rule's arithmetic overflowed,
        are both absent from the residual population and must not be counted as evidence.
        ``graded_condition_support`` already measures the graded rows as a fraction of all attempted
        rows (it subsumes ``condition_support``), so it is the correct row-level factor here.
        """
        if self.n_candidates == 0:
            return 0.0
        return (self.n_bindings / self.n_candidates) * self.graded_condition_support


def robust_median(values: np.ndarray) -> float:
    """Median of a finite sample that cannot itself overflow ``float64``.

    ``np.median`` averages the two central elements of an even-length sample, and that intermediate
    SUM overflows for values near the float64 ceiling.  The result feeds the relative-residual scale
    floor, so an infinite "median" floors every scale at infinity, drives every relative residual to
    zero, and accepts every candidate with a perfect hold rate -- a false discovery manufactured by
    a numerical artefact.  The fallback averages the two central order statistics as
    ``lo + (hi - lo) / 2``, which is exact for finite inputs and never leaves their range.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        med = float(np.median(values))
    if np.isfinite(med):
        return med
    lo = float(np.quantile(values, 0.5, method="lower"))
    hi = float(np.quantile(values, 0.5, method="higher"))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        # The sample itself carries an infinity; there is no finite central value to report.
        finite = values[np.isfinite(values)]
        return robust_median(finite) if finite.size else 1.0
    return lo + (hi - lo) / 2.0


def _union_overflow(*masks):
    """OR together the overflow masks of a node's operands (``None`` means "no overflow")."""
    out = None
    for mask in masks:
        if mask is None:
            continue
        out = mask.copy() if out is None else (out | mask)
    return out if out is not None and out.any() else None


def _blowup(result, *inputs):
    """Rows where FINITE operands produced an infinite result -- the expression itself overflowed.

    This is deliberately distinct from a ``NaN`` result.  A ``NaN`` means the term is *undefined*
    here (missing data, or a guarded division by exact zero), which is a property of the data and
    is legitimately dropped from the graded population.  An infinity produced from finite operands
    means the candidate's own arithmetic exceeded ``float64`` -- a property of the *candidate*.
    Silently dropping those rows would shrink the population a rule claims to describe while
    leaving its reported support untouched, which is a false-discovery path.

    ``inf - inf`` and ``inf / inf`` yield ``NaN`` from ALREADY-overflowed operands; those rows are
    caught by the operand's own mask and propagated by :func:`_union_overflow`, so testing for a
    non-finite result here would only misclassify the guarded ``NaN`` cases.
    """
    if result is None:
        return None
    bad = np.isinf(result)
    if not bad.any():
        return None
    for value in inputs:
        bad &= np.isfinite(value)
        if not bad.any():
            return None
    return bad


def _pairwise_row_sum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Use the historical NumPy reduction and report finite-member overflow."""
    matrix = np.asarray(matrix, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        output = matrix.sum(axis=1)
    finite_members = np.all(np.isfinite(matrix), axis=1)
    overflow = ~np.isfinite(output) & finite_members
    return output, (overflow if overflow.any() else None)


def _canonical_row_sum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Use padding-independent accurate summation for related-grain parity."""
    matrix = np.asarray(matrix, dtype=float)
    finite_members = np.all(np.isfinite(matrix), axis=1)
    output = np.full(matrix.shape[0], np.nan, dtype=float)
    overflow = np.zeros(matrix.shape[0], dtype=bool)
    for row in range(matrix.shape[0]):
        if not finite_members[row]:
            with np.errstate(over="ignore", invalid="ignore"):
                output[row] = np.sum(matrix[row])
            continue
        try:
            output[row] = math.fsum(matrix[row].tolist())
        except OverflowError:
            output[row] = np.inf
            overflow[row] = True
    overflow = ~np.isfinite(output) & finite_members
    return output, (overflow if overflow.any() else None)


def _shift_overflow(mask, steps: int, frame: Frame, nm: NameModel):
    """Carry an overflow mask through a lag, so a blown-up row still taints its shifted reader."""
    if mask is None:
        return None
    shifted = _lag(mask.astype(float), steps, frame, nm)
    if shifted is None:
        return None
    return np.nan_to_num(shifted, nan=0.0) > 0.0


def _window_overflow(mask, window: int, frame: Frame, nm: NameModel):
    """Carry an overflow mask through a rolling window (any tainted member taints the window)."""
    if mask is None:
        return None
    rolled, _written = _rolling(mask.astype(float), window, "MAX", frame, nm)
    if rolled is None:
        return None
    return np.nan_to_num(rolled, nan=0.0) > 0.0


def eval_term(term: A.Term, binder: str, binding: dict, frame: Frame,
              nm: NameModel):
    """Evaluate a term for one binding -> (N,) array, or ``None`` if out of scope."""
    return eval_term_overflow(term, binder, binding, frame, nm)[0]


def eval_term_overflow(term: A.Term, binder: str, binding: dict, frame: Frame,
                       nm: NameModel):
    """Evaluate a term and the rows on which its own arithmetic overflowed.

    Returns ``(value, overflow)``.  ``value`` is ``None`` when the term is out of scope, and
    ``overflow`` is either ``None`` (nothing overflowed) or a boolean row mask.
    """
    key = (
        binder,
        tuple(sorted(binding.items())),
        term,
    )
    if key not in frame.term_cache:
        pair = _eval_term_uncached(
            term,
            binder,
            binding,
            frame,
            nm,
        )
        frame.term_cache[key] = pair
        return pair
    return frame.term_cache[key]


def _eval_term_uncached(term: A.Term, binder: str, binding: dict, frame: Frame,
                        nm: NameModel):
    if isinstance(term, A.Const):
        return np.full(frame.n_rows, float(term.value)), None
    if isinstance(term, A.Ref):
        col = B.resolve_ref(term.role, binder, binding, nm)
        if col is None or not frame.has(col):
            return None, None
        # A non-finite value already present in the DATA is missing/undefined, not an overflow of
        # this rule's arithmetic, so a leaf never originates an overflow mask.
        return frame.col(col), None
    if isinstance(term, A.Scale):
        inner, inner_overflow = eval_term_overflow(term.term, binder, binding, frame, nm)
        if inner is None:
            return None, None
        with np.errstate(over="ignore", invalid="ignore"):
            out = term.coeff * inner
        return out, _union_overflow(inner_overflow, _blowup(out, inner))
    if isinstance(term, A.Add):
        acc = np.zeros(frame.n_rows)
        overflow = None
        for t in term.terms:
            v, v_overflow = eval_term_overflow(t, binder, binding, frame, nm)
            if v is None:
                return None, None
            with np.errstate(over="ignore", invalid="ignore"):
                nxt = acc + v
            overflow = _union_overflow(overflow, v_overflow, _blowup(nxt, acc, v))
            acc = nxt
        return acc, overflow
    if isinstance(term, A.Agg):
        cols = B.resolve_family(term.family_role, binder, binding, nm)
        if not cols:
            return None, None
        mat = np.stack([frame.col(c) for c in cols], axis=1)
        if term.kind not in ("SUM", "AVG", "MIN", "MAX"):
            raise TypeError(f"unknown term {term!r}")
        finite_members = np.all(np.isfinite(mat), axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            if term.kind == "SUM":
                reducer = (
                    _canonical_row_sum
                    if term.family_role.startswith("shard_")
                    else _pairwise_row_sum
                )
                out, reduction_overflow = reducer(mat)
            elif term.kind == "AVG":
                out = mat.mean(axis=1)
                reduction_overflow = None
            elif term.kind == "MIN":
                out = mat.min(axis=1)
                reduction_overflow = None
            else:
                out = mat.max(axis=1)
                reduction_overflow = None
        out = np.where(finite_members, out, np.nan)
        # SUM and AVG can return ``NaN`` from finite members when an intermediate partial sum
        # overflows and then cancels, so every non-finite result counts; MIN and MAX cannot.
        blown = (
            ~np.isfinite(out) if term.kind in ("SUM", "AVG") else np.isinf(out)
        )
        overflow = _union_overflow(
            reduction_overflow,
            blown & finite_members,
        )
        return out, overflow
    if isinstance(term, A.Mul):
        left, left_overflow = eval_term_overflow(term.left, binder, binding, frame, nm)
        right, right_overflow = eval_term_overflow(term.right, binder, binding, frame, nm)
        if left is None or right is None:
            return None, None
        with np.errstate(over="ignore", invalid="ignore"):
            out = left * right
        return out, _union_overflow(left_overflow, right_overflow, _blowup(out, left, right))
    if isinstance(term, A.Div):
        num, num_overflow = eval_term_overflow(term.num, binder, binding, frame, nm)
        den, den_overflow = eval_term_overflow(term.den, binder, binding, frame, nm)
        if num is None or den is None:
            return None, None
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.where(den == 0.0, np.nan, num / den)
        out = np.where(
            np.isfinite(num) & np.isfinite(den),
            out,
            np.nan,
        )
        # Division by exact zero is deliberately ``NaN`` (undefined), and ``_blowup`` only fires on
        # an infinity, so the guarded case is never mistaken for an overflow.  A finite but tiny
        # denominator IS an overflow and is caught here.
        return out, _union_overflow(num_overflow, den_overflow, _blowup(out, num, den))
    if isinstance(term, A.Lag):
        inner, inner_overflow = eval_term_overflow(term.term, binder, binding, frame, nm)
        if inner is None:
            return None, None
        lagged = _lag(inner, term.steps, frame, nm)
        if lagged is None:
            return None, None
        return lagged, _shift_overflow(inner_overflow, term.steps, frame, nm)
    if isinstance(term, A.Diff):
        inner, inner_overflow = eval_term_overflow(term.term, binder, binding, frame, nm)
        if inner is None:
            return None, None
        lagged = _lag(inner, term.steps, frame, nm)
        if lagged is None:
            return None, None
        with np.errstate(over="ignore", invalid="ignore"):
            out = inner - lagged
        return out, _union_overflow(
            inner_overflow,
            _shift_overflow(inner_overflow, term.steps, frame, nm),
            _blowup(out, inner, lagged),
        )
    if isinstance(term, A.Rolling):
        inner, inner_overflow = eval_term_overflow(term.term, binder, binding, frame, nm)
        if inner is None:
            return None, None
        rolled, written = _rolling(inner, term.window, term.kind, frame, nm)
        if rolled is None:
            return None, None
        # ``_rolling`` only emits a value when every window member is finite, so a NON-FINITE value
        # on a row it actually wrote can only have come from the reduction itself blowing up --
        # including the ``NaN`` that intermediate overflow and cancellation produce, which is not an
        # infinity and would otherwise pass as ordinary missing data.
        return rolled, _union_overflow(
            _window_overflow(inner_overflow, term.window, frame, nm),
            written & ~np.isfinite(rolled),
        )
    if isinstance(term, A.RelatedAgg):
        template = getattr(nm.adapter, "resolve_related", lambda *_: None)(term.role, binder)
        if template is None:
            return None, None
        joined, joined_overflow = _related_aggregate(template, frame)
        if joined is None:
            return None, None
        # An infinite result is a blow-up too, but most of them never reach the output: the
        # aggregation's own validity tests absorb them, which is why the mask is built inside it.
        return joined, _union_overflow(joined_overflow, np.isinf(joined))
    raise TypeError(f"unknown term {term!r}")


def _time_vector(frame: Frame, time_index: str) -> np.ndarray:
    """Return checked nanoseconds so ordering and cadence share one interpretation."""
    key = ("time_vector", time_index)
    if key not in frame.temporal_cache:
        frame.temporal_cache[key] = _datetime_ns(
            frame.row_context[time_index]
        )
    return frame.temporal_cache[key]


_MISSING_GROUP = "__missing__"


def canonical_typed_value(value):
    """Canonical Python representation of one categorical/group value."""
    if isinstance(value, tuple):
        return tuple(canonical_typed_value(item) for item in value)
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, np.timedelta64):
        return pd.Timedelta(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _typed_object_array(values) -> np.ndarray:
    """Object array that does not integerize NumPy temporal scalars."""
    array = np.asarray(values)
    if array.dtype.kind not in ("M", "m"):
        return np.asarray(values, dtype=object)
    output = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        output[index] = array[index]
    return output


def is_missing_scalar(value) -> bool:
    """Whether one scalar is a missing categorical/group value.

    Covers Python/NumPy NaNs, ``pd.NA``, ``NaT``, and ``None`` without treating a tuple/composite
    key as a vector of missingness flags.
    """
    if value is None:
        return True
    if isinstance(value, tuple):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def typed_group_key(label):
    """Identity of a group label that Python's ``==``/hashing does not collapse.

    ``True == 1`` and ``hash(True) == hash(1)`` -- and the same holds inside a composite key -- so
    bucketing by the raw label silently merges two genuinely different groups. Merging them here is
    not cosmetic: it lets a stratified subsample drop a group entirely, and a group that is dropped
    cannot fail the per-group gate.

    A missing label is normalised to one sentinel, because ``NaN != NaN``: keeping the raw value
    would fragment every missing-labelled row into a group of its own, and a group of one is split
    entirely into the evaluation half where it can neither be fitted nor meaningfully gated.
    """
    label = canonical_typed_value(label)
    if isinstance(label, tuple):
        return ("tuple", tuple(typed_group_key(item) for item in label))
    if is_missing_scalar(label):
        return ("missing", _MISSING_GROUP)
    return (type(label).__name__, label)


def typed_unique(values, *, drop_missing: bool = False) -> tuple:
    """Distinct values in first-seen order under :func:`typed_group_key`.

    pandas/NumPy uniqueness uses Python equality, where ``True == 1``. Condition domains and
    categorical targets must preserve that distinction or the grammar cannot even express the
    type-correct relation the evaluator is supposed to score.
    """
    seen = set()
    result = []
    for value in values:
        key = typed_group_key(value)
        if drop_missing and key[0] == "missing":
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(canonical_typed_value(value))
    return tuple(result)


def typed_equal(left, right) -> bool:
    """Type-sensitive scalar equality, recursive through tuple values."""
    return typed_group_key(left) == typed_group_key(right)


def typed_signature_value(value):
    """Tagged categorical value for persisted/recovery signatures.

    The outer tag tells the generic signature matcher this is an identity, not a fitted numeric
    threshold. Numeric tolerance must never turn category ``1.0`` into ``1.005``.
    """
    return ("category_value", typed_group_key(value))


def typed_binary_domain(values) -> bool:
    """Whether a typed domain is one Boolean/numeric-binary scalar type."""
    unique = typed_unique(values, drop_missing=True)
    if not unique:
        return False
    kinds = {typed_group_key(value)[0] for value in unique}
    if len(kinds) != 1:
        return False
    return all(
        (
            isinstance(value, numbers.Integral)
            and int(value) in (0, 1)
        )
        or (
            isinstance(value, numbers.Real)
            and not isinstance(value, numbers.Integral)
            and float(value) in (0.0, 1.0)
        )
        for value in unique
    )


def typed_condition_key(condition: A.Condition | None):
    """Canonical condition identity for solver, archive, and signatures."""
    if condition is None:
        return None
    if condition.op == "all":
        children = tuple(sorted(
            (
                typed_condition_key(child)
                for child in condition.values
                if isinstance(child, A.Condition)
            ),
            key=str,
        ))
        return ("all", children)
    values = tuple(
        sorted(
            (typed_group_key(value) for value in condition.values),
            key=typed_sort_key,
        )
    )
    return (condition.column, condition.op, values)


def typed_sort_key(typed):
    """Deterministic ordering for typed keys that preserves natural order within a type.

    Sorting by ``str`` alone reorders numerics lexicographically (``"10" < "9"``), and the order
    partitions are visited in decides the order their contributions are accumulated -- which changes
    the floating-point total. Ordering by type first and then by the value itself reproduces the
    natural ordering the previous ``groupby(sort=True)`` produced for a homogeneous column, and
    falls back to the rendered form only for values that cannot be compared.
    """
    if isinstance(typed, tuple) and typed and typed[0] == "tuple":
        return (
            1,
            "tuple",
            0,
            0.0,
            tuple(typed_sort_key(item) for item in typed[1]),
        )
    kind, value = typed
    try:
        if isinstance(value, numbers.Integral):
            return (0, kind, 0, int(value), "")
        if isinstance(value, numbers.Real):
            return (0, kind, 0, float(value), "")
        if isinstance(value, str):
            return (0, kind, 1, 0.0, value)
    except (TypeError, ValueError):
        pass
    return (0, kind, 2, 0.0, repr(value))


def _ordered_groups(frame: Frame, nm: NameModel):
    adapter = getattr(nm, "adapter", None)
    time_index = getattr(adapter, "time_index", "")
    if not time_index or time_index not in frame.row_context:
        return None
    times = _time_vector(frame, time_index)
    group_keys = tuple(getattr(adapter, "group_keys", ()))
    if group_keys and not all(key in frame.row_context for key in group_keys):
        return None
    cache_key = ("ordered_groups", time_index, group_keys)
    if cache_key in frame.temporal_cache:
        return frame.temporal_cache[cache_key]
    groups: dict[object, list[int]] = {}
    for row in range(frame.n_rows):
        if not group_keys:
            key = "__all__"
        elif len(group_keys) == 1:
            key = _typed_object_array(
                frame.row_context[group_keys[0]]
            )[row]
        else:
            key = tuple(
                _typed_object_array(frame.row_context[name])[row]
                for name in group_keys
            )
        groups.setdefault(typed_group_key(key), []).append(row)
    ordered = []
    for rows in groups.values():
        index = np.asarray(rows, dtype=int)
        order = np.argsort(times[index], kind="stable")
        ordered.append(index[order])
    frame.temporal_cache[cache_key] = ordered
    return ordered


def _consecutive_window_ends(
    frame: Frame,
    nm: NameModel,
    rows: np.ndarray,
    window: int,
) -> np.ndarray:
    adapter = getattr(nm, "adapter", None)
    time_index = getattr(adapter, "time_index", "")
    cache_key = (
        "consecutive",
        time_index,
        id(rows),
        int(window),
    )
    if cache_key in frame.temporal_cache:
        return frame.temporal_cache[cache_key]
    valid = np.zeros(rows.size, dtype=bool)
    if window <= 1:
        valid[:] = True
        frame.temporal_cache[cache_key] = valid
        return valid
    times = _datetime_ns(
        np.asarray(frame.row_context[time_index])[rows]
    )
    diffs = [
        (
            int(times[index + 1]) - int(times[index])
            if (
                times[index] != _NAT_NS
                and times[index + 1] != _NAT_NS
            )
            else None
        )
        for index in range(times.size - 1)
    ]
    positive = [delta for delta in diffs if delta is not None and delta > 0]
    if not positive:
        frame.temporal_cache[cache_key] = valid
        return valid
    cadence = min(positive)
    consecutive = np.asarray(
        [delta == cadence for delta in diffs],
        dtype=bool,
    )
    for end in range(window - 1, rows.size):
        valid[end] = bool(np.all(
            consecutive[end - window + 1:end]
        ))
    frame.temporal_cache[cache_key] = valid
    return valid


def _lag(values: np.ndarray, steps: int, frame: Frame, nm: NameModel):
    groups = _ordered_groups(frame, nm)
    if groups is None:
        return None
    out = np.full(frame.n_rows, np.nan, dtype=float)
    for rows in groups:
        if rows.size > steps:
            valid = _consecutive_window_ends(
                frame,
                nm,
                rows,
                steps + 1,
            )
            ends = np.flatnonzero(valid)
            out[rows[ends]] = values[rows[ends - steps]]
    return out


def _rolling(values: np.ndarray, window: int, kind: str, frame: Frame, nm: NameModel):
    """Rolling reduction -> ``(values, written)``.

    ``written`` marks the rows a value was actually computed for. Output ``NaN`` is ambiguous on its
    own -- it means either "no valid window here" (legitimate) or "the reduction itself produced
    NaN" (an overflow of the candidate's own arithmetic) -- and only the second is a blow-up.
    """
    groups = _ordered_groups(frame, nm)
    if groups is None:
        return None, None
    out = np.full(frame.n_rows, np.nan, dtype=float)
    written = np.zeros(frame.n_rows, dtype=bool)
    for rows in groups:
        ordered = np.asarray(values[rows], dtype=float)
        consecutive = _consecutive_window_ends(
            frame,
            nm,
            rows,
            window,
        )
        for end in range(window - 1, rows.size):
            if not consecutive[end]:
                continue
            chunk = ordered[end - window + 1:end + 1]
            if not np.all(np.isfinite(chunk)):
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                if kind == "SUM":
                    value = float(np.sum(chunk))
                elif kind == "AVG":
                    value = float(np.mean(chunk))
                elif kind == "MIN":
                    value = float(np.min(chunk))
                elif kind == "MAX":
                    value = float(np.max(chunk))
                else:
                    raise ValueError(f"unknown rolling aggregation {kind!r}")
            out[rows[end]] = value
            written[rows[end]] = True
    return out, written


def _sign_bound_raw_exact(atom, rho: np.ndarray) -> bool:
    """True iff ``atom`` is an atomic sign bound ``x OP 0`` that holds tolerance-free on every row.

    ``rho`` is the grounded residual ``left - right``; for a sign bound this is the column value
    (negated when the zero constant is on the left), so the recorded operator applied directly to
    rho reproduces the bound in either orientation.
    """
    if not isinstance(atom, A.Compare) or atom.op not in ("<", "<=", ">", ">="):
        return False
    left_zero = isinstance(atom.left, A.Const) and float(atom.left.value) == 0.0
    right_zero = isinstance(atom.right, A.Const) and float(atom.right.value) == 0.0
    if not (left_zero or right_zero):
        return False
    measured = atom.right if left_zero else atom.left
    if not isinstance(measured, A.Ref) or rho.size == 0:
        return False
    if atom.op == ">":
        return bool(np.all(rho > 0.0))
    if atom.op == ">=":
        return bool(np.all(rho >= 0.0))
    if atom.op == "<":
        return bool(np.all(rho < 0.0))
    return bool(np.all(rho <= 0.0))


def _row_group_keys(frame: Frame, nm: NameModel, rows: np.ndarray) -> np.ndarray | None:
    """Group label (as an object array) for each grounded row, or ``None`` when the data is not
    grouped. Used to keep subsampling from silently dropping an entire group."""
    adapter = getattr(nm, "adapter", None)
    group_keys = tuple(getattr(adapter, "group_keys", ()))
    if not group_keys or not all(key in frame.row_context for key in group_keys):
        return None
    columns = [
        _typed_object_array(frame.row_context[key])
        for key in group_keys
    ]
    if len(columns) == 1:
        return columns[0][rows]
    # A composite key must stay a ONE-dimensional object array whose elements are tuples. Passing a
    # list of tuples to ``np.array(..., dtype=object)`` builds a 2-D array instead, whose ``tolist()``
    # yields unhashable lists and breaks group bucketing, so fill an empty 1-D array element-wise.
    labels = np.empty(rows.size, dtype=object)
    for position, row in enumerate(rows):
        labels[position] = tuple(col[row] for col in columns)
    return labels


def _stratified_subsample(rows: np.ndarray, frame: Frame, nm: NameModel,
                          subsample: int, seed: int) -> np.ndarray | None:
    """Indices to keep for a group-stratified subsample of ``rows``.

    Pooled random subsampling can drop an entire failing group and wrongly accept a grouped
    universal law, so we sample WITHIN each group and always keep every group represented. If the
    cap is smaller than the number of grounded groups it cannot represent them all, so we keep the
    full population rather than risk hiding a group (``None`` means "no subsample"). Ungrouped data
    falls back to a plain reproducible draw.
    """
    rng = np.random.default_rng(seed)
    labels = _row_group_keys(frame, nm, rows)
    if labels is None:
        return rng.choice(rows.size, size=subsample, replace=False)
    typed = [typed_group_key(label) for label in labels.tolist()]
    unique = list(dict.fromkeys(typed))
    n_groups = len(unique)
    if subsample < n_groups:
        return None
    index_by_group: dict = {}
    for idx, label in enumerate(typed):
        index_by_group.setdefault(label, []).append(idx)
    per_group = max(1, subsample // n_groups)
    keep: list = []
    for label in unique:
        members = np.asarray(index_by_group[label], dtype=int)
        if members.size <= per_group:
            keep.extend(members.tolist())
        else:
            picked = rng.choice(members.size, size=per_group, replace=False)
            keep.extend(members[picked].tolist())
    return np.asarray(sorted(keep), dtype=int)


def ground(rule: A.Rule, frame: Frame, nm: NameModel,
           scale_floor_frac: float = 1e-6, subsample: int = 0, seed: int = 0) -> Grounded:
    """Ground a rule to its residual population over all snapshots x bindings.

    ``subsample`` (>0) caps the number of residual points kept, drawn reproducibly -- the
    coreset lever of Sec. 10.2 for scaling to millions of points.
    """
    bindings = B.enumerate_bindings(rule.binder, nm)
    condition_mask = _condition_mask(rule.condition, frame)
    if condition_mask is None:
        condition_mask = np.zeros(frame.n_rows, dtype=bool)
    condition_support = (
        float(np.count_nonzero(condition_mask)) / frame.n_rows
        if frame.n_rows else 0.0
    )
    lefts, rights, rhos, scales, row_indices, overflows = [], [], [], [], [], []
    n_ok = 0
    for b in bindings:
        L, left_overflow = eval_term_overflow(rule.atom.left, rule.binder, b, frame, nm)
        R, right_overflow = eval_term_overflow(rule.atom.right, rule.binder, b, frame, nm)
        if L is None or R is None:
            continue
        n_ok += 1
        L = L[condition_mask]
        R = R[condition_mask]
        with np.errstate(over="ignore", invalid="ignore"):
            rho = L - R
        s = np.maximum(np.abs(L), np.abs(R))
        # The residual itself can overflow even when both sides are finite, so the subtraction is
        # checked here in addition to the masks the two sides carry up.
        overflow = _union_overflow(
            None if left_overflow is None else left_overflow[condition_mask],
            None if right_overflow is None else right_overflow[condition_mask],
            _blowup(rho, L, R),
        )
        lefts.append(L)
        rights.append(R)
        rhos.append(rho)
        scales.append(s)
        row_indices.append(np.flatnonzero(condition_mask))
        overflows.append(
            np.zeros(rho.shape, dtype=bool) if overflow is None else overflow
        )
    if n_ok == 0:
        empty = np.empty(0)
        return Grounded(
            empty,
            empty,
            empty,
            empty,
            0,
            len(bindings),
            True,
            empty.astype(int),
            condition_support,
        )
    left = np.concatenate(lefts)
    right = np.concatenate(rights)
    rho = np.concatenate(rhos)
    scale = np.concatenate(scales)
    rows = np.concatenate(row_indices)
    overflowed = np.concatenate(overflows)
    # An overflowed row is excluded from the residual population -- an infinity would poison every
    # median, band fit and hold-rate on it -- but it is COUNTED, because a row the candidate blew up
    # on is a row the candidate fails to describe, not a row the data failed to supply.  Downstream
    # arithmetic can even map an infinity back to a finite value (``1/inf == 0``), so the taint is
    # tracked explicitly rather than inferred from the final value's finiteness.
    overflow_points = int(np.count_nonzero(overflowed))
    attempted_points = int(n_ok * np.count_nonzero(condition_mask))
    overflow_fraction = (
        float(overflow_points) / float(attempted_points) if attempted_points else 0.0
    )
    overflow_rows = rows[overflowed]
    mask = np.isfinite(rho) & np.isfinite(scale) & ~overflowed
    left, right, rho, scale, rows = (
        left[mask],
        right[mask],
        rho[mask],
        scale[mask],
        rows[mask],
    )
    # Exactness and group coverage are UNIVERSAL properties, so they must be read off the full
    # grounded population -- before subsampling, which is only a statistical estimator for the
    # aggregate hold rate and must never decide a universal claim.
    raw_exact_sign = _sign_bound_raw_exact(rule.atom, rho)
    graded_points = int(rho.size)
    # Keep the pre-subsample operands and rows: a fitted parameter introduces arithmetic this
    # function never saw, and its finite-arithmetic guard is a UNIVERSAL claim over the graded
    # population. Checking it on a coreset would let a subsample that happens to miss the blown-up
    # row accept a rule whose own expression overflows.
    full_left, full_right, full_rows = left, right, rows
    graded_condition_support = (
        float(graded_points) / float(n_ok * frame.n_rows)
        if n_ok and frame.n_rows
        else 0.0
    )
    if subsample and rho.size > subsample:
        keep = _stratified_subsample(
            rows, frame, nm, subsample, seed,
        )
        if keep is not None:
            left, right, rho, scale, rows = (
                left[keep],
                right[keep],
                rho[keep],
                scale[keep],
                rows[keep],
            )
    # global floor keeps near-zero-scale points from exploding the relative residual
    med = robust_median(scale[scale > 0]) if np.any(scale > 0) else 1.0
    floor = max(
        scale_floor_frac * med,
        float(np.nextafter(0.0, 1.0)),
    )
    scale = np.maximum(scale, floor)
    return Grounded(rho=rho, scale=scale, left=left, right=right, n_bindings=n_ok,
                    n_candidates=len(bindings), degenerate=False, row_indices=rows,
                    condition_support=condition_support, raw_exact_sign=raw_exact_sign,
                    graded_points=graded_points,
                    graded_condition_support=graded_condition_support,
                    overflow_points=overflow_points,
                    overflow_fraction=overflow_fraction,
                    attempted_points=attempted_points,
                    full_left=full_left, full_right=full_right,
                    full_row_indices=full_rows,
                    overflow_row_indices=overflow_rows)


def rel_residual(g: Grounded) -> np.ndarray:
    """``|rho| / s`` -- the dimensionless residual used for band fitting."""
    return np.abs(g.rho) / g.scale


def _condition_mask(condition: A.Condition | None, frame: Frame):
    if condition is None:
        return np.ones(frame.n_rows, dtype=bool)
    cache_key = typed_condition_key(condition)
    if cache_key in frame.condition_cache:
        return frame.condition_cache[cache_key]
    if condition.op == "all":
        mask = np.ones(frame.n_rows, dtype=bool)
        for child in condition.values:
            if not isinstance(child, A.Condition):
                return None
            child_mask = _condition_mask(child, frame)
            if child_mask is None:
                return None
            mask &= child_mask
        frame.condition_cache[cache_key] = mask
        return mask
    if condition.column not in frame.row_context:
        return None
    values = _typed_object_array(frame.row_context[condition.column])
    present = ~pd.isna(values)
    mask = np.zeros(frame.n_rows, dtype=bool)
    if condition.op == "==":
        target = condition.values[0]
        if not bool(pd.isna(target)):
            target_key = typed_group_key(target)
            mask[present] = np.fromiter(
                (
                    typed_group_key(value) == target_key
                    for value in values[present]
                ),
                dtype=bool,
                count=int(np.count_nonzero(present)),
            )
        frame.condition_cache[cache_key] = mask
        return mask
    if condition.op == "!=":
        target = condition.values[0]
        if not bool(pd.isna(target)):
            target_key = typed_group_key(target)
            mask[present] = np.fromiter(
                (
                    typed_group_key(value) != target_key
                    for value in values[present]
                ),
                dtype=bool,
                count=int(np.count_nonzero(present)),
            )
        frame.condition_cache[cache_key] = mask
        return mask
    allowed = {
        typed_group_key(value)
        for value in condition.values
        if not bool(pd.isna(value))
    }
    if condition.op == "in":
        mask[present] = np.fromiter(
            (typed_group_key(value) in allowed for value in values[present]),
            dtype=bool,
            count=int(np.count_nonzero(present)),
        )
        frame.condition_cache[cache_key] = mask
        return mask
    if condition.op == "not in":
        mask[present] = np.fromiter(
            (
                typed_group_key(value) not in allowed
                for value in values[present]
            ),
            dtype=bool,
            count=int(np.count_nonzero(present)),
        )
        frame.condition_cache[cache_key] = mask
        return mask
    return None


_NAT_NS = np.iinfo(np.int64).min
_RELATED_MISSING_KEY = object()


def _datetime_ns(values) -> np.ndarray:
    raw = np.asarray(values)
    output = np.empty(raw.size, dtype=np.int64)
    for index, value in enumerate(raw.reshape(-1)):
        value = canonical_typed_value(value)
        if is_missing_scalar(value):
            output[index] = _NAT_NS
            continue
        try:
            output[index] = pd.Timestamp(value).as_unit("ns").value
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"timestamp {value!r} is outside datetime64[ns] range"
            ) from error
    return output.reshape(raw.shape)


def _saturating_add_ns(times: np.ndarray, delta_ns: int) -> np.ndarray:
    """``times + delta_ns`` in nanoseconds -> ``(ends, saturated)``, saturating instead of wrapping.

    Timestamps are int64 nanoseconds, and int64 addition WRAPS on overflow: a window that starts
    near ``pd.Timestamp.max`` ends up *before* its own start, so the interval search finds nothing
    and an event that is active throughout the window reads as absent -- which accepts a false law
    at a perfect hold rate. Saturating at the int64 maximum keeps the window ordered.
    """
    times = np.asarray(times, dtype=np.int64)
    limit = np.iinfo(np.int64).max
    delta = int(delta_ns)
    if delta <= 0:
        return times + delta, np.zeros(times.shape, dtype=bool)
    # Compare against ``limit - delta`` rather than computing ``limit - times``: the latter itself
    # overflows for a PRE-EPOCH (negative) timestamp, which reported saturation for every date
    # before 1970 and pushed its window end to the maximum representable instant.
    threshold = limit - delta
    saturated = times > threshold
    with np.errstate(over="ignore"):
        shifted = times + delta
    return np.where(saturated, limit, shifted), saturated


def _related_key_part(value):
    return _RELATED_MISSING_KEY if bool(pd.isna(value)) else value


def _parent_time_index(template, frame: Frame):
    cache_key = (
        "parents",
        template.parent_time,
        tuple(template.parent_keys),
    )
    if cache_key in frame.related_index_cache:
        return frame.related_index_cache[cache_key]
    parent_times = _datetime_ns(
        np.asarray(frame.row_context[template.parent_time])
    )
    key_arrays = [
        _typed_object_array(frame.row_context[column])
        for column in template.parent_keys
    ]
    groups: dict[tuple, list[int]] = {}
    for row in range(frame.n_rows):
        key = tuple(
            typed_group_key(_related_key_part(array[row]))
            for array in key_arrays
        )
        groups.setdefault(key, []).append(row)
    indexed = (parent_times, groups)
    frame.related_index_cache[cache_key] = indexed
    return indexed


def _child_partition_index(template, frame: Frame, child):
    cache_key = (
        "partitions",
        template.relation,
        template.child_time,
        tuple(template.child_keys),
        tuple(template.partition_keys),
    )
    if cache_key in frame.related_index_cache:
        return frame.related_index_cache[cache_key]
    child_times = _datetime_ns(child[template.child_time])
    group_columns = tuple(dict.fromkeys(
        (*template.child_keys, *template.partition_keys)
    ))
    if group_columns:
        # Row-wise on the TYPED identity rather than ``groupby``: pandas merges ``True`` with ``1``
        # (they are equal and hash alike), which silently joins two genuinely different shards and
        # sums their readings together. Grouping here has to agree with the parent index, so both
        # sides use the same typed key.
        columns = [
            _typed_object_array(child[column].to_numpy())
            for column in group_columns
        ]
        buckets: dict[tuple, list[int]] = {}
        for position in range(len(child)):
            key = tuple(
                typed_group_key(_related_key_part(column[position]))
                for column in columns
            )
            buckets.setdefault(key, []).append(position)
        grouped = [
            (key, np.asarray(positions, dtype=int))
            for key, positions in sorted(
                buckets.items(),
                key=lambda item: tuple(typed_sort_key(part) for part in item[0]),
            )
        ]
    else:
        grouped = [((), np.arange(len(child), dtype=int))]
    by_parent: dict[tuple, list[dict]] = {}
    sum_index = 0
    for raw_key, raw_positions in grouped:
        parent_key = tuple(raw_key[:len(template.child_keys)])
        positions = np.asarray(raw_positions, dtype=int)
        times = child_times[positions]
        present = times != _NAT_NS
        positions = positions[present]
        times = times[present]
        if not positions.size:
            continue
        order = np.argsort(times, kind="stable")
        by_parent.setdefault(parent_key, []).append({
            "positions": positions[order],
            "times": times[order],
            "values": {},
            "reset_prefix": {},
            "sum_index": sum_index,
        })
        sum_index += 1
    frame.related_index_cache[cache_key] = by_parent
    return by_parent


def _partition_values(partition: dict, child, column: str, forward_fill: bool = True) -> np.ndarray:
    """Child values for one partition, optionally carrying the last reading forward.

    Forward filling is correct for monotone counters, where a skipped report genuinely means "the
    counter has not moved since the last reading", and it is what the materialised fast path does
    for those columns. It is NOT correct for a boundary *level* column: carrying a stale level
    forward invents a reading the child never emitted, and the materialised path does not do it
    there. The two implementations of the same cross-grain law must agree on missing data, so the
    caller states which convention this column follows.
    """
    key = (column, bool(forward_fill))
    if key not in partition["values"]:
        values = pd.to_numeric(
            child.iloc[partition["positions"]][column],
            errors="coerce",
        )
        # A non-finite reading is missing data everywhere else in the engine, and forward fill only
        # carries ``NaN`` -- so an infinity left in place would be treated as a real reading, and
        # the increment it feeds would silently under-count the counter it stands in for.
        values = values.mask(~np.isfinite(values.to_numpy(dtype=float)))
        if forward_fill:
            values = values.ffill()
        partition["values"][key] = values.to_numpy(dtype=float)
    return partition["values"][key]


def _is_reset(value) -> bool:
    """A reset marker is any binary-truthy flag, regardless of how the frame stored it.

    Real GTIB data carries ``reset_flag`` as a native bool, but a CSV round-trip, a parquet
    load, or the runtime null control (which regenerates the column as balanced ``{0.0, 1.0}``
    floats) can present the same flag as an integer, a float, or a string. Recognising only the
    Python ``True`` singleton silently dropped every non-bool reset, so counters were treated as
    monotone across genuine resets. Treat any finite non-zero numeric, ``True``, or an explicit
    truthy string as a reset; ``NaN``/missing is not a reset.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return math.isfinite(numeric) and abs(numeric) > 1e-9
    text = str(value).strip().lower()
    return text in ("true", "1", "1.0", "yes", "t")


def _partition_reset_prefix(partition: dict, child, column: str) -> np.ndarray:
    if column not in partition["reset_prefix"]:
        resets = np.fromiter(
            (
                _is_reset(value)
                for value in child.iloc[partition["positions"]][column].tolist()
            ),
            dtype=bool,
            count=len(partition["positions"]),
        )
        partition["reset_prefix"][column] = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum(resets, dtype=np.int64),
        ))
    return partition["reset_prefix"][column]


def _related_aggregate(template, frame: Frame):
    """Cross-grain aggregate for one role -> ``(values, overflow)``.

    The overflow mask is produced INSIDE the aggregation rather than inferred from the finiteness of
    the result. A counter difference that exceeds float64 is a blow-up of this aggregation's own
    arithmetic, but the surrounding validity test would mark that shard invalid and let it
    contribute zero, so the total came out finite and complete-looking while silently under-counting
    -- and a false cross-grain law was accepted at hold rate and support 1.0.
    """
    cache_key = (template.binder, template.role)
    if cache_key in frame.related_cache:
        cached_values, cached_overflow = frame.related_cache[cache_key]
        return (
            cached_values.copy(),
            None if cached_overflow is None else cached_overflow.copy(),
        )
    child = frame.relations.get(template.relation)
    if child is None:
        return None, None
    if template.mode == "span_any":
        output = _span_any(template, frame, child)
        if output is not None:
            frame.related_cache[cache_key] = (output, None)
            return output.copy(), None
        return None, None
    required_parent = {*template.parent_keys, template.parent_time}
    if any(column not in frame.row_context for column in required_parent):
        return None, None
    required_child = {
        *template.child_keys,
        *template.partition_keys,
        template.child_time,
        template.column,
        *template.validity_columns,
    }
    if template.reset_column:
        required_child.add(template.reset_column)
    if any(column not in child.columns for column in required_child):
        return None, None

    parent_times, parent_groups = _parent_time_index(template, frame)
    child_partitions = _child_partition_index(template, frame, child)
    output = np.full(frame.n_rows, np.nan, dtype=float)
    overflow = np.zeros(frame.n_rows, dtype=bool)
    window_ns = int(pd.Timedelta(seconds=int(template.window_seconds)).value)
    fill_columns = tuple(dict.fromkeys(
        (template.column, *template.validity_columns)
    ))
    for parent_key, parent_rows in parent_groups.items():
        partitions = child_partitions.get(parent_key, ())
        if not partitions:
            continue
        ordered_parent = np.asarray(sorted(
            (
                row for row in parent_rows
                if parent_times[row] != _NAT_NS
            ),
            key=lambda row: parent_times[row],
        ), dtype=int)
        if not ordered_parent.size:
            continue
        starts = parent_times[ordered_parent]
        ends, ends_saturated = _saturating_add_ns(starts, window_ns)
        contribution_matrix = np.zeros(
            (len(ordered_parent), len(partitions)),
            dtype=float,
        )
        complete = np.ones(len(ordered_parent), dtype=bool)
        any_valid = np.zeros(len(ordered_parent), dtype=bool)
        blown = np.zeros(len(ordered_parent), dtype=bool)
        for local_index, partition in enumerate(partitions):
            times = partition["times"]
            interval_starts = np.searchsorted(times, starts, side="left")
            # A saturated window is clamped to the representable ceiling, so its upper bound has
            # to be inclusive or a reading sitting exactly on the ceiling falls outside a window
            # that genuinely contains it.
            interval_ends = np.where(
                ends_saturated,
                np.searchsorted(times, ends, side="right"),
                np.searchsorted(times, ends, side="left"),
            )
            has_interval = interval_starts < interval_ends
            boundaries = interval_ends - 1
            # A boundary *level* is read as emitted; a counter is carried forward. See
            # `_partition_values`.
            forward_fill = template.mode != "sum_last"
            values = {
                column: _partition_values(partition, child, column, forward_fill)
                for column in fill_columns
            }
            partition_valid = np.zeros(len(ordered_parent), dtype=bool)
            contribution = np.zeros(len(ordered_parent), dtype=float)
            if template.mode == "sum_last":
                active = np.flatnonzero(has_interval)
                current = values[template.column][boundaries[active]]
                valid = np.isfinite(current)
                rows = active[valid]
                partition_valid[rows] = True
                contribution[rows] = current[valid]
                # A cross-grain total is only defined when EVERY required partition contributes.
                # Requiring mere structural coverage here let a partition whose reading was
                # missing or unusable drop out silently, contributing zero to the sum -- the total
                # then looked complete while quietly under-counting.
                complete &= partition_valid
                any_valid |= partition_valid
                contribution_matrix[
                    :,
                    local_index,
                ] = contribution
                continue
            prior_boundaries = (
                np.searchsorted(times, starts, side="left") - 1
            )
            has_prior = prior_boundaries >= 0
            prior_is_adjacent = np.zeros(len(ordered_parent), dtype=bool)
            prior_rows = np.flatnonzero(has_prior)
            prior_deadlines, _prior_saturated = _saturating_add_ns(
                times[prior_boundaries[prior_rows]],
                window_ns,
            )
            prior_is_adjacent[prior_rows] = (
                prior_deadlines >= starts[prior_rows]
            )
            coverage = has_interval & has_prior & prior_is_adjacent
            eligible = coverage.copy()
            if template.reset_column:
                prefix = _partition_reset_prefix(
                    partition,
                    child,
                    template.reset_column,
                )
                covered_rows = np.flatnonzero(coverage)
                eligible[covered_rows] = (
                    prefix[interval_ends[covered_rows]]
                    - prefix[interval_starts[covered_rows]]
                ) == 0
            active = np.flatnonzero(eligible)
            valid = np.ones(len(active), dtype=bool)
            active_blown = np.zeros(len(active), dtype=bool)
            for column in template.validity_columns:
                end_values = values[column][boundaries[active]]
                start_values = values[column][prior_boundaries[active]]
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = end_values - start_values
                active_blown |= (
                    np.isinf(delta)
                    & np.isfinite(end_values)
                    & np.isfinite(start_values)
                )
                valid &= np.isfinite(delta) & (delta >= 0.0)
            end_values = values[template.column][boundaries[active]]
            start_values = values[template.column][prior_boundaries[active]]
            with np.errstate(over="ignore", invalid="ignore"):
                delta = end_values - start_values
            # An infinite difference of two FINITE readings is this aggregation's own arithmetic
            # blowing up, not a missing reading. Marking the shard invalid would let it contribute
            # zero to a total that then looks complete.
            active_blown |= (
                np.isinf(delta) & np.isfinite(end_values) & np.isfinite(start_values)
            )
            blown[active[active_blown]] = True
            valid &= np.isfinite(delta) & (delta >= 0.0)
            rows = active[valid]
            partition_valid[rows] = True
            contribution[rows] = delta[valid]
            # Deliberately `coverage`, not `partition_valid`, and deliberately different from the
            # `sum_last` branch above. This mode sums *increments*: a shard that reset inside the
            # interval has no measurable increment, and the emitted per-minute value likewise
            # excludes it, so skipping that shard is what matches the data (see
            # test_related_delta_sums_valid_shards_across_reset). A *level* sum has no such escape
            # -- every shard's backlog exists whether or not it was reported -- which is why that
            # branch is all-or-nothing.
            complete &= coverage
            any_valid |= partition_valid
            contribution_matrix[
                :,
                local_index,
            ] = contribution
        if partitions:
            totals, reduction_overflow = _canonical_row_sum(
                contribution_matrix
            )
            if reduction_overflow is not None:
                blown |= reduction_overflow
        else:
            totals = np.zeros(len(ordered_parent), dtype=float)
        accepted = complete & any_valid
        blown &= complete
        output[ordered_parent[accepted]] = totals[accepted]
        overflow[ordered_parent[blown]] = True

    result_overflow = overflow if overflow.any() else None
    frame.related_cache[cache_key] = (output, result_overflow)
    return (
        output.copy(),
        None if result_overflow is None else result_overflow.copy(),
    )


def _span_child_index(template, frame: Frame, child):
    cache_key = (
        "spans",
        template.relation,
        template.span_start,
        template.span_end,
        tuple(template.child_keys),
    )
    if cache_key in frame.related_index_cache:
        return frame.related_index_cache[cache_key]
    starts = _datetime_ns(child[template.span_start])
    ends = _datetime_ns(child[template.span_end])
    if template.child_keys:
        # Row-wise on the typed identity, for the same reason as the partition index: ``groupby``
        # merges ``True`` with ``1`` before the key is ever seen, joining two different children.
        columns = [
            _typed_object_array(child[column].to_numpy())
            for column in template.child_keys
        ]
        buckets: dict[tuple, list[int]] = {}
        for position in range(len(child)):
            key = tuple(
                typed_group_key(_related_key_part(column[position]))
                for column in columns
            )
            buckets.setdefault(key, []).append(position)
        grouped = [
            (key, np.asarray(positions, dtype=int))
            for key, positions in sorted(
                buckets.items(),
                key=lambda item: tuple(typed_sort_key(part) for part in item[0]),
            )
        ]
    else:
        grouped = [((), np.arange(len(child), dtype=int))]
    groups = {}
    for raw_key, raw_positions in grouped:
        parent_key = tuple(raw_key)
        positions = np.asarray(raw_positions, dtype=int)
        group_starts = starts[positions]
        group_ends = ends[positions]
        present = (group_starts != _NAT_NS) & (group_ends != _NAT_NS)
        positions = positions[present]
        group_starts = group_starts[present]
        group_ends = group_ends[present]
        order = np.argsort(group_starts, kind="stable")
        groups[parent_key] = {
            "positions": positions[order],
            "starts": group_starts[order],
            "ends": group_ends[order],
            "prefix_max_end": {},
        }
    frame.related_index_cache[cache_key] = groups
    return groups


def _span_prefix_max_end(group: dict, template, child) -> np.ndarray:
    cache_key = (
        template.filter_column,
        tuple(
            sorted(
                (typed_group_key(value) for value in template.filter_values),
                key=typed_sort_key,
            )
        ),
    )
    if cache_key not in group["prefix_max_end"]:
        if template.filter_column:
            allowed = set(cache_key[1])
            filter_values = child.iloc[group["positions"]][
                template.filter_column
            ].to_numpy(dtype=object)
            eligible = np.fromiter(
                (
                    typed_group_key(value) in allowed
                    for value in filter_values
                ),
                dtype=bool,
                count=filter_values.size,
            )
        else:
            eligible = np.ones(len(group["positions"]), dtype=bool)
        filtered_ends = np.where(eligible, group["ends"], _NAT_NS)
        group["prefix_max_end"][cache_key] = np.maximum.accumulate(
            filtered_ends
        )
    return group["prefix_max_end"][cache_key]


def _span_any(template, frame: Frame, child):
    required_parent = {*template.parent_keys, template.parent_time}
    required_child = {
        *template.child_keys,
        template.span_start,
        template.span_end,
    }
    if template.filter_column:
        required_child.add(template.filter_column)
    if any(column not in frame.row_context for column in required_parent):
        return None
    if any(not column or column not in child.columns for column in required_child):
        return None

    parent_times, parent_groups = _parent_time_index(template, frame)
    child_groups = _span_child_index(template, frame, child)
    output = np.full(frame.n_rows, np.nan, dtype=float)
    interval_ns = int(
        pd.Timedelta(seconds=int(template.window_seconds)).value
    )
    for parent_key, parent_rows in parent_groups.items():
        rows = np.asarray([
            row for row in parent_rows
            if parent_times[row] != _NAT_NS
        ], dtype=int)
        if not rows.size:
            continue
        output[rows] = 0.0
        group = child_groups.get(parent_key)
        if group is None or not len(group["starts"]):
            continue
        starts = parent_times[rows]
        end_windows, windows_saturated = _saturating_add_ns(starts, interval_ns)
        candidates = np.where(
            windows_saturated,
            np.searchsorted(group["starts"], end_windows, side="right"),
            np.searchsorted(group["starts"], end_windows, side="left"),
        )
        has_candidate = candidates > 0
        if not np.any(has_candidate):
            continue
        prefix_max_end = _span_prefix_max_end(group, template, child)
        candidate_rows = np.flatnonzero(has_candidate)
        latest_end = prefix_max_end[candidates[candidate_rows] - 1]
        output[rows[candidate_rows]] = (
            latest_end > starts[candidate_rows]
        ).astype(float)
    return output
