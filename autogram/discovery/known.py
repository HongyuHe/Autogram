"""Discovery reporter: recover user-supplied known invariants from the learned portfolio.

Users describe their known invariants as *column-level relations over real column names*
(no internal role knowledge required).  The reporter reuses the same structural signatures the
synthetic recovery scorer uses, so a known invariant is "recovered" iff the learned portfolio
contains a rule that grounds to the same column-level relation.  Recall is reported for a
held-out validation split so it cannot be fit to (see the calibration protocol).

Supported relation shapes (op / rhs):

* ``~=`` / ``==`` with a column rhs                -> pairwise equality
* ``~=`` / ``==`` with ``{sum: [cols...]}`` rhs    -> reference == family sum
* ``~=`` / ``==`` with ``0``                        -> zero
* ``<|>`` with a column rhs                          -> presence pairing
* ``>=`` / ``<=`` with ``0``                         -> one-sided non-negativity / non-positivity
* ``==`` / ``~=`` with ``{ratio: [num, den]}``        -> ratio identity
* ``~∝`` with a column rhs                            -> fitted proportional equality
* ``{delta: ...}``, ``{roll_sum: ...}``, and ``where`` -> grouped temporal and conditional relations
* ``{related: role}``                                 -> related-grain aggregation
* ``:=`` with sustained/and/priority                  -> Boolean or categorical definition
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_ref
from ..dsl.evaluate import (
    canonical_typed_value,
    eval_term,
    is_missing_scalar,
    typed_group_key,
    typed_signature_value,
    typed_sort_key,
)
from .loop import DiscoveryResult
from .validate import (
    _equality_relation,
    relation_signature_matches,
    _unwrap_equality_relation,
    portfolio_relations,
)


@dataclass
class KnownInvariant:
    name: str
    op: str
    lhs: object
    rhs: object          # column name (str), {"sum": [...]}, or a number (0)
    where: object = None


def _known_positive_int(value, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive JSON integer")
    return value


def _known_finite_number(value, label: str) -> float:
    if (
        not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite JSON number")
    return float(value)


def _known_is_zero(value) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def load_known(path: str) -> List[KnownInvariant]:
    """Load known invariants from a YAML or JSON file with an ``invariants:`` list."""
    text = open(path, "r", encoding="utf-8").read()
    doc = None
    if path.lower().endswith((".yaml", ".yml")):
        try:
            import yaml  # optional dependency
            doc = yaml.safe_load(text)
        except Exception:
            doc = None
    if doc is None:
        doc = json.loads(text)
    out: List[KnownInvariant] = []
    for i, e in enumerate(doc.get("invariants", [])):
        invariant = KnownInvariant(
            name=str(e.get("name", f"inv{i}")),
            op=str(e["op"]),
            lhs=e["lhs"],
            rhs=e.get("rhs"),
            where=e.get("where"),
        )
        if (
            invariant.where is not None
            and _known_condition_signature(invariant.where) is None
        ):
            raise ValueError(
                f"known invariant {invariant.name!r} has an invalid condition"
            )
        try:
            signature = _signature(invariant)
        except ValueError as error:
            raise ValueError(
                f"known invariant {invariant.name!r}: {error}"
            ) from error
        if signature is None:
            raise ValueError(
                f"known invariant {invariant.name!r} has an unsupported or invalid form"
            )
        out.append(invariant)
    return out


def _signature(inv: KnownInvariant):
    base = _base_signature(inv)
    if base is None or inv.where is None:
        return base
    condition = _known_condition_signature(inv.where)
    return None if condition is None else ("conditional", (condition, base))


def _base_signature(inv: KnownInvariant):
    op, lhs, rhs = inv.op, inv.lhs, inv.rhs
    is_zero = _known_is_zero(rhs)
    if (
        op == "~band"
        and isinstance(lhs, str)
        and isinstance(rhs, dict)
        and "center" in rhs
    ):
        return (
            "healthy_band",
            (
                lhs,
                _known_finite_number(
                    rhs["center"],
                    "healthy-band center",
                ),
            ),
        )
    if (
        op in ("~=", "==")
        and isinstance(lhs, dict)
        and "sum" in lhs
        and isinstance(rhs, dict)
        and "sum" in rhs
    ):
        return _equality_relation(
            op,
            (
                "sum_balance",
                frozenset({
                    frozenset(str(column) for column in lhs["sum"]),
                    frozenset(str(column) for column in rhs["sum"]),
                }),
            ),
        )
    if op in ("~=", "==") and isinstance(lhs, str) and isinstance(rhs, dict) and "related" in rhs:
        return _equality_relation(
            op,
            ("related_aggregate", (lhs, str(rhs["related"]))),
        )
    if op == ":=" and isinstance(lhs, str) and isinstance(rhs, dict):
        if "sustained" in rhs:
            predicate = _known_sustained_signature(rhs["sustained"])
            return None if predicate is None else (
                "sustained_definition",
                (lhs, predicate),
            )
        if "and" in rhs:
            predicates = tuple(sorted(
                (_known_bound_signature(item) for item in rhs["and"]),
                key=str,
            ))
            if predicates and all(predicate is not None for predicate in predicates):
                return ("conjunction_definition", (lhs, predicates))
        if "priority" in rhs:
            cases = tuple(
                (
                    str(item["when"]),
                    typed_signature_value(item["value"]),
                )
                for item in rhs["priority"]
            )
            return (
                "categorical_definition",
                (lhs, cases, typed_signature_value(rhs.get("default"))),
            )
    if (
        op in (">=", "<=", ">", "<")
        and is_zero
        and isinstance(lhs, dict)
        and "lag" in lhs
    ):
        value = lhs["lag"]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        column, steps = value
        return (
            "lag_bound",
            (
                str(column),
                _known_positive_int(steps, "lag steps"),
                op,
            ),
        )
    if op in (">=", "<=", ">", "<") and is_zero and isinstance(lhs, dict) and "delta" in lhs:
        value = lhs["delta"]
        if isinstance(value, (list, tuple)):
            column, steps = value
        else:
            column, steps = value, 1
        return (
            "delta_bound",
            (
                str(column),
                _known_positive_int(steps, "delta steps"),
                op,
            ),
        )
    if op in ("~=", "==") and is_zero and isinstance(lhs, dict) and "delta" in lhs:
        value = lhs["delta"]
        if isinstance(value, (list, tuple)):
            column, steps = value
        else:
            column, steps = value, 1
        return _equality_relation(
            op,
            (
                "delta_zero",
                (
                    str(column),
                    _known_positive_int(steps, "delta steps"),
                ),
            ),
        )
    if op in ("~=", "==") and isinstance(lhs, str) and isinstance(rhs, dict) and "ratio" in rhs:
        values = list(rhs["ratio"])
        if len(values) == 2:
            num = _known_temporal_ref(values[0], "roll_sum")
            den = _known_temporal_ref(values[1], "roll_sum")
            if num is not None and den is not None and num[1] == den[1]:
                return _equality_relation(
                    op,
                    ("windowed_ratio", (lhs, num[0], den[0], num[1])),
                )
    if op in ("~=", "==") and is_zero:
        return _equality_relation(op, ("zero", lhs))
    if op in ("~=", "==") and isinstance(rhs, dict) and "sum" in rhs:
        return _equality_relation(
            op,
            ("ref_sum", (lhs, frozenset(str(c) for c in rhs["sum"]))),
        )
    if op in ("~=", "==") and isinstance(rhs, dict) and "ratio" in rhs:
        values = list(rhs["ratio"])
        if len(values) == 2:
            return _equality_relation(
                op,
                ("ratio", (lhs, str(values[0]), str(values[1]))),
            )
    if op in ("~=", "==") and isinstance(rhs, str):
        return _equality_relation(
            op,
            ("pair", frozenset({lhs, rhs})),
        )
    if op == "~∝" and isinstance(rhs, str):
        return ("proportional", (lhs, rhs))
    if op == "!=" and isinstance(lhs, str) and isinstance(rhs, str):
        return ("separation_pair", frozenset({lhs, rhs}))
    if op == "<|>" and isinstance(rhs, str):
        return ("presence_pair", frozenset({lhs, rhs}))
    if op in (">=", "<=") and is_zero:
        return ("one_sided", lhs, op)
    return None


def _known_condition_signature(where):
    if not isinstance(where, dict) or len(where) != 1:
        return None
    key, value = next(iter(where.items()))
    if key == "all" and isinstance(value, list):
        children = tuple(sorted(
            (_known_condition_signature(item) for item in value),
            key=str,
        ))
        return None if any(child is None for child in children) else ("all", children)
    if key.endswith("_in"):
        column = key[:-3]
        if not isinstance(value, (list, tuple)) or not value:
            return None
        canonical = [
            _known_condition_scalar(item)
            for item in value
        ]
        if any(item is None for item in canonical):
            return None
        values = tuple(sorted(
            (typed_signature_value(item) for item in canonical),
            key=lambda item: typed_sort_key(item[1]),
        ))
        return (column, "in", values)
    canonical = _known_condition_scalar(value)
    if canonical is None:
        return None
    return (str(key), "==", (typed_signature_value(canonical),))


def _known_condition_scalar(value):
    value = canonical_typed_value(value)
    if is_missing_scalar(value):
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        return float(value)
    return None


def _known_temporal_ref(value, form: str):
    if not isinstance(value, dict) or form not in value:
        return None
    payload = value[form]
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        return None
    return (
        str(payload[0]),
        _known_positive_int(payload[1], f"{form} window"),
    )


def _known_term_signature(value):
    if isinstance(value, str):
        return ("ref", value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return (
            "const",
            _known_finite_number(value, "term constant"),
        )
    if not isinstance(value, dict):
        return None
    if "delta" in value:
        payload = value["delta"]
        if isinstance(payload, (list, tuple)):
            if len(payload) != 2:
                return None
            return (
                "delta",
                _known_term_signature(payload[0]),
                _known_positive_int(payload[1], "delta steps"),
            )
        return ("delta", _known_term_signature(payload), 1)
    if "lag" in value:
        payload = value["lag"]
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return None
        return (
            "lag",
            _known_term_signature(payload[0]),
            _known_positive_int(payload[1], "lag steps"),
        )
    if "roll_sum" in value:
        payload = value["roll_sum"]
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return None
        return (
            "rolling",
            "SUM",
            _known_positive_int(payload[1], "rolling window"),
            _known_term_signature(payload[0]),
        )
    if "difference" in value:
        left, right = value["difference"]
        return ("difference", _known_term_signature(left), _known_term_signature(right))
    return None


def _known_bound_signature(value):
    if not isinstance(value, dict) or "bound" not in value:
        return None
    term, op, threshold = value["bound"]
    term_signature = _known_term_signature(term)
    return None if term_signature is None else (
        "bound",
        term_signature,
        str(op),
        _threshold_signature(threshold),
    )


def _known_sustained_signature(value):
    if not isinstance(value, dict):
        return None
    term_signature = _known_term_signature(value.get("term"))
    if term_signature is None:
        return None
    return (
        "sustained",
        _known_positive_int(value["window"], "sustained window"),
        (
            "bound",
            term_signature,
            str(value["op"]),
            _threshold_signature(value.get("threshold")),
        ),
    )


def _threshold_signature(value):
    if value is None:
        return None
    return _known_finite_number(value, "predicate threshold")


def shapes_for_invariant(inv: KnownInvariant) -> List[str]:
    """Map one known-invariant relation form to the generic proxy shape(s) that cover it.

    The mapping inspects only the *structure* of the relation (operator and right-hand-side
    form) -- never domain-specific words in the variable names -- so it is dataset-agnostic:

    * ``==`` with a column rhs                      -> ``["two_end"]``       (exact pairwise equality)
    * ``~=`` with a column rhs                      -> ``["offset_pair"]``   (approximate pairwise equality)
    * ``~=`` / ``==`` with a ``{sum: [...]}`` rhs   -> ``["row_sum", "col_sum"]`` (reference == family sum;
      the file format does not encode which matrix axis the family spans, so both are covered)
    * ``~=`` / ``==`` with ``0``                    -> ``["self_zero"]``     (equality to zero)
    * ``<|>`` with a column rhs                     -> ``["presence_pair"]`` (presence pairing)
    * ``>= 0``                                      -> ``["nonneg"]``
    * ``<= 0``                                      -> ``["nonpos"]``

    ``agg_ref_balance`` is intentionally never produced: the known-invariant file format cannot
    express a mixed reference-plus-sum balance, so that shape is reachable only via a custom
    ``RegimeSpec``.  Any unsupported form maps to ``[]``.
    """
    op, rhs = inv.op, inv.rhs
    _strength, base = _unwrap_equality_relation(_base_signature(inv))
    if inv.where is not None and base is not None:
        if base[0] == "pair":
            return ["conditional_pair"]
        if base[0] == "delta_bound" and op in (">=", ">"):
            return ["conditional_positive"]
        if base[0] == "delta_zero":
            return ["conditional_zero"]
        if base[0] == "proportional":
            return ["conditional_proportional"]
    if base is not None and base[0] == "related_aggregate":
        return ["cross_grain"]
    if base is not None and base[0] == "sustained_definition":
        return ["sustained"]
    if base is not None and base[0] == "conjunction_definition":
        return ["conjunction"]
    if base is not None and base[0] == "categorical_definition":
        return ["categorical"]
    if base is not None and base[0] == "healthy_band":
        return ["healthy_band"]
    if base is not None and base[0] == "lag_bound":
        return ["lag_bound"]
    if base is not None and base[0] == "sum_balance":
        return ["sum_balance"]
    if op in (">=", "<=", ">", "<") and isinstance(inv.lhs, dict) and "delta" in inv.lhs:
        return ["monotone"]
    if base is not None and base[0] == "windowed_ratio":
        return ["windowed_ratio"]
    is_zero = _known_is_zero(rhs)
    if op in ("~=", "==") and is_zero:
        return ["self_zero"]
    if op in ("~=", "==") and isinstance(rhs, dict) and "sum" in rhs:
        return ["row_sum", "col_sum"]
    if op in ("~=", "==") and isinstance(rhs, dict) and "ratio" in rhs:
        return ["ratio"]
    if op == "==" and isinstance(rhs, str):
        return ["two_end"]
    if op == "~=" and isinstance(rhs, str):
        return ["offset_pair"]
    if op == "<|>" and isinstance(rhs, str):
        return ["presence_pair"]
    if op == "~∝" and isinstance(rhs, str):
        return ["proportional"]
    if op == ">=" and is_zero:
        return ["nonneg"]
    if op == "<=" and is_zero:
        return ["nonpos"]
    return []


def abstract_shapes(known: List[KnownInvariant]) -> List[str]:
    """Union (first-seen order, deduped) of the proxy shapes covering ``known``.

    Returns an empty list when no invariant maps to a supported shape; the calibrator turns that
    into a loud error rather than silently proxying every shape.
    """
    out: List[str] = []
    for inv in known:
        for shape in shapes_for_invariant(inv):
            if shape not in out:
                out.append(shape)
    return out


def _one_sided_columns(result: DiscoveryResult, op: str) -> set:
    """Columns C for which the portfolio contains ``[forall b] <ref over C> op 0``."""
    ds = result.dataset
    nm = ds.name_model
    cols: set = set()
    for ev in result.portfolio:
        if ev.rule.condition is not None:
            continue
        atom = ev.rule.atom
        if not isinstance(atom, A.Compare):
            continue
        accepted_ops = {
            ">=": {">=", ">"},
            "<=": {"<=", "<"},
        }.get(op, {op})
        if atom.op not in accepted_ops:
            continue
        if not (isinstance(atom.right, A.Const) and float(atom.right.value) == 0.0):
            continue
        if not isinstance(atom.left, A.Ref):
            continue
        for b in enumerate_bindings(ev.rule.binder, nm):
            c = resolve_ref(atom.left.role, ev.rule.binder, b, nm)
            if c is not None:
                cols.add(c)
    return cols


def _column_satisfies_sign_exactly(frame, col: str, op: str) -> bool:
    """True iff every observed value of ``col`` satisfies ``value op 0`` EXACTLY (no tolerance).

    One-sided *evaluation* accepts a bound within a relative tolerance against a population scale
    floor, so ``hold_rate == 1.0`` does NOT mean ``x op 0`` holds on every raw value (a large-scale
    column can absorb a genuine violation). A lag sign law ``LAG_k(x) op 0`` is only a guaranteed
    consequence when the raw column satisfies ``op 0`` exactly, so we check the data directly here.
    """
    if not frame.has(col):
        return False
    v = frame.col(col)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return False
    if op == ">=":
        return bool(np.all(v >= 0.0))
    if op == ">":
        return bool(np.all(v > 0.0))
    if op == "<=":
        return bool(np.all(v <= 0.0))
    if op == "<":
        return bool(np.all(v < 0.0))
    return False


def _exact_lag_bound_columns(result: DiscoveryResult, lag_op: str) -> set:
    """Columns whose *exact* atomic sign law implies the shifted law ``LAG_k(C) lag_op 0``.

    A lag one-sided bound ``LAG_k(x) OP 0`` is not retained as its own rule -- it is the column's
    sign law shifted in time, and keeping the lag form would pollute the null-temporal control (a
    non-negative column's ``LAG_k(x) >= 0`` holds on shuffled null data). The lag law is genuinely
    guaranteed only when the column satisfies ``OP 0`` *exactly* (verified tolerance-free against the
    data): ``x OP 0`` on every row forces ``LAG_k(x) OP 0`` on every valid lagged row regardless of
    the shift. A merely-accepted atomic (whose ``hold_rate`` may be 1.0 only because the acceptance
    tolerance absorbed a violation relative to a large population scale) does NOT imply the shifted
    law, so it is excluded. We still require the engine to have discovered a same-direction atomic
    sign bound (the portfolio witness) before crediting the lag, tying recovery to what was learned.
    """
    frame = result.dataset.observed
    ds = result.dataset
    nm = ds.name_model
    lower = lag_op in (">=", ">")
    direction_ops = {">=", ">"} if lower else {"<=", "<"}
    cols: set = set()
    for ev in result.portfolio:
        if ev.rule.condition is not None:
            continue
        atom = ev.rule.atom
        if not isinstance(atom, A.Compare):
            continue
        if atom.op not in direction_ops:
            continue
        if not (isinstance(atom.right, A.Const) and float(atom.right.value) == 0.0):
            continue
        if not isinstance(atom.left, A.Ref):
            continue
        for b in enumerate_bindings(ev.rule.binder, nm):
            c = resolve_ref(atom.left.role, ev.rule.binder, b, nm)
            # Sound only when the raw column satisfies the lag's own operator exactly: an exact
            # non-strict ``x >= 0`` still cannot witness a strict ``LAG > 0`` (a zero value breaks
            # the shifted strict law), which ``_column_satisfies_sign_exactly`` enforces via ``lag_op``.
            if c is not None and _column_satisfies_sign_exactly(frame, c, lag_op):
                cols.add(c)
    return cols



def _col_values(frame, col: str):
    """Observed values of a column as a float vector, or ``None`` when the frame lacks it."""
    if not frame.has(col):
        return None
    values = np.asarray(frame.col(col), dtype=float)
    return values if values.size else None


def _shared_gradeable(anchor: np.ndarray, members: dict) -> np.ndarray:
    """Rows on which the whole summed grouping can be evaluated.

    A sum is undefined wherever ANY member is missing, so gradeability is a property of the
    grouping, not of one member at a time.  Testing a member against the anchor alone would keep a
    member that is non-zero only on rows the sum cannot be graded on -- a member that provably never
    changes the relation, whose retention splits two behaviourally identical catalogue entries
    across the held-out boundary.
    """
    gradeable = np.isfinite(anchor)
    for values in members.values():
        gradeable = gradeable & np.isfinite(values)
    return gradeable


def _stable_row_sum(magnitudes: dict) -> np.ndarray:
    """Row-wise sum of per-column magnitudes, in an order that does not depend on hashing.

    Floating-point addition is not associative, so accumulating in ``dict``/``frozenset`` iteration
    order makes the result depend on string hash randomisation -- and therefore makes the
    calibration/validation split itself differ between runs on identical input. Summing in
    name-sorted order is deterministic; ``math.fsum`` then makes each row's total exact, so the
    aggregate negligibility bound cannot flip on a rounding artefact either.
    """
    if not magnitudes:
        return np.zeros(0, dtype=float)
    ordered = [magnitudes[name] for name in sorted(magnitudes)]
    stacked = np.stack(ordered, axis=0)

    def _row_total(row) -> float:
        try:
            return math.fsum(row)
        except OverflowError:
            # An aggregate beyond float64 is, by definition, past any finite budget. Reporting it as
            # infinite keeps the caller's comparison well-defined instead of crashing the run.
            return float("inf")

    return np.fromiter(
        (_row_total(row) for row in stacked.T),
        dtype=float,
        count=stacked.shape[1],
    )


def _drop_negligible(cols, anchor_col: str, frame, zero_tol: float,
                     exact: bool = False) -> frozenset:
    """Drop summed columns that do not materially change the sum on any gradeable row.

    Two groupings that differ only by such columns describe the *same* physical fact, which is what
    licenses treating them as one relation.  Three properties make that licence sound:

    * **Pointwise, not central.**  A member must be within ``zero_tol`` of the anchor's magnitude on
      *every* gradeable row.  A central statistic cannot decide this: a column that is ``0`` on 51%
      of rows and ``1000`` on the rest has a zero median, and dropping it would credit a known
      invariant as recovered while it is violated on 49% of the data.
    * **Collective, not one-at-a-time.**  Individually-negligible members still add up: 607 members
      each under the tolerance contributed 6% of the total between them.  The members removed
      together must therefore stay within the same bound *in aggregate*; when they do not, only the
      members that are exactly zero everywhere are removed, whose combined contribution is exactly
      zero.  That fallback is also what keeps the transform idempotent -- re-canonicalising the
      reduced grouping removes nothing further.
    * **Domain-preserving.**  Removal must not widen the population the relation is graded on.  A
      member that is itself missing somewhere restricts the sum's domain, so dropping it would hand
      the reduced relation rows the original never had to satisfy.  The test is collective, not
      per member: a member missing only where another RETAINED member is missing too changes
      nothing, so the check compares the original and post-removal gradeability masks and shrinks
      the removal set to a fixpoint.
    * **Anchored and dimensionless.**  The bound is a fraction of the reference (left-hand side)
      column's magnitude on the same row, so the test is scale-free and dataset-agnostic.

    We never reduce a whole group to empty (that would collapse distinct laws), and a column the
    frame does not carry is always kept.
    """
    if zero_tol <= 0.0:
        return frozenset(cols)                       # exact column-set matching requested
    if exact:
        # An EXACT relation has no tolerance to spend: a member that is merely small still breaks
        # ``total == SUM(...)`` on every row it is non-zero. Setting the budget to zero expresses
        # exactly that, and -- crucially -- leaves the member on the SAME pipeline as an approximate
        # one, so the domain-preserving fixpoint below still applies. Short-circuiting to "drop the
        # identically-zero members" instead let a member that is zero where defined but MISSING
        # elsewhere be removed, which widens the graded population and credits a learned sum that
        # fails on the rows the known relation never had to satisfy.
        zero_tol = 0.0
    anchor = _col_values(frame, anchor_col)
    if anchor is None or not np.any(np.isfinite(anchor) & (np.abs(anchor) > 0.0)):
        return frozenset(cols)                       # no usable anchor -> do not canonicalize
    members = {}
    for col in sorted(cols):
        values = _col_values(frame, col)
        if values is not None and values.shape == anchor.shape:
            members[col] = values
    if not members:
        return frozenset(cols)
    gradeable = _shared_gradeable(anchor, members)
    if not np.any(gradeable):
        return frozenset(cols)
    # With ``zero_tol == 0`` (an exact relation) the budget is zero everywhere, so only members that
    # are exactly zero on every gradeable row qualify -- which is the correct reading of "exact".
    budget = zero_tol * np.abs(anchor[gradeable])
    candidates = {}
    for col, values in members.items():
        magnitude = np.abs(values[gradeable])
        if bool(np.all(magnitude <= budget)):
            candidates[col] = magnitude
    # Removing a member must not WIDEN the population the relation is graded on. A member that is
    # itself missing somewhere restricts the sum's domain, so dropping it would hand the reduced
    # relation rows the original never had to satisfy -- exactly how ``total == SUM(real)`` came to
    # be credited with recovering ``total == SUM(real, z)`` while failing on 90 of 100 rows. The
    # test is on the *resulting* grouping, not on each member against the anchor: a member missing
    # only where another RETAINED member is missing too changes nothing, and demanding otherwise
    # would split two identically-evaluated sums across the held-out boundary. Shrinking the removal
    # set only ever removes constraints, so this fixpoint terminates.
    while candidates:
        retained = {
            col: values for col, values in members.items()
            if col not in candidates
        }
        widened = _shared_gradeable(anchor, retained)
        if not np.any(widened & ~gradeable):
            break
        offenders = [
            col for col in sorted(candidates)
            if not bool(np.all(np.isfinite(members[col][widened])))
        ]
        if not offenders:
            break
        for col in offenders:
            candidates.pop(col, None)
    if not candidates:
        return frozenset(cols)
    combined = _stable_row_sum(candidates)
    if not bool(np.all(combined <= budget)):
        # Aggregate contribution is material: fall back to the members that contribute exactly
        # nothing, which is both sound and stable under re-canonicalisation.
        candidates = {
            col: magnitude
            for col, magnitude in candidates.items()
            if not bool(np.any(magnitude > 0.0))
        }
    kept = frozenset(col for col in cols if col not in candidates)
    return kept if kept else frozenset(cols)         # never canonicalize an entire group away


def _canonicalize(sig, frame, zero_tol: float, exact: bool | None = None):
    """Map a relation signature to a data-canonical form (negligible sum members removed).

    Only the sum-shaped signatures carry groupings, so only they are canonicalized; pairwise,
    zero, presence and one-sided signatures pass through unchanged.  The transform is idempotent
    and widens matching within the declared tolerance: an exact relation removes only members with
    zero contribution, while an approximate relation may remove a collectively bounded non-zero
    contribution. Anything that matched before still matches after canonicalizing.
    """
    if not isinstance(sig, tuple) or not sig:
        return sig
    if sig[0] == "equality" and len(sig) == 3:
        # ``exact=None`` means "read the exactness off this signature"; an explicit value is an
        # OVERRIDE and must win. Deriving it unconditionally made the override inert, so a learned
        # exact sum could never be canonicalised under the tolerance an approximate known permits
        # -- and the approximate known it satisfies was reported as unrecovered.
        resolved = (sig[1] == "exact") if exact is None else bool(exact)
        return (
            sig[0],
            sig[1],
            _canonicalize(sig[2], frame, zero_tol, exact=resolved),
        )
    if sig[0] == "conditional" and len(sig) == 2:
        condition, base = sig[1]
        return (
            "conditional",
            (condition, _canonicalize(base, frame, zero_tol, exact=exact)),
        )
    if sig[0] == "ref_sum":
        ref_col, cols = sig[1]
        return (
            "ref_sum",
            (
                ref_col,
                _drop_negligible(
                    cols, ref_col, frame, zero_tol, exact=bool(exact),
                ),
            ),
        )
    if sig[0] == "sum_balance":
        groups = tuple(sig[1])
        singletons = [group for group in groups if len(group) == 1]
        if len(groups) == 2 and len(singletons) == 1:
            singleton = frozenset(singletons[0])
            anchor = next(iter(singleton))
            other = groups[0] if groups[1] == singleton else groups[1]
            canonical_other = _drop_negligible(
                other,
                anchor,
                frame,
                zero_tol,
                exact=bool(exact),
            )
            return (
                "ref_sum",
                (anchor, canonical_other),
            )
        return sig
    if sig[0] == "agg_ref_balance":
        return (
            "sum_balance",
            frozenset(
                frozenset({ref, *fam})
                for ref, fam in sig[1]
            ),
        )
    return sig


def _lag_grounds_any_row(result: DiscoveryResult, column: str, steps: int) -> bool:
    """Does ``LAG_steps(column)`` actually have any observed row on this data?

    Implication from an exact atomic sign law is only a licence to *transfer* a law that the data
    witnesses; it is not a licence to invent one. A lag longer than the series -- ``LAG_100(x)`` on
    a fifty-row group -- grounds no rows at all, so there is nothing to transfer and crediting it
    would report a law the dataset never exhibits. Requiring at least one grounded row keeps the
    recall figure honest.
    """
    dataset = result.dataset
    name_model = dataset.name_model
    for ev in result.portfolio:
        atom = ev.rule.atom
        if (
            ev.rule.condition is not None
            or not isinstance(atom, A.Compare)
            or not isinstance(atom.left, A.Ref)
        ):
            continue
        for binding in enumerate_bindings(ev.rule.binder, name_model):
            resolved = resolve_ref(
                atom.left.role,
                ev.rule.binder,
                binding,
                name_model,
            )
            if resolved != column:
                continue
            try:
                values = eval_term(
                    A.Lag(A.Ref(atom.left.role), int(steps)),
                    ev.rule.binder,
                    binding,
                    dataset.observed,
                    name_model,
                )
            except Exception:
                continue
            if (
                values is not None
                and np.any(np.isfinite(np.asarray(values, dtype=float)))
            ):
                return True
    return False


def _matching_signatures(sig):
    candidates = [sig]
    if (
        isinstance(sig, tuple)
        and len(sig) == 3
        and sig[0] == "equality"
        and sig[1] == "approximate"
    ):
        candidates.append(("equality", "exact", sig[2]))
    elif (
        isinstance(sig, tuple)
        and len(sig) == 2
        and sig[0] == "conditional"
    ):
        condition, base = sig[1]
        candidates = [
            ("conditional", (condition, candidate))
            for candidate in _matching_signatures(base)
        ]
    return candidates


def _candidate_is_exact(candidate) -> bool:
    """Does this known-signature candidate assert an EXACT relation?

    Recurses through ``conditional`` nesting: a conditioned exact equality is still exact, and
    reading only the outer shape canonicalised it with an approximate tolerance -- which matched a
    conditioned exact known against a learned sum it is false against on every applicable row.
    """
    if not isinstance(candidate, tuple) or not candidate:
        return False
    if candidate[0] == "conditional" and len(candidate) == 2:
        _condition, base = candidate[1]
        return _candidate_is_exact(base)
    return (
        len(candidate) == 3
        and candidate[0] == "equality"
        and candidate[1] == "exact"
    )


def _matches_any(sig, frame, zero_tol: float, canon_by_tolerance: dict) -> bool:
    """Is any expansion of ``sig`` matched by a learned relation, at that expansion's tolerance?"""
    for candidate in _matching_signatures(sig):
        exact = _candidate_is_exact(candidate)
        canon_candidate = _canonicalize(candidate, frame, zero_tol, exact=exact)
        for learned in canon_by_tolerance[exact]:
            if relation_signature_matches(canon_candidate, learned):
                return True
    return False


def recover_known(result: DiscoveryResult, known: List[KnownInvariant],
                  zero_tol: float = 1e-4) -> dict:
    """Report per-invariant recovery + aggregate recall of the user's known invariants.

    A known invariant counts as recovered iff the learned portfolio contains a rule with the same
    *data-canonical* relation signature.  Canonicalization removes provably-negligible (near-zero)
    columns from any summed grouping (``zero_tol`` is the drop threshold, relative to the reference
    column's scale), so a known sum written over a slightly different column set -- e.g. one that
    includes a structurally-zero self term the induced grammar omits -- still matches the physically
    identical law the engine found.  Set ``zero_tol=0`` to require exact column-set matches.
    """
    frame = result.dataset.observed
    rels = portfolio_relations(result)
    # The learned side is canonicalised under the tolerance the KNOWN relation permits, not under
    # its own. A learned *exact* sum is still recovered by an approximate known written over a
    # slightly different column set -- the known one tolerates the difference, and it is the known
    # one whose recovery is being reported. Canonicalising the learned side by its own exactness
    # instead made an exact learned rule unmatchable by the approximate known it satisfies.
    canon_by_tolerance = {
        exact: [_canonicalize(s, frame, zero_tol, exact=exact) for s in rels]
        for exact in (False, True)
    }
    ge_cols = _one_sided_columns(result, ">=")
    le_cols = _one_sided_columns(result, "<=")
    report: List[dict] = []
    n_ok = 0
    for inv in known:
        sig = _signature(inv)
        recovered = False
        if sig is None:
            recovered = False
        elif sig[0] == "one_sided":
            recovered = inv.lhs in (ge_cols if sig[2] == ">=" else le_cols)
        elif sig[0] == "lag_bound":
            # Sound recovery of a lag sign law. It is recovered when either (a) an EXACT atomic sign
            # law (hold-rate 1.0) implies the shift -- ``x OP 0`` on every row forces ``LAG_k(x) OP 0``
            # on every valid lagged row -- or (b) a genuinely temporal lag rule survived in the
            # portfolio and matches structurally. A merely-accepted (non-exact) atomic does NOT imply
            # the shifted law over its distinct valid-row population, so it never counts here; that
            # keeps recovery from over-claiming a lag law the data does not actually witness.
            _column, _steps, _op = sig[1]
            recovered = (
                _column in _exact_lag_bound_columns(result, _op)
                and _lag_grounds_any_row(result, _column, _steps)
            )
            if not recovered:
                recovered = _matches_any(
                    sig, frame, zero_tol, canon_by_tolerance,
                )
        else:
            recovered = _matches_any(
                sig, frame, zero_tol, canon_by_tolerance,
            )
        n_ok += int(recovered)
        report.append({"name": inv.name, "op": inv.op, "recovered": bool(recovered),
                       "signature": str(sig)})
    recall = (n_ok / len(known)) if known else 0.0
    return {"recall": recall, "recovered": n_ok, "total": len(known), "invariants": report}
