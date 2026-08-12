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
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_ref
from ..dsl.evaluate import eval_term
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
        out.append(KnownInvariant(
            name=str(e.get("name", f"inv{i}")),
            op=str(e["op"]),
            lhs=e["lhs"],
            rhs=e.get("rhs"),
            where=e.get("where"),
        ))
    return out


def _signature(inv: KnownInvariant):
    base = _base_signature(inv)
    if base is None or inv.where is None:
        return base
    condition = _known_condition_signature(inv.where)
    return None if condition is None else ("conditional", (condition, base))


def _base_signature(inv: KnownInvariant):
    op, lhs, rhs = inv.op, inv.lhs, inv.rhs
    is_zero = isinstance(rhs, (int, float)) and float(rhs) == 0.0
    if (
        op == "~band"
        and isinstance(lhs, str)
        and isinstance(rhs, dict)
        and "center" in rhs
    ):
        return ("healthy_band", (lhs, float(rhs["center"])))
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
                (str(item["when"]), item["value"])
                for item in rhs["priority"]
            )
            return (
                "categorical_definition",
                (lhs, cases, rhs.get("default")),
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
        return ("lag_bound", (str(column), int(steps), op))
    if op in (">=", "<=", ">", "<") and is_zero and isinstance(lhs, dict) and "delta" in lhs:
        value = lhs["delta"]
        if isinstance(value, (list, tuple)):
            column, steps = value
        else:
            column, steps = value, 1
        return ("delta_bound", (str(column), int(steps), op))
    if op in ("~=", "==") and is_zero and isinstance(lhs, dict) and "delta" in lhs:
        value = lhs["delta"]
        if isinstance(value, (list, tuple)):
            column, steps = value
        else:
            column, steps = value, 1
        return _equality_relation(
            op,
            ("delta_zero", (str(column), int(steps))),
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
        # Preserve the declared scalar types (e.g. integer category codes): the learned rule's
        # membership values keep their observed dtype, so stringifying here would prevent a valid
        # int/float membership known from ever matching. Sort by string only for a stable order.
        values = tuple(sorted(value, key=str))
        return (column, "in", values)
    return (str(key), "==", (value,))


def _known_temporal_ref(value, form: str):
    if not isinstance(value, dict) or form not in value:
        return None
    payload = value[form]
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        return None
    return str(payload[0]), int(payload[1])


def _known_term_signature(value):
    if isinstance(value, str):
        return ("ref", value)
    if isinstance(value, (int, float)):
        return ("const", float(value))
    if not isinstance(value, dict):
        return None
    if "delta" in value:
        payload = value["delta"]
        if isinstance(payload, (list, tuple)):
            return ("delta", _known_term_signature(payload[0]), int(payload[1]))
        return ("delta", _known_term_signature(payload), 1)
    if "lag" in value:
        payload = value["lag"]
        if not isinstance(payload, (list, tuple)) or len(payload) != 2:
            return None
        return ("lag", _known_term_signature(payload[0]), int(payload[1]))
    if "roll_sum" in value:
        payload = value["roll_sum"]
        return ("rolling", "SUM", int(payload[1]), _known_term_signature(payload[0]))
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
        int(value["window"]),
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
    return float(value)


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
        if base[0] == "delta_bound" and op in (">=", ">"):
            return ["conditional_positive"]
        if base[0] == "delta_zero":
            return ["conditional_zero"]
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
        return ["monotone"]
    if op in (">=", "<=", ">", "<") and isinstance(inv.lhs, dict) and "delta" in inv.lhs:
        return ["monotone"]
    if base is not None and base[0] == "windowed_ratio":
        return ["windowed_ratio"]
    is_zero = isinstance(rhs, (int, float)) and float(rhs) == 0.0
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



def _col_scale(frame, col: str) -> float:
    """Robust magnitude (median absolute value) of a column's observed data.

    Returns 0.0 for a column the frame does not carry, so an unknown column is never treated as
    negligible (it is kept, which keeps matching conservative).
    """
    if not frame.has(col):
        return 0.0
    v = frame.col(col)
    if v.size == 0:
        return 0.0
    v = v[~np.isnan(v)]
    return float(np.median(np.abs(v))) if v.size else 0.0


def _drop_negligible(cols, anchor_col: str, frame, zero_tol: float) -> frozenset:
    """Drop summed columns whose observed data is negligible against the anchor's scale.

    A column that is (near-)zero across all observations adds ~0 to a sum, so removing it leaves
    the sum -- and therefore the equality it feeds -- unchanged.  Two groupings that differ only by
    such columns describe the *same* physical fact.  The negligibility scale is anchored on the
    reference (left-hand side) column, so the test is dimensionless and dataset-agnostic.  We never
    reduce a whole group to empty (that would collapse distinct laws), and unknown columns are kept.
    """
    scale = _col_scale(frame, anchor_col)
    if scale <= 0.0:
        return frozenset(cols)                       # no usable anchor scale -> do not canonicalize
    thresh = zero_tol * scale
    kept = frozenset(c for c in cols
                     if not (frame.has(c) and _col_scale(frame, c) < thresh))
    return kept if kept else frozenset(cols)         # never canonicalize an entire group away


def _canonicalize(sig, frame, zero_tol: float):
    """Map a relation signature to a data-canonical form (near-zero sum members removed).

    Only the sum-shaped signatures carry groupings, so only they are canonicalized; pairwise,
    zero, presence and one-sided signatures pass through unchanged.  The transform is idempotent
    and strictly widens matching: anything that matched exactly still matches after canonicalizing.
    """
    if not isinstance(sig, tuple) or not sig:
        return sig
    if sig[0] == "equality" and len(sig) == 3:
        return (
            sig[0],
            sig[1],
            _canonicalize(sig[2], frame, zero_tol),
        )
    if sig[0] == "conditional" and len(sig) == 2:
        condition, base = sig[1]
        return (
            "conditional",
            (condition, _canonicalize(base, frame, zero_tol)),
        )
    if sig[0] == "ref_sum":
        ref_col, cols = sig[1]
        return ("ref_sum", (ref_col, _drop_negligible(cols, ref_col, frame, zero_tol)))
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
    try:
        values = eval_term(
            A.Lag(A.Ref(column), int(steps)),
            _binder_for_column(result, column),
            {},
            dataset.observed,
            dataset.name_model,
        )
    except Exception:
        return False
    if values is None:
        return False
    return bool(np.any(np.isfinite(np.asarray(values, dtype=float))))


def _binder_for_column(result: DiscoveryResult, column: str) -> str:
    for ev in result.portfolio:
        if column in ev.rule.unparse():
            return ev.rule.binder
    return "record"


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
    canon_rels = [
        _canonicalize(s, frame, zero_tol)
        for s in rels
    ]
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
                recovered = any(
                    relation_signature_matches(
                        _canonicalize(candidate, frame, zero_tol),
                        learned,
                    )
                    for candidate in _matching_signatures(sig)
                    for learned in canon_rels
                )
        else:
            recovered = any(
                relation_signature_matches(
                    _canonicalize(candidate, frame, zero_tol),
                    learned,
                )
                for candidate in _matching_signatures(sig)
                for learned in canon_rels
            )
        n_ok += int(recovered)
        report.append({"name": inv.name, "op": inv.op, "recovered": bool(recovered),
                       "signature": str(sig)})
    recall = (n_ok / len(known)) if known else 0.0
    return {"recall": recall, "recovered": n_ok, "total": len(known), "invariants": report}
