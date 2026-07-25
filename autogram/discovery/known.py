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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_ref
from .loop import DiscoveryResult
from .validate import portfolio_relations


@dataclass
class KnownInvariant:
    name: str
    op: str
    lhs: str
    rhs: object          # column name (str), {"sum": [...]}, or a number (0)


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
            lhs=str(e["lhs"]),
            rhs=e.get("rhs"),
        ))
    return out


def _signature(inv: KnownInvariant):
    op, lhs, rhs = inv.op, inv.lhs, inv.rhs
    is_zero = isinstance(rhs, (int, float)) and float(rhs) == 0.0
    if op in ("~=", "==") and is_zero:
        return ("zero", lhs)
    if op in ("~=", "==") and isinstance(rhs, dict) and "sum" in rhs:
        return ("ref_sum", (lhs, frozenset(str(c) for c in rhs["sum"])))
    if op in ("~=", "==") and isinstance(rhs, str):
        return ("pair", frozenset({lhs, rhs}))
    if op == "<|>" and isinstance(rhs, str):
        return ("presence_pair", frozenset({lhs, rhs}))
    if op in (">=", "<=") and is_zero:
        return ("one_sided", lhs, op)
    return None


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
    is_zero = isinstance(rhs, (int, float)) and float(rhs) == 0.0
    if op in ("~=", "==") and is_zero:
        return ["self_zero"]
    if op in ("~=", "==") and isinstance(rhs, dict) and "sum" in rhs:
        return ["row_sum", "col_sum"]
    if op == "==" and isinstance(rhs, str):
        return ["two_end"]
    if op == "~=" and isinstance(rhs, str):
        return ["offset_pair"]
    if op == "<|>" and isinstance(rhs, str):
        return ["presence_pair"]
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
        atom = ev.rule.atom
        if atom.op != op:
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
    if sig[0] == "ref_sum":
        ref_col, cols = sig[1]
        return ("ref_sum", (ref_col, _drop_negligible(cols, ref_col, frame, zero_tol)))
    if sig[0] == "agg_ref_balance":
        sides = frozenset(
            (ref, _drop_negligible(fam, ref, frame, zero_tol)) for (ref, fam) in sig[1]
        )
        return ("agg_ref_balance", sides)
    return sig


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
    canon_rels = {_canonicalize(s, frame, zero_tol) for s in rels}
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
        else:
            recovered = _canonicalize(sig, frame, zero_tol) in canon_rels
        n_ok += int(recovered)
        report.append({"name": inv.name, "op": inv.op, "recovered": bool(recovered),
                       "signature": str(sig)})
    recall = (n_ok / len(known)) if known else 0.0
    return {"recall": recall, "recovered": n_ok, "total": len(known), "invariants": report}
