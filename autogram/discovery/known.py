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


def recover_known(result: DiscoveryResult, known: List[KnownInvariant]) -> dict:
    """Report per-invariant recovery + aggregate recall of the user's known invariants."""
    rels = portfolio_relations(result)
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
            recovered = sig in rels
        n_ok += int(recovered)
        report.append({"name": inv.name, "op": inv.op, "recovered": bool(recovered),
                       "signature": str(sig)})
    recall = (n_ok / len(known)) if known else 0.0
    return {"recall": recall, "recovered": n_ok, "total": len(known), "invariants": report}
