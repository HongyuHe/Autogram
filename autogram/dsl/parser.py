"""Serialization for DSL rules (the round-trippable canonical form).

Because the AST is plain data, the canonical serialization is a JSON-friendly dict
(``rule_to_dict`` / ``rule_from_dict``); this is what makes a learned rule auditable
and storable.  :meth:`~autogram.dsl.ast.Rule.unparse` provides the human-readable ASCII
surface rendering described in the grammar (Sec. 6.6); the dict form is the machine
contract used by the proposer backends and the archive.
"""

from __future__ import annotations

from . import ast as A


def term_to_dict(t: A.Term) -> dict:
    if isinstance(t, A.Const):
        return {"k": "Const", "value": t.value}
    if isinstance(t, A.Ref):
        return {"k": "Ref", "role": t.role}
    if isinstance(t, A.Scale):
        return {"k": "Scale", "coeff": t.coeff, "term": term_to_dict(t.term)}
    if isinstance(t, A.Add):
        return {"k": "Add", "terms": [term_to_dict(x) for x in t.terms]}
    if isinstance(t, A.Agg):
        return {"k": "Agg", "kind": t.kind, "family_role": t.family_role}
    raise TypeError(f"unknown term {t!r}")


def term_from_dict(d: dict) -> A.Term:
    k = d["k"]
    if k == "Const":
        return A.Const(float(d["value"]))
    if k == "Ref":
        return A.Ref(d["role"])
    if k == "Scale":
        return A.Scale(float(d["coeff"]), term_from_dict(d["term"]))
    if k == "Add":
        return A.Add(tuple(term_from_dict(x) for x in d["terms"]))
    if k == "Agg":
        return A.Agg(d["kind"], d["family_role"])
    raise ValueError(f"unknown term kind {k!r}")


def rule_to_dict(r: A.Rule) -> dict:
    return {
        "binder": r.binder,
        "op": r.atom.op,
        "left": term_to_dict(r.atom.left),
        "right": term_to_dict(r.atom.right),
        "tag": r.tag,
    }


def rule_from_dict(d: dict) -> A.Rule:
    atom = A.Compare(term_from_dict(d["left"]), d["op"], term_from_dict(d["right"]))
    return A.Rule(d["binder"], atom, tag=d.get("tag", ""))
