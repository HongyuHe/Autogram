"""Serialization for DSL rules (the round-trippable canonical form).

Because the AST is plain data, the canonical serialization is a JSON-friendly dict
(``rule_to_dict`` / ``rule_from_dict``); this is what makes a learned rule auditable
and storable.  :meth:`~autogram.dsl.ast.Rule.unparse` provides the human-readable ASCII
surface rendering described in the grammar (Sec. 6.6); the dict form is the machine
contract used by the proposer backends and the archive.
"""

from __future__ import annotations

import math

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
    if isinstance(t, A.Mul):
        return {"k": "Mul", "left": term_to_dict(t.left), "right": term_to_dict(t.right)}
    if isinstance(t, A.Div):
        return {"k": "Div", "num": term_to_dict(t.num), "den": term_to_dict(t.den)}
    if isinstance(t, A.Lag):
        return {"k": "Lag", "steps": t.steps, "term": term_to_dict(t.term)}
    if isinstance(t, A.Diff):
        return {"k": "Diff", "steps": t.steps, "term": term_to_dict(t.term)}
    if isinstance(t, A.Rolling):
        return {"k": "Rolling", "window": t.window, "kind": t.kind, "term": term_to_dict(t.term)}
    if isinstance(t, A.RelatedAgg):
        return {"k": "RelatedAgg", "role": t.role}
    raise TypeError(f"unknown term {t!r}")


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required(mapping: dict, key: str, label: str):
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _name(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if number <= 0 or number != value:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _scalar(value, label: str):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        number = _finite_float(value, label)
        return number
    raise ValueError(f"{label} must be a JSON scalar")


def term_from_dict(d: dict) -> A.Term:
    d = _mapping(d, "term")
    k = _required(d, "k", "term")
    if k == "Const":
        return A.Const(_finite_float(
            _required(d, "value", "constant"),
            "constant",
        ))
    if k == "Ref":
        return A.Ref(_name(
            _required(d, "role", "ref"),
            "ref role",
        ))
    if k == "Scale":
        return A.Scale(
            _finite_float(
                _required(d, "coeff", "scale"),
                "scale coefficient",
            ),
            term_from_dict(_required(d, "term", "scale")),
        )
    if k == "Add":
        terms = tuple(
            term_from_dict(x)
            for x in _required(d, "terms", "Add")
        )
        if not terms:
            raise ValueError("Add requires at least one term")
        return A.Add(terms)
    if k == "Agg":
        kind = _name(
            _required(d, "kind", "aggregation"),
            "aggregation kind",
        )
        if kind not in A.AGG_KINDS:
            raise ValueError(f"unknown aggregation kind {kind!r}")
        return A.Agg(
            kind,
            _name(
                _required(d, "family_role", "aggregation"),
                "family role",
            ),
        )
    if k == "Mul":
        return A.Mul(
            term_from_dict(_required(d, "left", "product")),
            term_from_dict(_required(d, "right", "product")),
        )
    if k == "Div":
        return A.Div(
            term_from_dict(_required(d, "num", "ratio")),
            term_from_dict(_required(d, "den", "ratio")),
        )
    if k == "Lag":
        return A.Lag(
            term_from_dict(_required(d, "term", "lag")),
            _positive_int(
                _required(d, "steps", "lag"),
                "lag steps",
            ),
        )
    if k == "Diff":
        return A.Diff(
            term_from_dict(_required(d, "term", "difference")),
            _positive_int(d.get("steps", 1), "difference steps"),
        )
    if k == "Rolling":
        kind = _name(
            d.get("kind", "SUM"),
            "rolling aggregation kind",
        )
        if kind not in ("SUM", "MIN", "MAX", "AVG"):
            raise ValueError(
                f"unknown rolling aggregation kind {kind!r}"
            )
        return A.Rolling(
            term_from_dict(_required(d, "term", "rolling")),
            _positive_int(
                _required(d, "window", "rolling"),
                "rolling window",
            ),
            kind,
        )
    if k == "RelatedAgg":
        return A.RelatedAgg(_name(
            _required(d, "role", "related aggregation"),
            "related role",
        ))
    raise ValueError(f"unknown term kind {k!r}")


def rule_to_dict(r: A.Rule) -> dict:
    payload = {"binder": r.binder, "tag": r.tag}
    if isinstance(r.atom, A.Compare):
        payload.update({
            "op": r.atom.op,
            "left": term_to_dict(r.atom.left),
            "right": term_to_dict(r.atom.right),
        })
    elif isinstance(r.atom, A.BooleanDefinition):
        payload.update({
            "atom_kind": "BooleanDefinition",
            "target": term_to_dict(r.atom.target),
            "predicate": predicate_to_dict(r.atom.predicate),
        })
    elif isinstance(r.atom, A.CategoryDefinition):
        payload.update({
            "atom_kind": "CategoryDefinition",
            "target_column": r.atom.target_column,
            "cases": [list(case) for case in r.atom.cases],
            "default": r.atom.default,
        })
    elif isinstance(r.atom, A.BandDefinition):
        payload.update({
            "atom_kind": "BandDefinition",
            "term": term_to_dict(r.atom.term),
            "center": r.atom.center,
        })
    else:
        raise TypeError(f"unknown rule atom {r.atom!r}")
    if r.condition is not None:
        payload["condition"] = condition_to_dict(r.condition)
    return payload


def rule_from_dict(d: dict, *, grammar=None) -> A.Rule:
    d = _mapping(d, "rule")
    kind = d.get("atom_kind", "Compare")
    if kind == "Compare":
        op = _name(
            _required(d, "op", "comparison"),
            "comparison operator",
        )
        if op not in A.OPS:
            raise ValueError(f"unknown comparison operator {op!r}")
        atom = A.Compare(
            term_from_dict(_required(d, "left", "comparison")),
            op,
            term_from_dict(_required(d, "right", "comparison")),
        )
    elif kind == "BooleanDefinition":
        atom = A.BooleanDefinition(
            term_from_dict(_required(
                d,
                "target",
                "Boolean definition",
            )),
            predicate_from_dict(_required(
                d,
                "predicate",
                "Boolean definition",
            )),
        )
    elif kind == "CategoryDefinition":
        target_column = _name(
            _required(
                d,
                "target_column",
                "category definition",
            ),
            "category target column",
        )
        cases = []
        for case in d.get("cases", ()):
            if not isinstance(case, (list, tuple)) or len(case) != 2:
                raise ValueError(
                    "category cases must be [column, value] pairs"
                )
            cases.append((
                _name(case[0], "category case column"),
                _scalar(case[1], "category case value"),
            ))
        if not cases:
            raise ValueError(
                "category definition requires at least one case"
            )
        atom = A.CategoryDefinition(
            target_column,
            tuple(cases),
            _scalar(
                _required(
                    d,
                    "default",
                    "category definition",
                ),
                "category default",
            ),
        )
    elif kind == "BandDefinition":
        center = d.get("center")
        atom = A.BandDefinition(
            term_from_dict(_required(d, "term", "band definition")),
            None if center is None else _finite_float(center, "band center"),
        )
    else:
        raise ValueError(f"unknown atom kind {kind!r}")
    condition = None
    if d.get("condition") is not None:
        condition = condition_from_dict(d["condition"])
    rule = A.Rule(
        _name(_required(d, "binder", "rule"), "binder"),
        atom,
        tag=str(d.get("tag", "")),
        condition=condition,
    )
    if grammar is not None:
        from .typecheck import is_admissible

        admissible, reason = is_admissible(rule, grammar)
        if not admissible:
            raise ValueError(f"inadmissible rule: {reason}")
    return rule


def condition_to_dict(condition: A.Condition) -> dict:
    if condition.op == "all":
        values = [
            condition_to_dict(value)
            for value in condition.values
            if isinstance(value, A.Condition)
        ]
    else:
        values = list(condition.values)
    return {
        "column": condition.column,
        "op": condition.op,
        "values": values,
    }


def condition_from_dict(raw: dict) -> A.Condition:
    raw = _mapping(raw, "condition")
    op = _name(
        _required(raw, "op", "condition"),
        "condition operator",
    )
    if op == "all":
        values = tuple(
            condition_from_dict(value)
            for value in raw.get("values", ())
        )
        if not (2 <= len(values) <= 16):
            raise ValueError(
                "condition conjunction must contain between 2 and 16 child conditions"
            )
        if any(child.op == "all" for child in values):
            raise ValueError("condition conjunctions may not nest")
        return A.Condition("", op, values)
    if op not in ("==", "in"):
        raise ValueError(f"unknown condition operator {op!r}")
    values = tuple(
        _scalar(value, "condition value")
        for value in raw.get("values", ())
    )
    if not values:
        raise ValueError("condition requires at least one value")
    if op == "==" and len(values) != 1:
        raise ValueError(
            "condition equality requires exactly one value"
        )
    return A.Condition(
        _name(raw.get("column"), "condition column"),
        op,
        values,
    )


def predicate_to_dict(predicate: A.Predicate) -> dict:
    if isinstance(predicate, A.Bound):
        return {
            "k": "Bound",
            "term": term_to_dict(predicate.term),
            "op": predicate.op,
            "threshold": predicate.threshold,
        }
    if isinstance(predicate, A.Sustained):
        return {
            "k": "Sustained",
            "window": predicate.window,
            "predicate": predicate_to_dict(predicate.predicate),
        }
    if isinstance(predicate, A.Conjunction):
        return {
            "k": "Conjunction",
            "predicates": [predicate_to_dict(item) for item in predicate.predicates],
        }
    raise TypeError(f"unknown predicate {predicate!r}")


def predicate_from_dict(payload: dict) -> A.Predicate:
    payload = _mapping(payload, "predicate")
    kind = _required(payload, "k", "predicate")
    if kind == "Bound":
        threshold = payload.get("threshold")
        op = _name(
            _required(payload, "op", "bound predicate"),
            "predicate operator",
        )
        if op not in ("<", "<=", ">", ">="):
            raise ValueError(f"unknown predicate operator {op!r}")
        return A.Bound(
            term_from_dict(_required(
                payload,
                "term",
                "bound predicate",
            )),
            op,
            (
                None
                if threshold is None
                else _finite_float(threshold, "predicate threshold")
            ),
        )
    if kind == "Sustained":
        inner = predicate_from_dict(_required(
            payload,
            "predicate",
            "sustained predicate",
        ))
        if not isinstance(inner, A.Bound):
            raise ValueError("Sustained requires a Bound predicate")
        return A.Sustained(
            inner,
            _positive_int(
                _required(
                    payload,
                    "window",
                    "sustained predicate",
                ),
                "sustained window",
            ),
        )
    if kind == "Conjunction":
        predicates = tuple(
            predicate_from_dict(item)
            for item in payload.get("predicates", ())
        )
        if len(predicates) < 2:
            raise ValueError(
                "conjunction requires at least two predicates"
            )
        return A.Conjunction(predicates)
    raise ValueError(f"unknown predicate kind {kind!r}")
