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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_ref
from ..dsl.evaluate import (
    canonical_typed_value,
    eval_term,
    is_missing_scalar,
    typed_condition_key,
    typed_group_key,
    typed_signature_value,
    typed_unique,
)
from ..dsl.scalar_codec import scalar_from_json
from .loop import DiscoveryResult
from .propose import (
    canonical_executable_condition,
    EXECUTABLE_CONDITION_GRAMMAR,
    condition_family_is_enumerable,
    condition_conjunction_arity_error,
)
from .validate import (
    _equality_relation,
    relation_signature_matches,
    _unwrap_equality_relation,
    portfolio_relations,
)

_KNOWN_COMPARISON_OPERATORS = ("<", "<=", ">", ">=")
_KNOWN_DEFINITION_FORMS = ("sustained", "and", "priority")
_KNOWN_INVARIANT_REQUIRED_KEYS = ("name", "op", "lhs", "rhs")
_KNOWN_INVARIANT_OPTIONAL_KEYS = ("where",)
_KNOWN_LHS_FORMS = ("sum", "lag", "delta")
_KNOWN_RELATION_RHS_FORMS = ("center", "sum", "ratio", "related")
_KNOWN_TERM_FORMS = ("delta", "lag", "roll_sum", "difference")
_KNOWN_MAX_DEFINITION_CONJUNCTION = 16


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


def _known_column_name(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a column name")
    return value


def _known_role_name(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a role name")
    return value


def _known_comparison_operator(value, label: str) -> str:
    if (
        not isinstance(value, str)
        or value not in _KNOWN_COMPARISON_OPERATORS
    ):
        supported = ", ".join(
            repr(operator)
            for operator in _KNOWN_COMPARISON_OPERATORS
        )
        raise ValueError(
            f"{label} must be a string in the supported comparison "
            f"set {supported}"
        )
    return value


def _known_typed_scalar(value, label: str):
    try:
        return canonical_typed_value(
            scalar_from_json(value, label)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a JSON scalar "
            "(finite number, string, boolean, null, or supported "
            "temporal scalar)"
        ) from error


def _known_category_scalar(value, label: str):
    value = _known_typed_scalar(value, label)
    if value is None:
        return None
    if not is_missing_scalar(value):
        return value
    raise ValueError(
        f"{label} must be a JSON scalar "
        "(finite number, string, boolean, null, or supported "
        "non-missing temporal scalar)"
    )


def _known_sum_columns(value, label: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list of columns")
    columns = tuple(
        _known_column_name(column, f"{label} member")
        for column in value
    )
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label} contains duplicate columns")
    return frozenset(columns)


def _known_sequence(value, label: str, *, arity: int | None = None):
    if not isinstance(value, (list, tuple)):
        requirement = (
            f"exactly {arity} items"
            if arity is not None
            else "one or more items"
        )
        raise ValueError(
            f"{label} must be a list or tuple with {requirement}"
        )
    if arity is not None and len(value) != arity:
        raise ValueError(
            f"{label} must be a list or tuple with exactly "
            f"{arity} items"
        )
    if arity is None and not value:
        raise ValueError(
            f"{label} must be a list or tuple with one or more items"
        )
    return tuple(value)


def _known_structured_form(
    value,
    label: str,
    discriminators: tuple[str, ...],
) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    forms = tuple(
        discriminator
        for discriminator in discriminators
        if discriminator in value
    )
    if len(forms) != 1:
        supported = ", ".join(repr(form) for form in discriminators)
        raise ValueError(
            f"{label} must contain exactly one discriminator from "
            f"{supported}"
        )
    form = forms[0]
    extra_keys = tuple(
        sorted(
            (key for key in value if key != form),
            key=repr,
        )
    )
    if extra_keys:
        rendered = ", ".join(repr(key) for key in extra_keys)
        raise ValueError(
            f"{label} has unexpected key(s): {rendered}"
        )
    return form


def _known_exact_mapping_keys(
    value,
    label: str,
    required_keys: tuple[str, ...],
    optional_keys: tuple[str, ...] = (),
):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    missing_keys = tuple(
        key
        for key in required_keys
        if key not in value
    )
    if missing_keys:
        rendered = ", ".join(repr(key) for key in missing_keys)
        raise ValueError(
            f"{label} is missing required key(s): {rendered}"
        )
    extra_keys = tuple(
        sorted(
            (
                key
                for key in value
                if key not in required_keys
                and key not in optional_keys
            ),
            key=repr,
        )
    )
    if extra_keys:
        rendered = ", ".join(repr(key) for key in extra_keys)
        raise ValueError(
            f"{label} has unexpected key(s): {rendered}"
        )
    return value


def _known_definition_form(rhs) -> str:
    if not isinstance(rhs, dict):
        raise ValueError("definition rhs must be a mapping")
    forms = tuple(
        form
        for form in _KNOWN_DEFINITION_FORMS
        if form in rhs
    )
    if len(forms) != 1:
        raise ValueError(
            "definition rhs must contain exactly one of "
            "'sustained', 'and', or 'priority'"
        )
    form = forms[0]
    if form == "priority" and "default" not in rhs:
        raise ValueError(
            "priority definition rhs must contain an explicit "
            "'default' key"
        )
    allowed_keys = (
        {form, "default"}
        if form == "priority"
        else {form}
    )
    extra_keys = tuple(
        sorted(
            (key for key in rhs if key not in allowed_keys),
            key=repr,
        )
    )
    if extra_keys:
        rendered = ", ".join(repr(key) for key in extra_keys)
        raise ValueError(
            f"{form} definition rhs has unexpected top-level "
            f"key(s): {rendered}"
        )
    return form


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
        _known_exact_mapping_keys(
            e,
            f"known invariant entry {i}",
            _KNOWN_INVARIANT_REQUIRED_KEYS,
            _KNOWN_INVARIANT_OPTIONAL_KEYS,
        )
        name = str(e["name"])
        where = e.get("where")
        if "where" in e:
            try:
                where = _canonical_known_condition(where)
            except ValueError as error:
                raise ValueError(
                    f"known invariant {name!r} has an invalid condition: "
                    f"{error}"
                ) from error
        invariant = KnownInvariant(
            name=name,
            op=e["op"],
            lhs=e["lhs"],
            rhs=e["rhs"],
            where=where,
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
    family = _known_condition_family(base)
    if not condition_family_is_enumerable(family):
        rendered = (
            family.replace("_", " ")
            if isinstance(family, str)
            else "this relation"
        )
        raise ValueError(
            f"conditions are not enumerable for {rendered} relations"
        )
    condition = _known_condition_signature(inv.where)
    return None if condition is None else ("conditional", (condition, base))


def _known_condition_family(base) -> str | None:
    _strength, relation = _unwrap_equality_relation(base)
    if not isinstance(relation, tuple) or not relation:
        return None
    kind = relation[0]
    if kind == "pair" and (
        len(relation) != 2
        or len(relation[1]) != 2
    ):
        return None
    if kind == "proportional" and (
        len(relation) != 2
        or len(relation[1]) != 2
        or relation[1][0] == relation[1][1]
    ):
        return None
    return kind


def _base_signature(inv: KnownInvariant):
    op, lhs, rhs = inv.op, inv.lhs, inv.rhs
    is_zero = _known_is_zero(rhs)
    lhs_form = (
        _known_structured_form(
            lhs,
            "structured lhs",
            _KNOWN_LHS_FORMS,
        )
        if isinstance(lhs, dict)
        else None
    )
    rhs_form = (
        _known_structured_form(
            rhs,
            "structured rhs",
            _KNOWN_RELATION_RHS_FORMS,
        )
        if op != ":=" and isinstance(rhs, dict)
        else None
    )
    if (
        op == "~band"
        and isinstance(lhs, str)
        and rhs_form == "center"
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
        and lhs_form == "sum"
        and rhs_form == "sum"
    ):
        left_columns = _known_sum_columns(lhs["sum"], "left sum")
        right_columns = _known_sum_columns(rhs["sum"], "right sum")
        return _equality_relation(
            op,
            (
                "sum_balance",
                frozenset({
                    left_columns,
                    right_columns,
                }),
            ),
        )
    if (
        op in ("~=", "==")
        and rhs_form == "related"
    ):
        return _equality_relation(
            op,
            (
                "related_aggregate",
                (
                    _known_column_name(lhs, "related aggregate lhs"),
                    _known_role_name(
                        rhs["related"],
                        "related aggregate operand",
                    ),
                ),
            ),
        )
    if op == ":=":
        form = _known_definition_form(rhs)
        if not isinstance(lhs, str):
            return None
        if form == "sustained":
            predicate = _known_sustained_signature(rhs["sustained"])
            return None if predicate is None else (
                "sustained_definition",
                (lhs, predicate),
            )
        if form == "and":
            if not isinstance(rhs["and"], (list, tuple)):
                raise ValueError(
                    "conjunction definition must be a list or tuple"
                )
            items = tuple(rhs["and"])
            predicates = tuple(sorted(
                (_known_bound_signature(item) for item in items),
                key=str,
            ))
            if not (
                2
                <= len(predicates)
                <= _KNOWN_MAX_DEFINITION_CONJUNCTION
            ):
                raise ValueError(
                    "conjunction definition must contain between 2 and "
                    f"{_KNOWN_MAX_DEFINITION_CONJUNCTION} bound predicates"
                )
            if all(predicate is not None for predicate in predicates):
                return ("conjunction_definition", (lhs, predicates))
        if form == "priority":
            items = _known_sequence(
                rhs["priority"],
                "priority cases",
            )
            if any(
                not isinstance(item, dict)
                or "when" not in item
                or "value" not in item
                for item in items
            ):
                raise ValueError(
                    "each priority case must be a mapping with "
                    "'when' and 'value'"
                )
            for item in items:
                _known_exact_mapping_keys(
                    item,
                    "priority case",
                    ("when", "value"),
                )
            cases = tuple(
                (
                    _known_role_name(
                        item["when"],
                        "priority case 'when'",
                    ),
                    _known_category_scalar(
                        item["value"],
                        "priority case 'value'",
                    ),
                )
                for item in items
            )
            return A.category_definition_semantic_signature(
                lhs,
                cases,
                _known_category_scalar(
                    rhs["default"],
                    "priority default",
                ),
                typed_signature_value,
            )
    if (
        op in (">=", "<=", ">", "<")
        and is_zero
        and lhs_form == "lag"
    ):
        column, steps = _known_sequence(
            lhs["lag"],
            "lag",
            arity=2,
        )
        return (
            "lag_bound",
            (
                _known_column_name(column, "lag operand"),
                _known_positive_int(steps, "lag steps"),
                op,
            ),
        )
    if op in (">=", "<=", ">", "<") and is_zero and lhs_form == "delta":
        value = lhs["delta"]
        if isinstance(value, (list, tuple)):
            column, steps = _known_sequence(
                value,
                "delta",
                arity=2,
            )
        else:
            column, steps = value, 1
        return (
            "delta_bound",
            (
                _known_column_name(column, "delta operand"),
                _known_positive_int(steps, "delta steps"),
                op,
            ),
        )
    if op in ("~=", "==") and is_zero and lhs_form == "delta":
        value = lhs["delta"]
        if isinstance(value, (list, tuple)):
            column, steps = _known_sequence(
                value,
                "delta",
                arity=2,
            )
        else:
            column, steps = value, 1
        return _equality_relation(
            op,
            (
                "delta_zero",
                (
                    _known_column_name(column, "delta operand"),
                    _known_positive_int(steps, "delta steps"),
                ),
            ),
        )
    if op in ("~=", "==") and isinstance(lhs, str) and rhs_form == "ratio":
        values = _known_sequence(
            rhs["ratio"],
            "ratio",
            arity=2,
        )
        num = _known_temporal_ref(values[0], "roll_sum")
        den = _known_temporal_ref(values[1], "roll_sum")
        if num is not None and den is not None and num[1] == den[1]:
            return _equality_relation(
                op,
                ("windowed_ratio", (lhs, num[0], den[0], num[1])),
            )
    if op in ("~=", "==") and is_zero:
        return _equality_relation(
            op,
            ("zero", _known_column_name(lhs, "zero lhs")),
        )
    if op in ("~=", "==") and rhs_form == "sum":
        return _equality_relation(
            op,
            (
                "ref_sum",
                (
                    _known_column_name(lhs, "sum lhs"),
                    _known_sum_columns(rhs["sum"], "right sum"),
                ),
            ),
        )
    if op in ("~=", "==") and rhs_form == "ratio":
        result_column = _known_column_name(lhs, "ratio lhs")
        values = _known_sequence(
            rhs["ratio"],
            "ratio",
            arity=2,
        )
        if any(not isinstance(value, str) for value in values):
            raise ValueError(
                "ratio operands must be column names"
            )
        return _equality_relation(
            op,
            ("ratio", (result_column, values[0], values[1])),
        )
    if op in ("~=", "==") and isinstance(rhs, str):
        left_column = _known_column_name(lhs, "equality lhs")
        return _equality_relation(
            op,
            ("pair", frozenset({left_column, rhs})),
        )
    if op == "~∝" and isinstance(rhs, str):
        return (
            "proportional",
            (_known_column_name(lhs, "proportional lhs"), rhs),
        )
    if op == "!=" and isinstance(lhs, str) and isinstance(rhs, str):
        return ("separation_pair", frozenset({lhs, rhs}))
    if op == "<|>" and isinstance(rhs, str):
        return (
            "presence_pair",
            frozenset({
                _known_column_name(lhs, "presence lhs"),
                rhs,
            }),
        )
    if op in (">=", "<=") and is_zero:
        return (
            "one_sided",
            _known_column_name(lhs, "one-sided lhs"),
            op,
        )
    return None


def _parse_known_condition(where, *, allow_composite: bool = True):
    if not isinstance(where, dict) or len(where) != 1:
        raise ValueError(
            "condition must be a mapping with exactly one key"
        )
    key, value = next(iter(where.items()))
    if not isinstance(key, str) or not key:
        raise ValueError(
            "condition column must be a non-empty string"
        )
    if key == "all":
        if not allow_composite:
            raise ValueError("condition conjunctions may not nest")
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "condition conjunction must be a list or tuple"
            )
        if len(value) not in (
            EXECUTABLE_CONDITION_GRAMMAR.conjunction_arities
        ):
            raise ValueError(condition_conjunction_arity_error())
        return A.Condition(
            "",
            "all",
            tuple(
                _parse_known_condition(
                    item,
                    allow_composite=False,
                )
                for item in value
            ),
        )
    if key.endswith("_in"):
        column = key[:-3]
        if not column:
            raise ValueError(
                "membership condition column must be a non-empty string"
            )
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(
                "membership condition requires at least one value"
            )
        canonical = []
        for item in value:
            scalar = _known_condition_scalar(item)
            if scalar is None:
                raise ValueError(
                    "membership condition values must be non-missing "
                    "JSON scalars"
                )
            canonical.append(scalar)
        return A.Condition(column, "in", tuple(canonical))
    canonical = _known_condition_scalar(value)
    if canonical is None:
        raise ValueError(
            "condition value must be a non-missing JSON scalar"
        )
    return A.Condition(key, "==", (canonical,))


def _known_condition_ast(where) -> A.Condition:
    condition = _parse_known_condition(where)
    if condition.op != "all":
        return canonical_executable_condition(condition)
    children = tuple(
        canonical_executable_condition(child)
        for child in condition.values
    )
    if any(
        child.op
        not in EXECUTABLE_CONDITION_GRAMMAR.conjunction_child_ops
        for child in children
    ):
        raise ValueError(
            "condition conjunction children must use equality conditions"
        )
    columns = tuple(
        typed_group_key(child.column)
        for child in children
    )
    if len(columns) != len(set(columns)):
        raise ValueError(
            "condition conjunction must use distinct columns"
        )
    return A.Condition(
        "",
        "all",
        tuple(sorted(
            children,
            key=lambda child: repr(typed_condition_key(child)),
        )),
    )


def _known_condition_mapping(condition: A.Condition):
    if condition.op == "all":
        return {
            "all": [
                _known_condition_mapping(child)
                for child in condition.values
                if isinstance(child, A.Condition)
            ],
        }
    if condition.op == "in":
        return {
            f"{condition.column}_in": list(condition.values),
        }
    return {
        condition.column: condition.values[0],
    }


def _canonical_known_condition(where):
    return _known_condition_mapping(_known_condition_ast(where))


def validate_known_conditions(
    known: Sequence[KnownInvariant],
    condition_columns: Mapping[object, Sequence[object]],
    max_condition_values: int,
) -> None:
    """Validate known conditions against the exact runtime condition grammar."""
    if (
        not isinstance(max_condition_values, int)
        or isinstance(max_condition_values, bool)
        or max_condition_values <= 0
    ):
        raise ValueError(
            "max_condition_values must be a positive integer"
        )
    domains = {
        column: typed_unique(values, drop_missing=True)
        for column, values in condition_columns.items()
    }
    for invariant in known:
        if invariant.where is None:
            continue
        try:
            condition = canonical_executable_condition(
                _known_condition_ast(invariant.where),
                condition_columns=domains,
                max_condition_values=max_condition_values,
            )
            simple_conditions = (
                condition.values
                if condition.op == "all"
                else (condition,)
            )
            for simple in simple_conditions:
                domain = domains[simple.column]
                if len(domain) <= 1:
                    raise ValueError(
                        f"condition column {simple.column!r} does not "
                        "have an enumerable multi-value domain"
                    )
        except ValueError as error:
            raise ValueError(
                f"known invariant {invariant.name!r} has an "
                f"infeasible runtime condition: {error}"
            ) from error


def _known_condition_signature(where):
    try:
        condition = _known_condition_ast(where)
    except ValueError:
        return None
    if condition.op == "all":
        children = tuple(sorted(
            (
                _known_condition_signature(
                    _known_condition_mapping(child)
                )
                for child in condition.values
                if isinstance(child, A.Condition)
            ),
            key=str,
        ))
        return ("all", children)
    values = tuple(
        typed_signature_value(item)
        for item in condition.values
    )
    return (condition.column, condition.op, values)


def _known_condition_scalar(value):
    try:
        value = _known_typed_scalar(value, "condition value")
    except ValueError:
        return None
    if is_missing_scalar(value):
        return None
    return value


def _known_temporal_ref(value, form: str):
    if not isinstance(value, dict):
        return None
    _known_structured_form(
        value,
        f"{form} reference",
        (form,),
    )
    payload = _known_sequence(
        value[form],
        form,
        arity=2,
    )
    if not isinstance(payload[0], str):
        raise ValueError(
            f"{form} operand must be a column name"
        )
    return (
        payload[0],
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
    form = _known_structured_form(
        value,
        "term",
        _KNOWN_TERM_FORMS,
    )
    if form == "delta":
        payload = value["delta"]
        if isinstance(payload, (list, tuple)):
            payload = _known_sequence(
                payload,
                "delta",
                arity=2,
            )
            term = _known_term_signature(payload[0])
            if term is None:
                raise ValueError(
                    "delta operand must be a supported term"
                )
            return (
                "delta",
                term,
                _known_positive_int(payload[1], "delta steps"),
            )
        term = _known_term_signature(payload)
        if term is None:
            raise ValueError(
                "delta operand must be a supported term"
            )
        return ("delta", term, 1)
    if form == "lag":
        payload = _known_sequence(
            value["lag"],
            "lag",
            arity=2,
        )
        term = _known_term_signature(payload[0])
        if term is None:
            raise ValueError(
                "lag operand must be a supported term"
            )
        return (
            "lag",
            term,
            _known_positive_int(payload[1], "lag steps"),
        )
    if form == "roll_sum":
        payload = _known_sequence(
            value["roll_sum"],
            "roll_sum",
            arity=2,
        )
        term = _known_term_signature(payload[0])
        if term is None:
            raise ValueError(
                "roll_sum operand must be a supported term"
            )
        return (
            "rolling",
            "SUM",
            _known_positive_int(payload[1], "rolling window"),
            term,
        )
    if form == "difference":
        left, right = _known_sequence(
            value["difference"],
            "difference",
            arity=2,
        )
        left_signature = _known_term_signature(left)
        right_signature = _known_term_signature(right)
        if left_signature is None or right_signature is None:
            raise ValueError(
                "difference operands must be supported terms"
            )
        return (
            "difference",
            left_signature,
            right_signature,
        )
    return None


def _known_bound_signature(value):
    if not isinstance(value, dict):
        return None
    _known_structured_form(
        value,
        "bound predicate",
        ("bound",),
    )
    term, op, threshold = _known_sequence(
        value["bound"],
        "bound",
        arity=3,
    )
    operator = _known_comparison_operator(op, "bound operator")
    term_signature = _known_term_signature(term)
    return None if term_signature is None else (
        "bound",
        term_signature,
        operator,
        _threshold_signature(threshold),
    )


def _known_sustained_signature(value):
    _known_exact_mapping_keys(
        value,
        "sustained predicate",
        ("term", "op", "threshold", "window"),
    )
    operator = _known_comparison_operator(
        value.get("op"),
        "sustained operator",
    )
    term_signature = _known_term_signature(value.get("term"))
    if term_signature is None:
        return None
    return (
        "sustained",
        _known_positive_int(value["window"], "sustained window"),
        (
            "bound",
            term_signature,
            operator,
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
    direction_ops = (
        {">"} if lag_op == ">"
        else {"<"} if lag_op == "<"
        else {">=", ">"} if lower
        else {"<=", "<"}
    )
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
        canonical = _drop_negligible(
            cols, ref_col, frame, zero_tol, exact=bool(exact),
        )
        if len(canonical) == 1:
            return (
                "pair",
                frozenset({
                    ref_col,
                    next(iter(canonical)),
                }),
            )
        return (
            "ref_sum",
            (
                ref_col,
                canonical,
            ),
        )
    if sig[0] == "sum_balance":
        groups = tuple(
            frozenset(
                column
                for column in group
                if not (
                    len(group) > 1
                    and frame.has(column)
                    and np.all(np.isfinite(frame.col(column)))
                    and not np.any(frame.col(column) != 0.0)
                )
            ) or group
            for group in sig[1]
        )
        singletons = [group for group in groups if len(group) == 1]
        if len(groups) == 2 and len(singletons) == 2:
            return (
                "pair",
                frozenset(
                    next(iter(group))
                    for group in groups
                ),
            )
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
            if len(canonical_other) == 1:
                return (
                    "pair",
                    frozenset({
                        anchor,
                        next(iter(canonical_other)),
                    }),
                )
            return (
                "ref_sum",
                (anchor, canonical_other),
            )
        return (
            "sum_balance",
            frozenset(groups),
        )
    if sig[0] == "agg_ref_balance":
        if any(ref in fam for ref, fam in sig[1]):
            return sig
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
    tolerance_exact = _candidate_is_exact(sig)
    for candidate in _matching_signatures(sig):
        canon_candidate = _canonicalize(
            candidate,
            frame,
            zero_tol,
            exact=tolerance_exact,
        )
        for learned in canon_by_tolerance[tolerance_exact]:
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
    rels = portfolio_relations(
        result,
        require_exact_definition_masks=True,
    )
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
