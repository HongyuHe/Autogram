"""Explicit (schema-aware) rendering of DSL rules for human-facing output.

:meth:`autogram.dsl.ast.Rule.unparse` is the canonical, context-free persisted surface form. It is
structurally round-trippable, deliberately hides the binder's bound variable, and prints roles by
name (``measurement_origination``, ``SUM(demand_col)``). That is compact but opaque: the reader
cannot see that every role is silently a function of the quantified entity.

:func:`render_rule` produces the *explicit* form instead, using the compiled
:class:`~autogram.schema.adapter.SchemaAdapter` to expand every hidden variable:

* the quantifier names its bound variable(s) -- ``[forall node X]``, ``[forall link X, Y]``;
* every single-column ``Ref`` is shown as its grounding template -- ``low_{X}_origination``;
* every family ``Agg`` is shown as an explicit indexed reduction with its membership predicate --
  ``SUM(demand_col)`` becomes ``Σ_{j≠X} high_{j}_{X}``.

The result is a faithful, self-contained reading of the invariant.  Rendering never affects
evaluation, dedup or acceptance; it is output-only, and falls back to ``rule.unparse()`` when no
adapter is available.  :func:`parse_rule_line` is the inverse of ``unparse`` (compact text -> AST),
used to re-render already-saved ``.dl`` portfolios in the explicit form.
"""

from __future__ import annotations

import json
import math
import re
from typing import List, Optional, Tuple

from . import ast as A
from .scalar_codec import scalar_from_text


def _finite_number(text: str, label: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


# Bound-variable name(s) each enumerate strategy introduces (mirrors
# ``SchemaAdapter.enumerate_bindings``: per_node -> {X}, per_directed_link -> {X, Y}, ...).
_STRATEGY_VARS = {
    "per_measured_col": ("col",),
    "per_node": ("X",),
    "per_directed_link": ("X", "Y"),
    "singleton": (),
}

# Reduction glyphs (SUM renders as the summation sign; the rest keep a short prefix).
_AGG_GLYPH = {"SUM": "Σ", "MIN": "min", "MAX": "max", "AVG": "mean"}

# Dummy index letters for family membership, chosen to avoid the binder variables (X/Y/col).
_DUMMY_POOL = ("j", "k", "l", "m", "i", "p", "q", "r")


def _binder_vars(adapter, binder: str) -> Tuple[str, ...]:
    return _STRATEGY_VARS.get(adapter.binder_enumerate.get(binder, ""), ())


def _schema_facts(adapter) -> dict:
    """Directions the schema treats as directed-link vs single-node, read off ref templates."""
    link_dirs, single_dirs = set(), set()
    for (_binder, _role), tmpl in adapter.ref_templates.items():
        m = re.fullmatch(r".+?_\{[XY]\}_(.+)_\{[XY]\}", tmpl)   # <kind>_{X}_<dir>_{Y}
        if m:
            link_dirs.add(m.group(1))
            continue
        m = re.fullmatch(r".+?_\{X\}_(.+)", tmpl)               # <kind>_{X}_<dir> (single node)
        if m and "{" not in m.group(1):
            single_dirs.add(m.group(1))
    return {"link_dirs": link_dirs, "single_dirs": single_dirs}


def _family_skeleton(sel, adapter, facts) -> Optional[Tuple[str, List[str]]]:
    """A column template with named ``{source}``/``{destination}``/``{peer}`` slots for a family."""
    nk, dk = adapter.noisy_kind, adapter.demand_kind
    if sel.match_kind == dk:
        return f"{dk}_{{source}}_{{destination}}", ["source", "destination"]
    d = sel.match_direction
    if d in facts["link_dirs"]:
        return f"{nk}_{{source}}_{d}_{{peer}}", ["source", "peer"]
    if d in facts["single_dirs"] or d is not None:
        return f"{nk}_{{source}}_{d}", ["source"]
    return None


def _render_family(agg: A.Agg, binder: str, adapter, facts) -> str:
    """``SUM(demand_col)`` -> ``Σ_{j≠X} high_{j}_{X}`` using the family selector's predicate."""
    glyph = _AGG_GLYPH.get(agg.kind, agg.kind)
    sel = adapter.family_selectors.get((binder, agg.family_role))
    skel = _family_skeleton(sel, adapter, facts) if sel is not None else None
    if skel is None:                                    # unknown family: keep the compact form
        return agg.unparse()
    template, slots = skel
    bvars = set(_binder_vars(adapter, binder))
    assign: dict = {}
    # 1) slots pinned to a binder variable by an equality predicate (source == X)
    for slot, op, rhs in sel.predicates:
        if op == "==" and rhs in bvars:
            assign[slot] = rhs
    # 2) remaining structural slots get fresh dummy indices
    pool = (d for d in _DUMMY_POOL if d not in bvars)
    for slot in slots:
        if slot not in assign:
            assign[slot] = next(pool)
    # 3) equality ties to another slot (source == @destination)
    for slot, op, rhs in sel.predicates:
        if op == "==" and rhs.startswith("@") and rhs[1:] in assign:
            assign[slot] = assign[rhs[1:]]
    # 4) inequality constraints become the membership condition
    constraints: List[str] = []
    for slot, op, rhs in sel.predicates:
        if op != "!=":
            continue
        if rhs == "*":
            continue
        other = assign[rhs[1:]] if rhs.startswith("@") and rhs[1:] in assign else rhs
        constraints.append(f"{assign[slot]}≠{other}")
    free = [assign[s] for s in slots if assign[s] not in bvars]
    index = ", ".join(constraints) if constraints else ", ".join(dict.fromkeys(free))
    # Brace every substituted index in the body so a bound variable reads uniformly with the
    # single-column templates (``high_{j}_{X}`` alongside ``low_{X}_origination``).
    body = template.format(**{s: "{" + assign[s] + "}" for s in slots})
    return f"{glyph}_{{{index}}} {body}" if index else f"{glyph} {body}"


def _render_term(term: A.Term, binder: str, adapter, facts) -> str:
    if isinstance(term, A.Const):
        return term.unparse()
    if isinstance(term, A.Ref):
        return adapter.ref_templates.get((binder, term.role), term.role)
    if isinstance(term, A.Agg):
        return _render_family(term, binder, adapter, facts)
    if isinstance(term, A.Scale):
        child = _render_term(term.term, binder, adapter, facts)
        if isinstance(term.term, A.Add):
            child = f"({child})"
        return f"{A.Const(term.coeff).unparse()}*{child}"
    if isinstance(term, A.Add):
        if len(term.terms) <= 1:
            children = ", ".join(
                _render_term(child, binder, adapter, facts)
                for child in term.terms
            )
            return f"ADD({children})"
        children = []
        for child in term.terms:
            rendered = _render_term(child, binder, adapter, facts)
            if isinstance(child, A.Add):
                rendered = f"({rendered})"
            children.append(rendered)
        return " + ".join(children)
    if isinstance(term, A.Mul):
        left = _render_term(term.left, binder, adapter, facts)
        right = _render_term(term.right, binder, adapter, facts)
        if isinstance(term.left, A.Add):
            left = f"({left})"
        if isinstance(term.right, A.Add):
            right = f"({right})"
        return f"({left} * {right})"
    if isinstance(term, A.Div):
        num = _render_term(term.num, binder, adapter, facts)
        den = _render_term(term.den, binder, adapter, facts)
        if isinstance(term.num, A.Add):
            num = f"({num})"
        if isinstance(term.den, A.Add):
            den = f"({den})"
        return f"({num} / {den})"
    if isinstance(term, A.Lag):
        return f"B^{term.steps}({_render_term(term.term, binder, adapter, facts)})"
    if isinstance(term, A.Diff):
        suffix = "" if term.steps == 1 else f"_{term.steps}"
        return f"Δ{suffix}({_render_term(term.term, binder, adapter, facts)})"
    if isinstance(term, A.Rolling):
        return (
            f"{term.kind}_{term.window}("
            f"{_render_term(term.term, binder, adapter, facts)})"
        )
    if isinstance(term, A.RelatedAgg):
        return term.unparse()
    return term.unparse()


def render_rule(rule: A.Rule, adapter=None) -> str:
    """Render ``rule`` in explicit form, expanding hidden binder variables via ``adapter``.

    Falls back to :meth:`~autogram.dsl.ast.Rule.unparse` when no adapter is supplied (or the binder
    is unknown to it), so callers without schema context still get the canonical compact form.
    """
    if adapter is None or rule.binder not in getattr(adapter, "binders", ()):
        return rule.unparse()
    facts = _schema_facts(adapter)
    var_list = ", ".join(_binder_vars(adapter, rule.binder))
    head = f"[forall {rule.binder} {var_list}]" if var_list else f"[forall {rule.binder}]"
    if isinstance(rule.atom, A.BooleanDefinition):
        target = _render_term(rule.atom.target, rule.binder, adapter, facts)
        condition = f" where {rule.condition.unparse()}" if rule.condition is not None else ""
        return f"{head} {target} := {rule.atom.predicate.unparse()}{condition}"
    if isinstance(rule.atom, A.CategoryDefinition):
        condition = f" where {rule.condition.unparse()}" if rule.condition is not None else ""
        return f"{head} {rule.atom.unparse()}{condition}"
    if isinstance(rule.atom, A.BandDefinition):
        term = _render_term(rule.atom.term, rule.binder, adapter, facts)
        center = "?" if rule.atom.center is None else A.Const(rule.atom.center).unparse()
        condition = f" where {rule.condition.unparse()}" if rule.condition is not None else ""
        return f"{head} {term} ~band {center}{condition}"
    left = _render_term(rule.atom.left, rule.binder, adapter, facts)
    right = _render_term(rule.atom.right, rule.binder, adapter, facts)
    condition = f" where {rule.condition.unparse()}" if rule.condition is not None else ""
    return f"{head} {left} {rule.atom.op} {right}{condition}"


# ---------------------------------------------------------------------------
# Inverse of Rule.unparse: parse the compact surface form back into an AST.
# ---------------------------------------------------------------------------

_OPS = (" <|> ", " ~∝ ", " ~= ", " == ", " != ", " <= ", " >= ", " < ", " > ")
_AGG_RE = re.compile(r"^(SUM|MIN|MAX|AVG)\((\w+)\)$")
_FINITE_TOKEN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NONFINITE_TOKEN = r"[+-]?(?:nan|inf(?:inity)?)"
_SCALE_RE = re.compile(rf"^({_FINITE_TOKEN})\*(.+)$")
_NONFINITE_SCALE_RE = re.compile(
    rf"^({_NONFINITE_TOKEN})\*(.+)$",
    re.IGNORECASE,
)
_NUM_RE = re.compile(rf"^{_FINITE_TOKEN}$")
_NONFINITE_RE = re.compile(rf"^{_NONFINITE_TOKEN}$", re.IGNORECASE)
_ADD_RE = re.compile(r"^ADD\((.*)\)$")
_LAG_RE = re.compile(r"^LAG_(\d+)\((.*)\)$")
_DIFF_RE = re.compile(r"^DELTA_(\d+)\((.*)\)$")
_ROLL_RE = re.compile(r"^ROLL_(SUM|MIN|MAX|AVG)_(\d+)\((.*)\)$")
_RELATED_RE = re.compile(r"^RELATED\((\w+)\)$")


def _split_top(s: str, sep: str) -> List[str]:
    """Split ``s`` on ``sep`` only outside parentheses and quoted strings."""
    out, depth, last = [], 0, 0
    quoted = False
    escaped = False
    i, n, m = 0, len(s), len(sep)
    while i <= n - m:
        c = s[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
            i += 1
            continue
        if c == '"':
            quoted = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0 and s[i:i + m] == sep:
            out.append(s[last:i])
            last = i + m
            i += m
            continue
        i += 1
    out.append(s[last:])
    return out


def _parse_primary(s: str) -> A.Term:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        for sep, ctor in ((" * ", "mul"), (" / ", "div")):
            parts = _split_top(inner, sep)
            if len(parts) == 2:
                left, right = _parse_primary(parts[0]), _parse_primary(parts[1])
                return A.Mul(left, right) if ctor == "mul" else A.Div(left, right)
        return _parse_term(inner)
    m = _ADD_RE.match(s)
    if m:
        inner = m.group(1)
        if not inner:
            return A.Add(())
        return A.Add(tuple(
            _parse_term(part)
            for part in _split_top(inner, ", ")
        ))
    m = _AGG_RE.match(s)
    if m:
        return A.Agg(m.group(1), m.group(2))
    m = _LAG_RE.match(s)
    if m:
        return A.Lag(_parse_term(m.group(2)), int(m.group(1)))
    m = _DIFF_RE.match(s)
    if m:
        return A.Diff(_parse_term(m.group(2)), int(m.group(1)))
    m = _ROLL_RE.match(s)
    if m:
        return A.Rolling(_parse_term(m.group(3)), int(m.group(2)), m.group(1))
    m = _RELATED_RE.match(s)
    if m:
        return A.RelatedAgg(m.group(1))
    m = _NONFINITE_SCALE_RE.match(s)
    if m:
        _finite_number(m.group(1), "scale coefficient")
    if _NONFINITE_RE.match(s):
        _finite_number(s, "constant")
    m = _SCALE_RE.match(s)
    if m:
        return A.Scale(
            _finite_number(m.group(1), "scale coefficient"),
            _parse_primary(m.group(2)),
        )
    if _NUM_RE.match(s):
        return A.Const(_finite_number(s, "constant"))
    return A.Ref(s)


def _parse_term(s: str) -> A.Term:
    parts = _split_top(s.strip(), " + ")
    if len(parts) > 1:
        return A.Add(tuple(_parse_primary(p) for p in parts))
    return _parse_primary(parts[0])


def parse_rule_line(text: str) -> A.Rule:
    """Parse one compact ``unparse`` line ``[forall <binder>] <left> <op> <right>`` into a Rule."""
    text = text.strip()
    m = re.match(r"^\[forall\s+(\w+)\]\s*(.*)$", text)
    if not m:
        raise ValueError(f"not a rule line: {text!r}")
    binder, body = m.group(1), m.group(2)
    condition = None
    condition_parts = _split_top(body, " where ")
    if len(condition_parts) == 2:
        body, condition_text = condition_parts
        condition = _parse_condition(condition_text)
    elif len(condition_parts) > 2:
        raise ValueError(f"ambiguous condition in rule: {text!r}")
    definition = _split_top(body, " := ")
    if len(definition) == 2:
        target, predicate_text = definition
        if predicate_text.startswith("PRIORITY("):
            atom = _parse_category_definition(target.strip(), predicate_text)
        else:
            atom = A.BooleanDefinition(
                _parse_term(target),
                _parse_predicate(predicate_text),
            )
        return A.Rule(binder, atom, condition=condition)
    band = _split_top(body, " ~band ")
    if len(band) == 2:
        center = (
            None
            if band[1].strip() == "?"
            else _finite_number(band[1].strip(), "band center")
        )
        return A.Rule(
            binder,
            A.BandDefinition(_parse_term(band[0]), center),
            condition=condition,
        )
    for op in _OPS:
        parts = _split_top(body, op)
        if len(parts) == 2:
            left = _parse_term(parts[0])
            right = _parse_term(parts[1])
            return A.Rule(
                binder,
                A.Compare(left, op.strip(), right),
                condition=condition,
            )
    raise ValueError(f"no comparison operator found in: {body!r}")


def _parse_condition(text: str) -> A.Condition:
    text = text.strip()
    all_match = re.fullmatch(r"ALL\((.*)\)", text)
    if all_match:
        return A.Condition(
            "",
            "all",
            tuple(
                _parse_condition(part)
                for part in _split_top(all_match.group(1), ", ")
                if part.strip()
            ),
        )
    try:
        column, remainder = _parse_condition_identifier(text)
    except ValueError as error:
        raise ValueError(f"invalid condition: {text!r}") from error
    match = re.fullmatch(r"\s+in\s+\((.*)\)", remainder)
    if match:
        values = tuple(
            _parse_scalar(part.strip())
            for part in _split_top(match.group(1), ", ")
            if part.strip()
        )
        return A.Condition(column, "in", values)
    match = re.fullmatch(r"\s+(==|!=)\s+(.+)", remainder)
    if not match:
        raise ValueError(f"invalid condition: {text!r}")
    return A.Condition(
        column,
        match.group(1),
        (_parse_scalar(match.group(2).strip()),),
    )


def _parse_identifier(text: str, label: str) -> Tuple[str, str]:
    if text.startswith('"'):
        try:
            identifier, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid quoted {label}") from error
        if not isinstance(identifier, str):
            raise ValueError(f"{label} must be a string")
        return identifier, text[end:]
    match = re.match(r"\w+", text)
    if match is None:
        raise ValueError(f"invalid {label}")
    return match.group(0), text[match.end():]


def _parse_condition_identifier(text: str) -> Tuple[str, str]:
    return _parse_identifier(text, "condition identifier")


def _parse_predicate(text: str) -> A.Predicate:
    text = text.strip()
    if text.startswith("(") and text.endswith(")") and _balanced_outer(text):
        text = text[1:-1].strip()
    conjunction = _split_top(text, " AND ")
    if len(conjunction) > 1:
        return A.Conjunction(tuple(_parse_predicate(part) for part in conjunction))
    sustained = re.fullmatch(r"ALWAYS_(\d+)\((.*)\)", text)
    if sustained:
        predicate = _parse_predicate(sustained.group(2))
        if not isinstance(predicate, A.Bound):
            raise ValueError("ALWAYS requires a bound predicate")
        return A.Sustained(predicate, int(sustained.group(1)))
    for operator in (" <= ", " >= ", " < ", " > "):
        parts = _split_top(text, operator)
        if len(parts) == 2:
            threshold = (
                None
                if parts[1].strip() == "?"
                else _finite_number(
                    parts[1].strip(),
                    "predicate threshold",
                )
            )
            return A.Bound(_parse_term(parts[0]), operator.strip(), threshold)
    raise ValueError(f"invalid Boolean predicate: {text!r}")


def _parse_category_definition(target: str, text: str) -> A.CategoryDefinition:
    text = text.strip()
    if not text.startswith("PRIORITY(") or not text.endswith(")"):
        raise ValueError(f"invalid categorical definition: {text!r}")
    parts = _split_top(text[len("PRIORITY("):-1], "; default=")
    if len(parts) != 2:
        raise ValueError(f"invalid categorical definition: {text!r}")
    target_column, target_remainder = _parse_identifier(
        target.strip(),
        "categorical target identifier",
    )
    if target_remainder.strip():
        raise ValueError(
            f"invalid categorical target identifier: {target!r}"
        )
    cases = []
    case_text = parts[0].strip()
    if case_text:
        for item in _split_top(case_text, ", "):
            column, remainder = _parse_identifier(
                item.strip(),
                "categorical case identifier",
            )
            remainder = remainder.lstrip()
            if not remainder.startswith("->"):
                raise ValueError(
                    f"invalid categorical case: {item!r}"
                )
            value = remainder[2:].strip()
            if not value:
                raise ValueError(
                    f"invalid categorical case: {item!r}"
                )
            cases.append((column, _parse_scalar(value)))
    return A.CategoryDefinition(
        target_column,
        tuple(cases),
        _parse_scalar(parts[1].strip()),
    )


def _parse_scalar(text: str):
    return scalar_from_text(text, "condition/category scalar")


def _balanced_outer(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0
