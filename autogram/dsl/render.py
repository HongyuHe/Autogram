"""Explicit (schema-aware) rendering of DSL rules for human-facing output.

:meth:`autogram.dsl.ast.Rule.unparse` is the *canonical, context-free* surface form used for
signatures, dedup and serialization -- it deliberately hides the binder's bound variable and
prints roles by name (``measurement_origination``, ``SUM(demand_col)``).  That is compact but
opaque: the reader cannot see that every role is silently a function of the quantified entity.

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

import re
from typing import List, Optional, Tuple

from . import ast as A

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
        return f"{A.Const(term.coeff).unparse()}*{_render_term(term.term, binder, adapter, facts)}"
    if isinstance(term, A.Add):
        return " + ".join(_render_term(t, binder, adapter, facts) for t in term.terms)
    if isinstance(term, A.Mul):
        return (f"({_render_term(term.left, binder, adapter, facts)} * "
                f"{_render_term(term.right, binder, adapter, facts)})")
    if isinstance(term, A.Div):
        return (f"({_render_term(term.num, binder, adapter, facts)} / "
                f"{_render_term(term.den, binder, adapter, facts)})")
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
    left = _render_term(rule.atom.left, rule.binder, adapter, facts)
    right = _render_term(rule.atom.right, rule.binder, adapter, facts)
    return f"{head} {left} {rule.atom.op} {right}"


# ---------------------------------------------------------------------------
# Inverse of Rule.unparse: parse the compact surface form back into an AST.
# ---------------------------------------------------------------------------

_OPS = (" <|> ", " ~= ", " == ", " != ", " <= ", " >= ")
_AGG_RE = re.compile(r"^(SUM|MIN|MAX|AVG)\((\w+)\)$")
_SCALE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\*(.+)$")
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _split_top(s: str, sep: str) -> List[str]:
    """Split ``s`` on ``sep`` only at parenthesis depth 0."""
    out, depth, last = [], 0, 0
    i, n, m = 0, len(s), len(sep)
    while i <= n - m:
        c = s[i]
        if c == "(":
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
    m = _AGG_RE.match(s)
    if m:
        return A.Agg(m.group(1), m.group(2))
    m = _SCALE_RE.match(s)
    if m:
        return A.Scale(float(m.group(1)), _parse_primary(m.group(2)))
    if _NUM_RE.match(s):
        return A.Const(float(s))
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
    for op in _OPS:
        parts = _split_top(body, op)
        if len(parts) == 2:
            left = _parse_term(parts[0])
            right = _parse_term(parts[1])
            return A.Rule(binder, A.Compare(left, op.strip(), right))
    raise ValueError(f"no comparison operator found in: {body!r}")
