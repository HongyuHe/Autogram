"""Trusted compiler: :class:`SchemaSpec` (data) -> :class:`SchemaAdapter` (callable).

This is the *only* component that turns a declarative spec into something the engine runs, so
it is the trust boundary.  It is deliberately small and total:

* it accepts only the bounded vocabulary declared in :mod:`autogram.schema.spec`
  (``ENUMERATE_STRATEGIES``, ``PRED_SLOTS``, ``PRED_OPS``, ``CELL_CODECS``);
* it compiles regexes with :func:`re.compile` (a regex is a *pattern*, never executable code)
  and raises :class:`CompileError` on a bad pattern or an undeclared group reference;
* it performs **no** ``eval``/``exec`` and imports nothing the spec names.

A spec that passes :func:`compile_spec` cannot widen the engine's trusted surface; it can only
re-parametrise the four seams the adapter interprets.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from .adapter import SchemaAdapter, _Pattern
from .spec import (
    CELL_CODECS,
    ENUMERATE_STRATEGIES,
    PRED_OPS,
    PRED_SLOTS,
    SchemaSpec,
)


class CompileError(ValueError):
    """Raised when a :class:`SchemaSpec` is structurally invalid or unsafe to compile."""


def _compile_pattern(p) -> _Pattern:
    if p.matcher not in ("regex", "split"):
        raise CompileError(f"pattern {p.name!r}: unknown matcher {p.matcher!r}")
    rx = None
    if p.matcher == "regex":
        if not p.regex:
            raise CompileError(f"pattern {p.name!r}: regex matcher needs a non-empty regex")
        try:
            rx = re.compile(p.regex)
        except re.error as e:                       # malformed pattern -> hard error
            raise CompileError(f"pattern {p.name!r}: bad regex: {e}") from e
        declared = set(rx.groupindex)
        referenced = set(p.node_groups) | set(p.token_groups)
        for g in (p.src_group, p.dst_group, p.peer_group):
            if g:
                referenced.add(g)
        missing = referenced - declared
        if missing:
            raise CompileError(
                f"pattern {p.name!r}: references undefined regex groups {sorted(missing)}")
    else:
        if not p.prefix:
            raise CompileError(f"pattern {p.name!r}: split matcher needs a prefix")
        if len(p.split_slots) != 2:
            raise CompileError(f"pattern {p.name!r}: split_slots must name exactly two slots")
        for s in p.split_slots:
            if s not in PRED_SLOTS:
                raise CompileError(f"pattern {p.name!r}: bad split slot {s!r}")
    return _Pattern(
        matcher=p.matcher, kind=p.kind, direction=p.direction, rx=rx,
        node_groups=tuple(p.node_groups), src_group=p.src_group, dst_group=p.dst_group,
        peer_group=p.peer_group, token_groups=tuple(p.token_groups),
        prefix=p.prefix, sep=p.sep, split_slots=tuple(p.split_slots),
    )


def _validate_selectors(spec: SchemaSpec) -> None:
    for sel in spec.family_selectors:
        for pred in sel.predicates:
            if len(pred) != 3:
                raise CompileError(
                    f"family {sel.family_role!r}: predicate {pred!r} must be [slot, op, rhs]")
            slot, op, _ = pred
            if slot not in PRED_SLOTS:
                raise CompileError(f"family {sel.family_role!r}: bad slot {slot!r}")
            if op not in PRED_OPS:
                raise CompileError(f"family {sel.family_role!r}: bad op {op!r}")


def compile_spec(spec: SchemaSpec) -> SchemaAdapter:
    """Validate and compile ``spec`` into a runnable :class:`SchemaAdapter`."""
    if not spec.patterns:
        raise CompileError("spec has no column patterns")

    onto = spec.ontology
    for b, strat in spec.binder_enumerate.items():
        if b not in onto.binders:
            raise CompileError(f"enumerate strategy declared for unknown binder {b!r}")
        if strat not in ENUMERATE_STRATEGIES:
            raise CompileError(f"binder {b!r}: unknown enumerate strategy {strat!r}")
    for b in onto.binders:
        if b not in spec.binder_enumerate:
            raise CompileError(f"binder {b!r} has no enumerate strategy")

    if spec.cell_codec.kind not in CELL_CODECS:
        raise CompileError(f"unknown cell codec {spec.cell_codec.kind!r}")

    _validate_selectors(spec)

    patterns = tuple(_compile_pattern(p) for p in spec.patterns)

    ref_templates: Dict[Tuple[str, str], str] = {}
    for t in spec.ref_templates:
        if t.binder not in onto.binders:
            raise CompileError(f"ref template for unknown binder {t.binder!r}")
        if t.role not in onto.ref_roles.get(t.binder, ()):
            raise CompileError(
                f"ref template role {t.role!r} not in ontology for binder {t.binder!r}")
        ref_templates[(t.binder, t.role)] = t.template

    selectors: Dict[Tuple[str, str], object] = {}
    for sel in spec.family_selectors:
        if sel.binder not in onto.binders:
            raise CompileError(f"family selector for unknown binder {sel.binder!r}")
        if sel.family_role not in onto.fam_roles.get(sel.binder, ()):
            raise CompileError(
                f"family role {sel.family_role!r} not in ontology for binder {sel.binder!r}")
        selectors[(sel.binder, sel.family_role)] = sel

    return SchemaAdapter(
        name=spec.name,
        patterns=patterns,
        ref_roles=dict(onto.ref_roles),
        fam_roles=dict(onto.fam_roles),
        binders=tuple(onto.binders),
        ops=tuple(onto.ops),
        agg_kinds=tuple(onto.agg_kinds),
        ref_templates=ref_templates,
        family_selectors=selectors,
        binder_enumerate=dict(spec.binder_enumerate),
        codec_kind=spec.cell_codec.kind,
        codec_primary=spec.cell_codec.primary,
        codec_clean=spec.cell_codec.clean,
        noisy_kind=spec.noisy_kind,
        demand_kind=spec.demand_kind,
        link_marker_dir=spec.link_marker_dir,
        ref_glyphs=dict(onto.ref_glyphs),
        fam_glyphs=dict(onto.fam_glyphs),
    )
