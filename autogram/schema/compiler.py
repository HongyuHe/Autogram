"""Trusted compiler: :class:`GrammarSpec` (data) -> :class:`SchemaAdapter` (callable).

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

import math
import numbers
import re
import string
from dataclasses import replace
from typing import Dict, Tuple

from ..dsl import ast as A
from .adapter import SchemaAdapter, _Pattern
from .spec import (
    CELL_CODECS,
    ENUMERATE_STRATEGIES,
    PRED_OPS,
    PRED_SLOTS,
    GrammarSpec,
)


class CompileError(ValueError):
    """Raised when a :class:`GrammarSpec` is structurally invalid or unsafe to compile."""


# Trusted hard ceilings on every unbounded numeric knob a proposer/model can declare. The compiler
# is the trust boundary: an untrusted spec may only *name* a bounded grammar, never demand one so
# large that enumeration materialises billions of terms (e.g. a huge ``max_lag``) before the
# ``max_rules`` ceiling can fail. These maxima are generous relative to any real dataset yet keep
# a hostile or buggy spec from exhausting memory. They are policy of the trusted base, not the spec.
_MAX_DEGREE = 8
_MAX_LAG = 1024
_MAX_WINDOW = 1_000_000
_MAX_WINDOW_COUNT = 64
_MAX_RUN_LENGTH = 1024
_MAX_RUN_LENGTH_COUNT = 64
_MAX_CONJUNCTION_TERMS = 16
_MAX_CONDITION_VALUES = 1024
_MAX_CONDITION_DOMAIN = 64
_MAX_REGEX_LENGTH = 4096


_BINDING_FIELDS = {
    "per_measured_col": frozenset({"col"}),
    "per_node": frozenset({"X"}),
    "per_directed_link": frozenset({"X", "Y"}),
    "singleton": frozenset(),
}


def _validate_bounded_int(label: str, value, *, minimum: int, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        raise CompileError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )


def _validate_unique(label: str, values) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise CompileError(f"duplicate {label} {value!r}")
        seen.add(value)


def _validate_typed_unique(label: str, values) -> None:
    """Finite JSON-scalar uniqueness, where ``True`` and ``1`` are distinct."""
    from ..dsl.evaluate import canonical_typed_value, typed_group_key

    seen = set()
    for value in values:
        value = canonical_typed_value(value)
        if not (
            value is None
            or isinstance(value, (str, bool))
            or (
                isinstance(value, numbers.Integral)
                and not isinstance(value, bool)
            )
            or (
                isinstance(value, numbers.Real)
                and not isinstance(value, numbers.Integral)
                and math.isfinite(float(value))
            )
        ):
            raise TypeError(f"{label} must be a finite JSON scalar")
        key = typed_group_key(value)
        if key in seen:
            raise CompileError(f"duplicate {label} {value!r}")
        seen.add(key)


def _canonical_typed_values(values) -> tuple:
    from ..dsl.evaluate import typed_unique

    return typed_unique(values)


def _validate_names(label: str, values) -> None:
    values = tuple(values)
    for value in values:
        if not isinstance(value, str) or not value:
            raise CompileError(f"{label} names must be non-empty strings")
    _validate_unique(label, values)


_ROLE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _validate_role_names(label: str, values) -> None:
    """Role names must be plain identifiers.

    Roles are rendered verbatim by ``unparse()``, which is the canonical structural key used for
    enumeration dedup, Z3 leaf identity, and archive/parameter lookup. Allowing a role to contain
    operator characters, spaces, or parentheses (e.g. ``"(a / b)"``) would let a compound term and a
    single role render identically and collide, so the trusted boundary restricts roles to
    identifiers -- exactly what name-driven induction and profiling already produce.
    """
    _validate_names(label, values)
    for value in values:
        # ``fullmatch`` (not ``match``) so the whole string must be an identifier: an anchored
        # ``$`` pattern would still accept a trailing newline (``"x\n"``), which would corrupt the
        # ``unparse()`` structural key that role identity depends on.
        if not _ROLE_NAME_RE.fullmatch(value):
            raise CompileError(
                f"{label} {value!r} must be a plain identifier (letters, digits, underscore)"
            )
        if value.lower() in {"nan", "inf", "infinity"}:
            raise CompileError(
                f"{label} {value!r} is a reserved numeric token"
            )


def _validate_positive_int(label: str, value, *, minimum: int = 1) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise CompileError(f"{label} must be an integer >= {minimum}")


def _validate_template_fields(template, strategy: str) -> None:
    if not isinstance(template.template, str) or not template.template:
        raise CompileError(
            f"ref template {template.role!r}: template must be a non-empty string"
        )
    allowed = _BINDING_FIELDS[strategy]
    try:
        parts = tuple(string.Formatter().parse(template.template))
    except ValueError as error:
        raise CompileError(
            f"ref template {template.role!r}: malformed format string"
        ) from error
    for _literal, field, format_spec, conversion in parts:
        if field is None:
            continue
        if (
            field not in allowed
            or bool(format_spec)
            or conversion is not None
        ):
            raise CompileError(
                f"ref template {template.role!r}: unsafe format field "
                f"{field!r}; allowed fields are {sorted(allowed)}"
            )


def _compile_pattern(p) -> _Pattern:
    if p.matcher not in ("regex", "split"):
        raise CompileError(f"pattern {p.name!r}: unknown matcher {p.matcher!r}")
    rx = None
    if p.matcher == "regex":
        if not p.regex:
            raise CompileError(f"pattern {p.name!r}: regex matcher needs a non-empty regex")
        if len(p.regex) > _MAX_REGEX_LENGTH:
            raise CompileError(
                f"pattern {p.name!r}: regex exceeds the trusted length ceiling "
                f"({len(p.regex)} > {_MAX_REGEX_LENGTH})"
            )
        try:
            rx = re.compile(p.regex)
        except re.error as e:                       # malformed pattern -> hard error
            raise CompileError(f"pattern {p.name!r}: bad regex: {e}") from e
        declared = set(rx.groupindex)
        referenced = set(p.node_groups) | set(p.token_groups)
        for g in (p.source_group, p.destination_group, p.peer_group):
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
        node_groups=tuple(p.node_groups), source_group=p.source_group, destination_group=p.destination_group,
        peer_group=p.peer_group, token_groups=tuple(p.token_groups),
        prefix=p.prefix, sep=p.sep, split_slots=tuple(p.split_slots),
    )


def _validate_selectors(spec: GrammarSpec) -> None:
    for sel in spec.family_selectors:
        if any(not isinstance(column, str) or not column for column in sel.columns):
            raise CompileError(f"family {sel.family_role!r}: explicit columns must be non-empty strings")
        _validate_unique(
            f"column in family {sel.family_role!r}",
            sel.columns,
        )
        for pred in sel.predicates:
            if len(pred) != 3:
                raise CompileError(
                    f"family {sel.family_role!r}: predicate {pred!r} must be [slot, op, rhs]")
            slot, op, _ = pred
            if slot not in PRED_SLOTS:
                raise CompileError(f"family {sel.family_role!r}: bad slot {slot!r}")
            if op not in PRED_OPS:
                raise CompileError(f"family {sel.family_role!r}: bad op {op!r}")


def compile_spec(spec: GrammarSpec) -> SchemaAdapter:
    """Validate and compile ``spec`` into a runnable :class:`SchemaAdapter`."""
    if not spec.patterns:
        raise CompileError("spec has no column patterns")

    onto = spec.ontology
    _validate_role_names("binder", onto.binders)
    _validate_names("pattern", (pattern.name for pattern in spec.patterns))
    for mapping_name, mapping in (
        ("ref_roles", onto.ref_roles),
        ("fam_roles", onto.fam_roles),
    ):
        unknown = set(mapping) - set(onto.binders)
        if unknown:
            raise CompileError(
                f"{mapping_name} declared for unknown binders {sorted(unknown)}"
            )
        for binder, roles in mapping.items():
            _validate_role_names(f"{mapping_name}[{binder!r}] role", roles)
    _validate_unique("operator", onto.ops)
    _validate_unique("aggregation", onto.agg_kinds)
    unknown_ops = set(onto.ops) - set(A.OPS)
    if unknown_ops:
        raise CompileError(
            f"unknown comparison operator(s) {sorted(unknown_ops)}"
        )
    unknown_aggregations = set(onto.agg_kinds) - set(A.AGG_KINDS)
    if unknown_aggregations:
        raise CompileError(
            "unknown aggregation kind(s) "
            f"{sorted(unknown_aggregations)}"
        )
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
    _validate_bounded_int("max_degree", spec.max_degree, minimum=1, maximum=_MAX_DEGREE)
    _validate_bounded_int("max_lag", spec.max_lag, minimum=0, maximum=_MAX_LAG)
    if len(spec.windows) > _MAX_WINDOW_COUNT:
        raise CompileError(
            f"too many temporal windows ({len(spec.windows)} > {_MAX_WINDOW_COUNT})"
        )
    for window in spec.windows:
        _validate_bounded_int("temporal window", window, minimum=1, maximum=_MAX_WINDOW)
    _validate_unique("temporal window", spec.windows)
    _validate_bounded_int(
        "max_condition_values",
        spec.max_condition_values,
        minimum=1,
        maximum=_MAX_CONDITION_VALUES,
    )
    if len(spec.run_lengths) > _MAX_RUN_LENGTH_COUNT:
        raise CompileError(
            f"too many run lengths ({len(spec.run_lengths)} > {_MAX_RUN_LENGTH_COUNT})"
        )
    for window in spec.run_lengths:
        _validate_bounded_int("run length", window, minimum=1, maximum=_MAX_RUN_LENGTH)
    _validate_unique("run length", spec.run_lengths)
    _validate_bounded_int(
        "max_conjunction_terms",
        spec.max_conjunction_terms,
        minimum=2,
        maximum=_MAX_CONJUNCTION_TERMS,
    )
    unknown_boolean_binders = set(spec.boolean_roles) - set(onto.binders)
    if unknown_boolean_binders:
        raise CompileError(
            "Boolean roles declared for unknown binders "
            f"{sorted(unknown_boolean_binders)}"
        )
    for binder, roles in spec.boolean_roles.items():
        _validate_role_names(f"Boolean role for {binder!r}", roles)
        unknown = set(roles) - set(onto.ref_roles.get(binder, ()))
        if unknown:
            raise CompileError(
                f"Boolean roles {sorted(unknown)} are not ref roles for "
                f"binder {binder!r}"
            )
    _validate_role_names(
        "condition column",
        spec.condition_columns,
    )
    for name, values in spec.condition_columns.items():
        # An empty value tuple is legitimate: induction declares condition column *names*
        # from the column vocabulary, while their observed *values* are populated later from
        # data during profiling. Both the proposer and admissibility skip empty-domain
        # columns, so an unpopulated domain is harmless; only reject unhashable values.
        if len(values) > _MAX_CONDITION_DOMAIN:
            raise CompileError(
                f"condition column {name!r} declares {len(values)} values, exceeding the "
                f"trusted domain ceiling {_MAX_CONDITION_DOMAIN}"
            )
        try:
            _validate_typed_unique(
                f"condition value for {name!r}",
                values,
            )
        except TypeError as error:
            raise CompileError(
                f"condition column {name!r} contains an unhashable value"
            ) from error

    _validate_selectors(spec)
    related_templates = {}
    for template in spec.related_templates:
        if template.binder not in onto.binders:
            raise CompileError(f"related template for unknown binder {template.binder!r}")
        if template.mode not in ("sum_delta", "sum_last", "span_any"):
            raise CompileError(
                f"related template {template.role!r}: unknown mode {template.mode!r}"
            )
        if len(template.parent_keys) != len(template.child_keys):
            raise CompileError(
                f"related template {template.role!r}: parent_keys and child_keys must align"
            )
        _validate_positive_int(
            f"related template {template.role!r} window_seconds",
            template.window_seconds,
        )
        if int(template.window_seconds) > (2 ** 63 - 1) // 1_000_000_000:
            raise CompileError(
                f"related template {template.role!r}: window_seconds exceeds "
                "the datetime64[ns] range"
            )
        if not template.role or not template.relation or not template.column:
            raise CompileError(
                "related template role, relation, and column must be non-empty"
            )
        _validate_role_names(
            "related template role",
            (template.role,),
        )
        if not template.parent_time or not template.child_time:
            raise CompileError(
                f"related template {template.role!r}: time columns must be non-empty"
            )
        if template.mode == "span_any" and (
            not template.span_start
            or not template.span_end
        ):
            raise CompileError(
                f"related template {template.role!r}: span_any requires "
                "non-empty span bounds"
            )
        if template.mode == "span_any" and (
            bool(template.filter_values)
            and not template.filter_column
        ):
            raise CompileError(
                f"related template {template.role!r}: span filter values "
                "require a filter column"
            )
        if template.mode == "span_any" and template.filter_values:
            try:
                _validate_typed_unique(
                    f"span filter value for {template.role!r}",
                    template.filter_values,
                )
            except TypeError as error:
                raise CompileError(
                    f"related template {template.role!r} contains "
                    "an unhashable span filter value"
                ) from error
        key = (template.binder, template.role)
        if key in related_templates:
            raise CompileError(
                f"duplicate related template {key!r}"
            )
        related_templates[key] = replace(
            template,
            filter_values=_canonical_typed_values(
                template.filter_values
            ),
        )

    patterns = tuple(_compile_pattern(p) for p in spec.patterns)

    ref_templates: Dict[Tuple[str, str], str] = {}
    for t in spec.ref_templates:
        if t.binder not in onto.binders:
            raise CompileError(f"ref template for unknown binder {t.binder!r}")
        if t.role not in onto.ref_roles.get(t.binder, ()):
            raise CompileError(
                f"ref template role {t.role!r} not in ontology for binder {t.binder!r}")
        key = (t.binder, t.role)
        if key in ref_templates:
            raise CompileError(f"duplicate ref template {key!r}")
        _validate_template_fields(
            t,
            spec.binder_enumerate[t.binder],
        )
        ref_templates[key] = t.template

    selectors: Dict[Tuple[str, str], object] = {}
    for sel in spec.family_selectors:
        if sel.binder not in onto.binders:
            raise CompileError(f"family selector for unknown binder {sel.binder!r}")
        if sel.family_role not in onto.fam_roles.get(sel.binder, ()):
            raise CompileError(
                f"family role {sel.family_role!r} not in ontology for binder {sel.binder!r}")
        key = (sel.binder, sel.family_role)
        if key in selectors:
            raise CompileError(f"duplicate family selector {key!r}")
        selectors[key] = sel

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
        link_marker_direction=spec.link_marker_direction,
        max_degree=spec.max_degree,
        role_exclusions=tuple(spec.role_exclusions),
        ref_glyphs=dict(onto.ref_glyphs),
        fam_glyphs=dict(onto.fam_glyphs),
        time_index=spec.time_index,
        group_keys=tuple(spec.group_keys),
        condition_columns={
            key: _canonical_typed_values(values)
            for key, values in spec.condition_columns.items()
        },
        temporal_enabled=bool(spec.temporal_enabled),
        max_lag=int(spec.max_lag),
        windows=tuple(sorted({int(window) for window in spec.windows})),
        conditional_enabled=bool(spec.conditional_enabled),
        max_condition_values=int(spec.max_condition_values),
        related_templates=related_templates,
        boolean_roles={binder: tuple(roles) for binder, roles in spec.boolean_roles.items()},
        advanced_enabled=bool(spec.advanced_enabled),
        run_lengths=tuple(sorted({int(window) for window in spec.run_lengths})),
        max_conjunction_terms=int(spec.max_conjunction_terms),
        metadata_columns=tuple(spec.metadata_columns),
        band_enabled=bool(spec.band_enabled),
    )
