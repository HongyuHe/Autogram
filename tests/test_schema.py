"""GrammarSpec interface + trusted compiler/adapter (kept functionality), via induced specs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from autogram.discovery import synth
from autogram.discovery.induce import (
    _spec_from_json,
    _spec_to_json,
    induce_spec,
)
from autogram.schema import CompileError, compile_spec
from autogram.schema.spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    GrammarSpec,
    RefTemplate,
    RelatedTemplate,
    RoleOntology,
)


def test_induced_spec_compiles(adapter):
    # the session adapter is a compiled induced spec; it exposes the induced ontology
    assert "link" in adapter.binders and "node" in adapter.binders
    assert adapter.noisy_kind == "measurement" and adapter.demand_kind == "flow"


def test_compiler_rejects_unhashable_span_filter_values():
    spec = GrammarSpec(
        name="span-filter",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        related_templates=(
            RelatedTemplate(
                binder="record",
                role="event",
                relation="events",
                column="flag",
                mode="span_any",
                parent_keys=(),
                child_keys=(),
                partition_keys=(),
                parent_time="timestamp",
                child_time="timestamp",
                window_seconds=60,
                span_start="span_start",
                span_end="span_end",
                filter_column="kind",
                filter_values=([],),
            ),
        ),
        cell_codec=CellCodec(kind="scalar"),
    )

    with pytest.raises(CompileError, match="unhashable span filter"):
        compile_spec(spec)


def test_compiler_rejects_composite_condition_values_before_proposal():
    spec = GrammarSpec(
        name="composite-condition",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        condition_columns={"kind": ((1, 2),)},
        cell_codec=CellCodec(kind="scalar"),
    )

    with pytest.raises(CompileError, match="condition column"):
        compile_spec(spec)


def test_compiler_rejects_unrepresentable_related_window():
    base = _node_template_spec()
    invalid = GrammarSpec(
        **{
            **base.__dict__,
            "related_templates": (
                RelatedTemplate(
                    binder="node",
                    role="related",
                    relation="raw",
                    column="value",
                    mode="sum_last",
                    parent_keys=(),
                    child_keys=(),
                    partition_keys=(),
                    parent_time="timestamp",
                    child_time="timestamp",
                    window_seconds=10 ** 30,
                ),
            ),
        }
    )

    with pytest.raises(CompileError, match="datetime64\\[ns\\] range"):
        compile_spec(invalid)


def test_numpy_span_filter_values_compile_and_serialize_as_python_scalars():
    spec = GrammarSpec(
        name="numpy-span-filter",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        related_templates=(
            RelatedTemplate(
                binder="record",
                role="event",
                relation="events",
                column="flag",
                mode="span_any",
                parent_keys=(),
                child_keys=(),
                partition_keys=(),
                parent_time="timestamp",
                child_time="timestamp",
                window_seconds=60,
                span_start="span_start",
                span_end="span_end",
                filter_column="kind",
                filter_values=(np.int64(1),),
            ),
        ),
        cell_codec=CellCodec(kind="scalar"),
    )

    adapter = compile_spec(spec)
    compiled = adapter.related_templates[("record", "event")]
    assert compiled.filter_values == (1,)
    assert type(compiled.filter_values[0]) is int
    json.dumps(_spec_to_json(spec))


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_lag", 2.9),
        ("windows", [3.9]),
        ("run_lengths", [5.9]),
        ("max_conjunction_terms", 1),
    ],
)
def test_induced_schema_rejects_fractional_or_silently_widened_bounds(
    field,
    value,
):
    spec = GrammarSpec(
        name="bounds",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    payload = _spec_to_json(spec)
    payload[field] = value

    with pytest.raises(ValueError, match="must"):
        _spec_from_json(payload)


def test_induced_zero_condition_value_cap_uses_protocol_default():
    spec = GrammarSpec(
        name="condition-default",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    payload = _spec_to_json(spec)
    payload["max_condition_values"] = 0

    restored = _spec_from_json(payload)

    assert restored.max_condition_values == 4


def test_induced_zero_conjunction_cap_uses_protocol_default():
    spec = GrammarSpec(
        name="conjunction-default",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    payload = _spec_to_json(spec)
    payload["max_conjunction_terms"] = 0

    assert _spec_from_json(payload).max_conjunction_terms == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("windows", 0),
        ("run_lengths", False),
        ("max_condition_values", 0.0),
    ],
)
def test_induced_schema_rejects_falsy_wrong_typed_bounds(field, value):
    spec = GrammarSpec(
        name="wrong-type-bounds",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    payload = _spec_to_json(spec)
    payload[field] = value

    with pytest.raises(ValueError, match="must"):
        _spec_from_json(payload)


def test_infer_tokens_requires_full_column_match():
    # A regex token pattern must match the whole column name; a column that merely contains the
    # pattern as a prefix (with trailing junk) must not inject a spurious node token (I4).
    spec = _node_template_spec()
    adapter = compile_spec(spec)

    assert adapter.infer_tokens(["metric_alpha"]) == frozenset({"alpha"})
    # "metric_beta extra" only *starts* with the pattern; fullmatch rejects it
    assert adapter.infer_tokens(["metric_beta suffix\nrow"]) == frozenset()
    assert adapter.parse_column("metric_beta suffix\nrow", frozenset({"beta"})) is None
    assert adapter.parse_column("metric_alpha", frozenset({"alpha"})) is not None


def test_compiler_round_trip():
    d = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=1)
    spec = induce_spec(d.columns)
    adapter = compile_spec(spec)
    # every ref template role is declared in the ontology for its binder
    for (binder, role) in adapter.ref_templates:
        assert role in adapter.ref_roles.get(binder, ())


def test_compiler_rejects_bad_strategy():
    spec = induce_spec(synth.make_synthetic(n_entities=4, n_snapshots=2, seed=0).columns)
    bad = GrammarSpec(
        name="bad", patterns=spec.patterns, ontology=spec.ontology,
        ref_templates=spec.ref_templates, family_selectors=spec.family_selectors,
        binder_enumerate={**spec.binder_enumerate, "cell": "no_such_strategy"},
        cell_codec=spec.cell_codec, noisy_kind=spec.noisy_kind, demand_kind=spec.demand_kind)
    with pytest.raises(CompileError):
        compile_spec(bad)


def test_compiler_rejects_bad_codec():
    spec = induce_spec(synth.make_synthetic(n_entities=4, n_snapshots=2, seed=0).columns)
    bad = GrammarSpec(
        name="bad", patterns=spec.patterns, ontology=spec.ontology,
        ref_templates=spec.ref_templates, family_selectors=spec.family_selectors,
        binder_enumerate=spec.binder_enumerate, cell_codec=CellCodec(kind="not_a_codec"))
    with pytest.raises(CompileError):
        compile_spec(bad)


def test_compiler_rejects_bad_regex():
    onto = RoleOntology(binders=("cell",), ref_roles={"cell": ("self",)}, fam_roles={"cell": ()})
    bad = GrammarSpec(
        name="bad",
        patterns=(ColumnPattern(name="x", matcher="regex", kind="measurement",
                                direction="o", regex=r"(?P<n>.+"),),
        ontology=onto, ref_templates=(), family_selectors=(),
        binder_enumerate={"cell": "per_measured_col"}, cell_codec=CellCodec(kind="scalar"))
    with pytest.raises(CompileError):
        compile_spec(bad)


@pytest.mark.parametrize(
    "ops,aggregations",
    [
        (("BOGUS",), ("SUM",)),
        (("==",), ("BOGUS",)),
    ],
)
def test_compiler_rejects_unknown_dsl_vocabularies(
    ops,
    aggregations,
):
    spec = GrammarSpec(
        name="bad-vocabulary",
        patterns=(
            ColumnPattern(
                name="x",
                matcher="regex",
                kind="measurement",
                direction="value",
                regex=r"^x$",
            ),
        ),
        ontology=RoleOntology(
            binders=("cell",),
            ref_roles={"cell": ("self",)},
            fam_roles={"cell": ()},
            ops=ops,
            agg_kinds=aggregations,
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"cell": "per_measured_col"},
        cell_codec=CellCodec(kind="scalar"),
    )

    with pytest.raises(CompileError, match="unknown"):
        compile_spec(spec)


def _node_template_spec(*templates, max_degree=1):
    return GrammarSpec(
        name="template-boundary",
        patterns=(
            ColumnPattern(
                name="measurement",
                matcher="regex",
                kind="measurement",
                direction="value",
                regex=r"^metric_(?P<node>.+)$",
                node_groups=("node",),
                token_groups=("node",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("value",)},
            fam_roles={"node": ()},
        ),
        ref_templates=templates or (
            RefTemplate("node", "value", "metric_{X}"),
        ),
        family_selectors=(),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
        max_degree=max_degree,
    )


def test_compiler_rejects_unsafe_ref_template_fields():
    spec = _node_template_spec(
        RefTemplate("node", "value", "metric_{X.missing}"),
    )

    with pytest.raises(CompileError, match="format field"):
        compile_spec(spec)


def test_compiler_rejects_duplicate_declarations():
    duplicate_templates = _node_template_spec(
        RefTemplate("node", "value", "metric_{X}"),
        RefTemplate("node", "value", "other_{X}"),
    )
    duplicate_patterns = GrammarSpec(
        **{
            **_node_template_spec().__dict__,
            "patterns": (
                _node_template_spec().patterns[0],
                _node_template_spec().patterns[0],
            ),
        }
    )

    with pytest.raises(CompileError, match="duplicate ref template"):
        compile_spec(duplicate_templates)
    with pytest.raises(CompileError, match="duplicate pattern"):
        compile_spec(duplicate_patterns)


def test_compiler_rejects_duplicate_explicit_family_columns():
    base = _node_template_spec()
    duplicate_family = GrammarSpec(
        **{
            **base.__dict__,
            "ontology": RoleOntology(
                binders=("node",),
                ref_roles={"node": ("value",)},
                fam_roles={"node": ("family",)},
            ),
            "family_selectors": (
                FamilySelector(
                    "node",
                    "family",
                    "measurement",
                    columns=("metric_a", "metric_a"),
                ),
            ),
        }
    )

    with pytest.raises(CompileError, match="duplicate column"):
        compile_spec(duplicate_family)


@pytest.mark.parametrize("binder", ["a::b", "a b", "a\n"])
def test_compiler_requires_plain_binder_identifiers(binder):
    base = _node_template_spec()
    invalid = GrammarSpec(
        **{
            **base.__dict__,
            "ontology": RoleOntology(
                binders=(binder,),
                ref_roles={binder: ()},
                fam_roles={binder: ()},
            ),
            "ref_templates": (),
            "binder_enumerate": {binder: "singleton"},
        }
    )

    with pytest.raises(CompileError, match="plain identifier"):
        compile_spec(invalid)


@pytest.mark.parametrize(
    "column",
    ["a == 1, b", "b == 2, c", "a::b", "a\n"],
)
def test_compiler_requires_plain_condition_column_identifiers(column):
    base = _node_template_spec()
    invalid = GrammarSpec(
        **{
            **base.__dict__,
            "condition_columns": {column: (1, 2)},
        }
    )

    with pytest.raises(CompileError, match="plain identifier"):
        compile_spec(invalid)


def test_compiler_rejects_nonpositive_degree_bound():
    with pytest.raises(CompileError, match="max_degree"):
        compile_spec(_node_template_spec(max_degree=0))


def test_compiler_enforces_trusted_numeric_ceilings():
    from dataclasses import replace as _replace

    base = _node_template_spec()
    over_lag = _replace(base, temporal_enabled=True, max_lag=10_000_000)
    over_windows = _replace(base, temporal_enabled=True, windows=tuple(range(1, 200)))
    over_degree = _replace(base, max_degree=9)

    with pytest.raises(CompileError, match="max_lag"):
        compile_spec(over_lag)
    with pytest.raises(CompileError, match="too many temporal windows"):
        compile_spec(over_windows)
    with pytest.raises(CompileError, match="max_degree"):
        compile_spec(over_degree)


def test_compiler_matches_columns_by_full_string_not_prefix():
    # An unanchored/partial regex must not over-match: `metric_a` should classify column
    # `metric_a` but never `metric_a_extra`.
    adapter = compile_spec(_node_template_spec())
    from autogram.loader.names import NameModel

    nm = NameModel.from_columns_with_adapter(["metric_a", "metric_a_extra"], adapter)
    # Both parse (the pattern is `^metric_(?P<node>.+)$`), but a partial pattern must not leak.
    loose = GrammarSpec(
        **{
            **_node_template_spec().__dict__,
            "patterns": (
                ColumnPattern(
                    name="loose",
                    matcher="regex",
                    kind="measurement",
                    direction="value",
                    regex=r"metric_a",
                    node_groups=(),
                    token_groups=(),
                ),
            ),
        }
    )
    loose_adapter = compile_spec(loose)
    assert loose_adapter.parse_column("metric_a", ()) is not None
    assert loose_adapter.parse_column("metric_a_extra", ()) is None


def test_condition_conjunction_arity_is_bounded():
    from autogram.dsl.parser import condition_from_dict

    payload = {
        "op": "all",
        "values": [
            {"op": "==", "column": f"c{i}", "values": [i]}
            for i in range(100)
        ],
    }
    with pytest.raises(ValueError, match="between 2 and 16"):
        condition_from_dict(payload)


def test_condition_enumeration_fails_loud_before_combinatorial_blowup():
    from autogram.dsl.grammar import Grammar
    from autogram.discovery.propose import EnumerationProposer, SearchSpaceTruncatedError

    grammar = Grammar(
        binders=("record",),
        ops=("~=", "=="),
        ref_roles={"record": ("x", "y")},
        fam_roles={"record": ()},
        conditional_enabled=True,
        condition_columns={"code": tuple(range(60))},
        max_condition_values=60,
    )
    with pytest.raises(SearchSpaceTruncatedError, match="condition grammar"):
        EnumerationProposer(grammar)._conditions()


def test_compiler_rejects_non_identifier_role_names():
    from dataclasses import replace as _replace

    base = _node_template_spec()
    bad = _replace(
        base,
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("value", "(a / b)")},
            fam_roles={"node": ()},
        ),
    )
    with pytest.raises(CompileError, match="plain identifier"):
        compile_spec(bad)


def test_compiler_rejects_role_name_with_trailing_newline():
    from dataclasses import replace as _replace

    # A ``$``-anchored regex would accept ``"value\n"`` (``$`` matches before a trailing newline);
    # role identity is unparse()-based, so a trailing newline must be rejected at the boundary.
    base = _node_template_spec()
    bad = _replace(
        base,
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("value\n",)},
            fam_roles={"node": ()},
        ),
        ref_templates=(RefTemplate("node", "value\n", "metric_{X}"),),
    )
    with pytest.raises(CompileError, match="plain identifier"):
        compile_spec(bad)
