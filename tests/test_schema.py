"""SchemaSpec interface + trusted compiler/adapter (kept functionality), via induced specs."""

from __future__ import annotations

import pytest

from autogram.discovery import synth
from autogram.discovery.induce import induce_spec
from autogram.schema import CompileError, compile_spec
from autogram.schema.spec import CellCodec, ColumnPattern, RoleOntology, SchemaSpec


def test_induced_spec_compiles(adapter):
    # the session adapter is a compiled induced spec; it exposes the induced ontology
    assert "link" in adapter.binders and "node" in adapter.binders
    assert adapter.noisy_kind == "meas" and adapter.demand_kind == "demand"


def test_compiler_round_trip():
    d = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=1)
    spec = induce_spec(d.columns)
    adapter = compile_spec(spec)
    # every ref template role is declared in the ontology for its binder
    for (binder, role) in adapter.ref_templates:
        assert role in adapter.ref_roles.get(binder, ())


def test_compiler_rejects_bad_strategy():
    spec = induce_spec(synth.make_synthetic(n_entities=4, n_snapshots=2, seed=0).columns)
    bad = SchemaSpec(
        name="bad", patterns=spec.patterns, ontology=spec.ontology,
        ref_templates=spec.ref_templates, family_selectors=spec.family_selectors,
        binder_enumerate={**spec.binder_enumerate, "cell": "no_such_strategy"},
        cell_codec=spec.cell_codec, noisy_kind=spec.noisy_kind, demand_kind=spec.demand_kind)
    with pytest.raises(CompileError):
        compile_spec(bad)


def test_compiler_rejects_bad_codec():
    spec = induce_spec(synth.make_synthetic(n_entities=4, n_snapshots=2, seed=0).columns)
    bad = SchemaSpec(
        name="bad", patterns=spec.patterns, ontology=spec.ontology,
        ref_templates=spec.ref_templates, family_selectors=spec.family_selectors,
        binder_enumerate=spec.binder_enumerate, cell_codec=CellCodec(kind="not_a_codec"))
    with pytest.raises(CompileError):
        compile_spec(bad)


def test_compiler_rejects_bad_regex():
    onto = RoleOntology(binders=("cell",), ref_roles={"cell": ("self",)}, fam_roles={"cell": ()})
    bad = SchemaSpec(
        name="bad",
        patterns=(ColumnPattern(name="x", matcher="regex", kind="meas",
                                direction="o", regex=r"(?P<n>.+"),),
        ontology=onto, ref_templates=(), family_selectors=(),
        binder_enumerate={"cell": "per_measured_col"}, cell_codec=CellCodec(kind="scalar"))
    with pytest.raises(CompileError):
        compile_spec(bad)
