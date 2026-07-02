"""LLM-style schema induction from column names."""

from __future__ import annotations

import pytest

from autogram.discovery import synth
from autogram.discovery.induce import (
    SubagentSchemaInducer,
    available_inducer_backends,
    induce_adapter,
    induce_spec,
    make_inducer,
)
from autogram.loader.names import NameModel


def test_exposes_only_two_llm_schema_backends():
    assert available_inducer_backends() == ("subagent", "openai")


def test_subagent_backend_without_transport_raises_hard_error():
    d = synth.make_synthetic(n_entities=3, n_snapshots=2, noise=0.0, seed=0)
    with pytest.raises(RuntimeError, match="Subagent schema induction requires"):
        induce_spec(d.columns, SubagentSchemaInducer(responder=None))


def test_subagent_backend_induces_synthetic_structure():
    d = synth.make_synthetic(n_entities=5, n_snapshots=4, noise=0.0, seed=0)
    spec = induce_spec(d.columns, make_inducer("subagent"))
    assert "node" in spec.ontology.binders
    assert "link" in spec.ontology.binders
    assert set(spec.ontology.ref_roles["node"]) >= {"measurement_source", "measurement_destination", "demand_self"}
    assert set(spec.ontology.fam_roles["node"]) >= {"demand_row", "demand_col", "fam_to", "fam_from"}


def test_induced_adapter_parses_columns():
    d = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=0)
    adapter = induce_adapter(d.columns, make_inducer("subagent"))
    nm = NameModel.from_columns_with_adapter(d.columns, adapter)
    assert len(nm.by_name) == len(d.columns)
    assert nm.nodes == frozenset(d.entities)
    assert nm.low_cols and nm.high_cols


def test_induction_is_rename_robust_in_structure():
    v2 = synth.Vocab(measurement="signal", demand="route", source="out", destination="inn",
                     to="unto", frm="fro", entity_prefix="z")
    d1 = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=0)
    d2 = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=0, vocab=v2)
    s1 = induce_spec(d1.columns, make_inducer("subagent"))
    s2 = induce_spec(d2.columns, make_inducer("subagent"))
    assert set(s1.ontology.binders) == set(s2.ontology.binders)
    assert {b: len(r) for b, r in s1.ontology.ref_roles.items()} == {b: len(r) for b, r in s2.ontology.ref_roles.items()}
    assert {b: len(r) for b, r in s1.ontology.fam_roles.items()} == {b: len(r) for b, r in s2.ontology.fam_roles.items()}


def test_openai_backend_uses_injected_responder(data):
    calls = {"n": 0}

    def responder(_prompt):
        calls["n"] += 1
        return make_inducer("subagent").to_json_spec(data.columns)

    adapter = induce_adapter(data.columns, make_inducer("openai", responder=responder))
    assert calls["n"] == 1
    assert "node" in adapter.binders
