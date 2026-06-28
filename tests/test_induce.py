"""Heuristic schema induction from column names (P2): roles are created, not chosen."""

from __future__ import annotations

from autogram.discovery import synth
from autogram.discovery.induce import HeuristicInducer, induce_adapter, induce_spec
from autogram.loader.names import NameModel


def test_induces_entities_keywords_connectors():
    d = synth.make_synthetic(n_entities=5, n_snapshots=4, noise=0.0, seed=0)
    info = HeuristicInducer().analyse(d.columns)
    assert info.measured_kind == "meas"
    assert info.demand_kind == "flow"
    assert set(info.entities) == {f"n{i}" for i in range(5)}
    assert set(info.keywords) == {"src", "dst"}
    assert set(info.connectors) == {"to", "from"}


def test_induced_adapter_parses_columns():
    d = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=0)
    adapter = induce_adapter(d.columns)
    nm = NameModel.from_columns_with_adapter(d.columns, adapter)
    # every structured column parses to semantics
    assert len(nm.by_name) == len(d.columns)
    # entity universe is recovered
    assert nm.nodes == frozenset(d.entities)
    # both layers are populated
    assert nm.low_cols and nm.high_cols


def test_induction_is_rename_robust_in_structure():
    """A renamed vocabulary induces the same structural shape (counts of roles/binders)."""
    v2 = synth.Vocab(meas="signal", demand="route", src="out", dst="inn",
                     to="unto", frm="fro", entity_prefix="z")
    d1 = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=0)
    d2 = synth.make_synthetic(n_entities=4, n_snapshots=4, noise=0.0, seed=0, vocab=v2)
    s1 = induce_spec(d1.columns)
    s2 = induce_spec(d2.columns)
    assert set(s1.ontology.binders) == set(s2.ontology.binders)
    assert {b: len(r) for b, r in s1.ontology.ref_roles.items()} == \
           {b: len(r) for b, r in s2.ontology.ref_roles.items()}
    assert {b: len(r) for b, r in s1.ontology.fam_roles.items()} == \
           {b: len(r) for b, r in s2.ontology.fam_roles.items()}


def test_inducer_handles_demandless_schema():
    """No demand/pair kind -> entities still inferred from connector flanks (fallback path)."""
    cols = [f"meas_n{i}_to_n{j}" for i in range(4) for j in range(4) if i != j]
    cols += [f"meas_n{i}_src" for i in range(4)]
    info = HeuristicInducer().analyse(cols)
    assert info.demand_kind is None
    assert set(info.entities) == {f"n{i}" for i in range(4)}
    assert "to" in info.connectors
