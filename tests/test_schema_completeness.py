"""Regression tests for real-subagent SchemaSpec completeness checks."""

from __future__ import annotations

import json

import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.induce import (
    SchemaCompletenessError,
    SubagentSchemaInducer,
    _spec_from_json,
    _validate_schema_completeness,
    induce_spec,
)
from autogram.discovery.loop import discover
from autogram.discovery.validate import score_recovery
from autogram.dsl import ast as A
from autogram.loader.loader import build_dataset
from autogram.loader.names import NameModel
from autogram.schema.compiler import compile_spec


def _broken_peer_payload(peer_group: str = "") -> dict:
    def pattern(name, kind, direction, regex, nodes, source="", destination="", peer=""):
        return {
            "name": name,
            "matcher": "regex",
            "kind": kind,
            "direction": direction,
            "regex": regex,
            "node_groups": nodes,
            "source_group": source,
            "destination_group": destination,
            "peer_group": peer,
            "token_groups": nodes,
            "prefix": "",
            "sep": "_",
            "split_slots": ["source", "destination"],
        }

    return {
        "name": "broken-directed-peer",
        "patterns": [
            pattern("flow_demand", "flow", "demand", r"^flow_(?P<source>[^_]+)_(?P<destination>[^_]+)$", ["source", "destination"], "source", "destination"),
            pattern("measurement_source", "measurement", "source", r"^measurement_(?P<source>[^_]+)_source$", ["source"], "source"),
            pattern("measurement_destination", "measurement", "destination", r"^measurement_(?P<source>[^_]+)_destination$", ["source"], "source"),
            pattern("measurement_to", "measurement", "to", r"^measurement_(?P<source>[^_]+)_to_(?P<destination>[^_]+)$", ["source", "destination"], "source", "destination", peer_group),
            pattern("measurement_from", "measurement", "from", r"^measurement_(?P<source>[^_]+)_from_(?P<destination>[^_]+)$", ["source", "destination"], "source", "destination", peer_group),
        ],
        "ontology": {
            "binders": ["cell", "node", "network", "link"],
            "ref_roles": {"cell": ["self"], "node": [], "network": [], "link": ["o0", "o0_rev", "o1", "o1_rev", "demand", "demand_rev"]},
            "fam_roles": {"cell": [], "node": ["fam_from", "fam_to"], "network": ["all_demand", "all_measurement_source", "all_measurement_destination"], "link": []},
            "ops": ["~=", "==", "!=", "<=", ">=", "<|>"],
            "agg_kinds": ["SUM", "MIN", "MAX", "AVG"],
            "ref_glyphs": {},
            "fam_glyphs": {},
        },
        "ref_templates": [
            {"binder": "cell", "role": "self", "template": "{col}"},
            {"binder": "link", "role": "o0", "template": "measurement_{X}_from_{Y}"},
            {"binder": "link", "role": "o0_rev", "template": "measurement_{Y}_from_{X}"},
            {"binder": "link", "role": "o1", "template": "measurement_{X}_to_{Y}"},
            {"binder": "link", "role": "o1_rev", "template": "measurement_{Y}_to_{X}"},
            {"binder": "link", "role": "demand", "template": "flow_{X}_{Y}"},
            {"binder": "link", "role": "demand_rev", "template": "flow_{Y}_{X}"},
        ],
        "family_selectors": [],
        "binder_enumerate": {"cell": "per_measured_col", "node": "per_node", "network": "singleton", "link": "per_directed_link"},
        "cell_codec": {"kind": "scalar", "primary": "ground_truth", "clean": "hidden_ground_truth"},
        "noisy_kind": "measurement",
        "demand_kind": "flow",
        "link_marker_direction": "from",
        "notes": "second directed endpoint was labeled destination instead of peer",
    }


class _DottedVocab(synth.Vocab):
    def entity(self, i: int) -> str:
        return f"pop{i}.site-{i}"


def _broken_dotted_demand_payload(mode: str) -> dict:
    payload = _broken_peer_payload(peer_group="destination")
    demand = next(p for p in payload["patterns"] if p["name"] == "flow_demand")
    if mode == "zero":
        demand["regex"] = r"^flow_(?P<source>[^_.]+)_(?P<destination>[^_.]+)$"
        payload["notes"] = "demand pattern splits entity tokens at dots and grounds zero columns"
    elif mode == "truncated":
        demand["regex"] = r"^flow_(?P<source>[^.]+)\.[^_]+_(?P<destination>[^.]+)\.[^_]+$"
        payload["notes"] = "demand pattern matches dotted columns but truncates entity tokens"
    else:  # pragma: no cover - tests only pass known modes
        raise ValueError(mode)
    return payload


def test_subagent_repairs_second_directed_endpoint_to_peer_before_use():
    data = synth.make_synthetic(n_entities=3, n_snapshots=8, noise=0.0, seed=0)
    payload = json.dumps(_broken_peer_payload(peer_group=""))
    inducer = SubagentSchemaInducer(responder=lambda _prompt: payload)

    spec = induce_spec(data.columns, inducer)
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(data.columns, adapter)
    bindings = adapter.enumerate_bindings("link", nm)

    assert {p.direction: p.peer_group for p in spec.patterns if p.direction in {"from", "to"}} == {"from": "peer", "to": "peer"}
    assert bindings
    assert all(b["Y"] for b in bindings)


def test_demand_completeness_reports_zero_grounding_for_dotted_entities():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
        vocab=_DottedVocab(),
    )
    spec = _spec_from_json(_broken_dotted_demand_payload("zero"))

    with pytest.raises(SchemaCompletenessError, match="demand.*grounded 0"):
        _validate_schema_completeness(spec, data.columns)


def test_subagent_repairs_dotted_demand_pattern_before_use():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
        vocab=_DottedVocab(),
    )
    payload = json.dumps(_broken_dotted_demand_payload("zero"))
    inducer = SubagentSchemaInducer(responder=lambda _prompt: payload, max_attempts=1)

    spec = induce_spec(data.columns, inducer)
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(data.columns, adapter)
    demand = [
        sem for sem in nm.by_name.values()
        if sem.kind == adapter.demand_kind and sem.direction == "demand"
    ]

    assert len(demand) == 9
    assert {sem.source for sem in demand} == set(data.entities)
    assert {sem.destination for sem in demand} == set(data.entities)
    assert set(nm.node_list()) >= set(data.entities)


def test_subagent_repairs_dotted_directed_link_pattern_against_demand_entities():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
        vocab=_DottedVocab(),
    )
    payload = _broken_dotted_demand_payload("zero")
    for pattern in payload["patterns"]:
        if pattern["direction"] in {"to", "from"}:
            pattern["regex"] = (
                rf"^measurement_(?P<source>[^.]+)\.[^_]+_{pattern['direction']}_"
                r"(?P<destination>[^.]+)\.[^_]+$"
            )
    inducer = SubagentSchemaInducer(responder=lambda _prompt: json.dumps(payload), max_attempts=1)

    spec = induce_spec(data.columns, inducer)
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(data.columns, adapter)
    bindings = adapter.enumerate_bindings("link", nm)

    assert bindings
    assert {b["X"] for b in bindings} <= set(data.entities)
    assert {b["Y"] for b in bindings} <= set(data.entities)


def test_subagent_repairs_mislabeled_directed_pair_kind_for_link_binder():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
        vocab=_DottedVocab(),
    )
    payload = _broken_dotted_demand_payload("zero")
    for pattern in payload["patterns"]:
        if pattern["direction"] in {"to", "from"}:
            pattern["kind"] = "edge"
    inducer = SubagentSchemaInducer(responder=lambda _prompt: json.dumps(payload), max_attempts=1)

    spec = induce_spec(data.columns, inducer)
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(data.columns, adapter)

    assert adapter.enumerate_bindings("link", nm)
    assert {
        sem.direction for sem in nm.by_name.values()
        if sem.kind == adapter.noisy_kind and len(sem.nodes) >= 2
    } == {"from", "to"}


def test_dotted_demand_row_col_families_recover_across_repaired_inductions():
    for family, mode in (("row_sum", "zero"), ("col_sum", "truncated")):
        for seed in (0, 1, 2):
            data = synth.make_synthetic(
                n_entities=3,
                n_snapshots=80,
                noise=0.0,
                seed=seed,
                vocab=_DottedVocab(),
                families=(family,),
            )
            payload = json.dumps(_broken_dotted_demand_payload(mode))
            res = discover(
                data.columns,
                data.matrix,
                inducer=SubagentSchemaInducer(responder=lambda _prompt, p=payload: p, max_attempts=1),
                discovery_cfg=DiscoveryConfig(seed=seed),
                search_cfg=SearchConfig(seed=seed, max_complexity=8),
                name=f"dotted_demand_{family}_{seed}",
                timestamps=data.timestamps,
            )
            rec = score_recovery(res, data.planted)
            assert getattr(rec, family) >= 0.8, (
                family,
                seed,
                rec.as_dict(),
                [e.rule.unparse() for e in res.portfolio],
                res.diagnostics,
            )


def test_zero_grounding_declared_binder_reports_diagnostic():
    data = synth.make_synthetic(n_entities=3, n_snapshots=8, noise=0.0, seed=0)
    spec = _spec_from_json(_broken_peer_payload(peer_group="destination"))
    adapter = compile_spec(spec)
    adapter.ref_roles["link"] = tuple(adapter.ref_roles["link"]) + ("missing",)
    adapter.ref_templates[("link", "missing")] = "missing_{X}_{Y}"
    ds = build_dataset(data.columns, data.matrix, adapter, "diagnostic", data.timestamps)

    ev = DataOnlyEvaluator(ds, DiscoveryConfig()).evaluate(
        A.Rule("link", A.Compare(A.Ref("missing"), "~=", A.Ref("o0")))
    )

    assert not ev.accepted
    assert "grounded 0 points" in ev.reason
    assert "binder 'link'" in ev.reason


def test_directed_link_families_recover_across_repeated_repaired_inductions():
    directed_families = ("two_end", "offset_pair", "presence_pair")
    for family in directed_families:
        for seed in (0, 1, 2):
            data = synth.make_synthetic(
                n_entities=3,
                n_snapshots=80,
                noise=0.0,
                seed=seed,
                families=(family,),
            )
            payload = json.dumps(_broken_peer_payload(peer_group=""))
            res = discover(
                data.columns,
                data.matrix,
                inducer=SubagentSchemaInducer(responder=lambda _prompt, p=payload: p),
                discovery_cfg=DiscoveryConfig(seed=seed),
                search_cfg=SearchConfig(seed=seed, max_complexity=8),
                name=f"directed_{family}_{seed}",
                timestamps=data.timestamps,
            )
            rec = score_recovery(res, data.planted)
            assert getattr(rec, family) >= 0.8, (family, seed, rec.as_dict(), [e.rule.unparse() for e in res.portfolio])
