"""Regression tests for real-subagent GrammarSpec completeness checks."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.induce import (
    OpenAISchemaInducer,
    SchemaCompletenessError,
    SubagentSchemaInducer,
    _spec_from_json,
    _predicates,
    _validate_schema_completeness,
    induce_spec,
)
from autogram.discovery.loop import discover
from autogram.discovery.validate import score_recovery
from autogram.dsl import ast as A
from autogram.loader.loader import build_dataset
from autogram.loader.names import NameModel
from autogram.schema.compiler import compile_spec
from autogram.schema.spec import FamilySelector, RefTemplate


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


def test_predicate_parser_accepts_model_field_value_aliases():
    assert _predicates([
        {"field": "source", "op": "==", "value": "@X"},
        {"lhs": "destination", "op": "!=", "rhs": "X"},
    ]) == (
        ("source", "==", "X"),
        ("destination", "!=", "X"),
    )


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


def test_empty_optional_kind_and_list_condition_fields_use_safe_defaults():
    data = synth.make_synthetic(n_entities=3, n_snapshots=8, noise=0.0, seed=0)
    payload = _broken_peer_payload(peer_group="destination")
    payload["noisy_kind"] = ""
    payload["condition_columns"] = []
    payload["boolean_roles"] = []
    payload["max_condition_values"] = 0
    payload["max_conjunction_terms"] = 0
    inducer = SubagentSchemaInducer(
        responder=lambda _prompt: json.dumps(payload),
        max_attempts=1,
    )

    spec = induce_spec(data.columns, inducer)

    assert spec.noisy_kind == "measurement"
    assert spec.condition_columns == {}
    assert spec.max_condition_values == 4
    assert spec.max_conjunction_terms == 3


def test_too_small_conjunction_bound_is_rejected():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="destination")
    payload["max_conjunction_terms"] = 1

    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        induce_spec(
            data.columns,
            SubagentSchemaInducer(
                responder=lambda _prompt: json.dumps(payload),
                max_attempts=1,
            ),
        )


def test_subagent_preserves_declared_proportional_operator():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="destination")
    payload["ontology"]["ops"].append("~\u221d")

    spec = induce_spec(
        data.columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: json.dumps(payload),
            max_attempts=1,
        ),
    )

    assert "~\u221d" in spec.ontology.ops


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


def test_subagent_repairs_directed_pair_mislabeled_as_single_node():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="")
    for pattern in payload["patterns"]:
        if pattern["direction"] in {"to", "from"}:
            pattern["node_groups"] = ["source"]
            pattern["destination_group"] = ""
            pattern["peer_group"] = ""
            pattern["token_groups"] = ["source"]
            pattern["split_slots"] = ["source"]
    inducer = SubagentSchemaInducer(
        responder=lambda _prompt: json.dumps(payload),
        max_attempts=1,
    )

    spec = induce_spec(data.columns, inducer)
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(data.columns, adapter)

    assert adapter.enumerate_bindings("link", nm)
    assert {
        pattern.direction: pattern.node_groups
        for pattern in spec.patterns
        if pattern.direction in {"from", "to"}
    } == {
        "from": ("source", "peer"),
        "to": ("source", "peer"),
    }


def test_subagent_repairs_structural_patterns_shadowed_by_catch_all():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="")
    demand_pattern = next(
        pattern
        for pattern in payload["patterns"]
        if pattern["direction"] == "demand"
    )
    payload["patterns"] = [{
        "name": "cell_any",
        "matcher": "regex",
        "kind": "",
        "direction": "",
        "regex": r"^.+$",
        "node_groups": [],
        "source_group": "",
        "destination_group": "",
        "peer_group": "",
        "token_groups": [],
        "prefix": "",
        "sep": "_",
        "split_slots": ["source", "destination"],
    }, demand_pattern]
    payload["noisy_kind"] = "wrong"
    payload["link_marker_direction"] = "missing"
    inducer = SubagentSchemaInducer(
        responder=lambda _prompt: json.dumps(payload),
        max_attempts=1,
    )

    spec = induce_spec(data.columns, inducer)
    adapter = compile_spec(spec)
    nm = NameModel.from_columns_with_adapter(data.columns, adapter)

    assert len([
        semantics
        for semantics in nm.by_name.values()
        if semantics.kind == "flow" and semantics.direction == "demand"
    ]) == 9
    assert adapter.enumerate_bindings("link", nm)


def test_canonical_family_selectors_override_malformed_model_variants():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="destination")
    payload["family_selectors"] = [
        {
            "binder": "node",
            "family_role": "demand_row",
            "match_kind": "flow",
            "match_direction": "demand",
            "predicates": [
                ["destination", "==", "X"],
                ["source", "!=", "X"],
            ],
        },
        {
            "binder": "node",
            "family_role": "demand_col",
            "match_kind": "flow",
            "match_direction": "demand",
            "predicates": [
                ["source", "==", "X"],
                ["destination", "!=", "X"],
            ],
        },
    ]
    payload["ontology"]["ref_roles"]["network"] = ["arbitrary"]
    payload["ref_templates"].append({
        "binder": "network",
        "role": "arbitrary",
        "template": "flow_n0_n0",
    })

    spec = induce_spec(
        data.columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: json.dumps(payload),
            max_attempts=1,
        ),
    )
    selectors = {
        selector.family_role: selector
        for selector in spec.family_selectors
    }

    assert selectors["demand_row"].predicates == (
        ("source", "==", "X"),
        ("destination", "!=", "X"),
    )
    assert selectors["demand_col"].predicates == (
        ("destination", "==", "X"),
        ("source", "!=", "X"),
    )
    assert spec.ontology.ref_roles["network"] == ()


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


def test_string_capability_boolean_is_rejected():
    data = synth.make_synthetic(
            n_entities=3,
            n_snapshots=8,
            noise=0.0,
            seed=0,
    )
    payload = _broken_peer_payload(peer_group="destination")
    payload["temporal_enabled"] = "false"

    with pytest.raises(RuntimeError, match="invalid or incomplete"):
            induce_spec(
                data.columns,
                SubagentSchemaInducer(
                    responder=lambda _prompt: json.dumps(payload),
                    max_attempts=1,
                ),
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


def test_schema_completeness_rejects_dead_declared_roles():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    base = _spec_from_json(_broken_peer_payload(peer_group="destination"))
    ref_roles = dict(base.ontology.ref_roles)
    fam_roles = dict(base.ontology.fam_roles)
    ref_roles["network"] = ("ghost_ref",)
    fam_roles["network"] = (
        *fam_roles.get("network", ()),
        "ghost_family",
    )
    broken = replace(
        base,
        ontology=replace(
            base.ontology,
            ref_roles=ref_roles,
            fam_roles=fam_roles,
        ),
        ref_templates=(
            *base.ref_templates,
            RefTemplate("network", "ghost_ref", "missing_column"),
        ),
        family_selectors=(
            *base.family_selectors,
            FamilySelector(
                "network",
                "ghost_family",
                "missing_kind",
            ),
        ),
    )

    with pytest.raises(SchemaCompletenessError, match="ghost_ref|ghost_family"):
        _validate_schema_completeness(broken, data.columns)


def test_subagent_prunes_dead_declared_roles_before_use():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="destination")
    payload["ontology"]["ref_roles"]["network"] = ["ghost_ref"]
    payload["ref_templates"].append({
        "binder": "network",
        "role": "ghost_ref",
        "template": "missing_column",
    })

    spec = induce_spec(
        data.columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: json.dumps(payload),
            max_attempts=1,
        ),
    )

    assert "ghost_ref" not in spec.ontology.ref_roles["network"]
    _validate_schema_completeness(spec, data.columns)


def test_subagent_prunes_boolean_roles_not_declared_as_refs():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    payload = _broken_peer_payload(peer_group="destination")
    payload["boolean_roles"] = {
        "network": ["all_measurement_flag"],
    }

    spec = induce_spec(
        data.columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: json.dumps(payload),
            max_attempts=1,
        ),
    )

    assert spec.boolean_roles.get("network", ()) == ()
    _validate_schema_completeness(spec, data.columns)


def test_subagent_prunes_declared_binder_with_no_live_bindings():
    columns = [
        "measurement_a_source",
        "measurement_a_destination",
        "flow_a_a",
    ]
    payload = _broken_peer_payload(peer_group="destination")

    spec = induce_spec(
        columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: json.dumps(payload),
            max_attempts=1,
        ),
    )

    assert "link" not in spec.ontology.binders
    assert "link" not in spec.ontology.ref_roles
    assert "link" not in spec.ontology.fam_roles
    assert "link" not in spec.binder_enumerate
    assert "link" not in spec.boolean_roles
    assert all(template.binder != "link" for template in spec.ref_templates)
    assert all(selector.binder != "link" for selector in spec.family_selectors)
    assert all(template.binder != "link" for template in spec.related_templates)
    _validate_schema_completeness(spec, columns)


def test_openai_applies_the_same_deterministic_schema_cleanup():
    columns = [
        "measurement_a_source",
        "measurement_a_destination",
        "flow_a_a",
    ]
    payload = json.dumps(_broken_peer_payload(peer_group="destination"))
    expected = induce_spec(
        columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: payload,
            max_attempts=1,
        ),
    )

    actual = induce_spec(
        columns,
        OpenAISchemaInducer(responder=lambda _prompt: payload),
    )

    assert actual == expected


def test_subagent_prunes_dead_canonicalized_single_node_role():
    data = synth.make_synthetic(
        n_entities=3,
        n_snapshots=8,
        noise=0.0,
        seed=0,
    )
    columns = [
        *data.columns,
        *(
            f"measurement_input_increment_{entity}"
            for entity in data.entities
        ),
    ]
    payload = _broken_peer_payload(peer_group="destination")
    payload["patterns"].append({
        "name": "measurement_input_increment",
        "matcher": "regex",
        "kind": "measurement",
        "direction": "input_increment",
        "regex": r"^measurement_input_increment_(?P<source>[^_]+)$",
        "node_groups": ["source"],
        "source_group": "source",
        "destination_group": "",
        "peer_group": "",
        "token_groups": ["source"],
        "prefix": "",
        "sep": "_",
        "split_slots": ["source", "destination"],
    })

    spec = induce_spec(
        columns,
        SubagentSchemaInducer(
            responder=lambda _prompt: json.dumps(payload),
            max_attempts=1,
        ),
    )

    assert "measurement_input_increment" not in spec.ontology.ref_roles["node"]
    _validate_schema_completeness(spec, columns)


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
