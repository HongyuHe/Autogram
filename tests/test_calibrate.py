"""Fast tests for the calibration wiring: capability widening, tiers, ladder, defaults."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autogram.config import DiscoveryConfig
from autogram.calibrate import (
    CalibrationConfig, _capability_tiers, _derive_regime, _knob_schedule, _merge_specs,
    _reachable_capability_tiers, _split_known,
    _make_calibration_inducer, _spec_summary, _widen_spec, calibrate,
)
from autogram.discovery.known import KnownInvariant, load_known
from autogram.discovery.known import _signature as _known_signature
from autogram.discovery.induce import SchemaInducer
from autogram.discovery.loop import build_dataframe_grammar
from autogram.discovery.regime import ProxyEntry, RegimeSpec
from autogram.loader.gtib import profile_dataframe
from autogram.loader.loader import build_dataset
from autogram.schema.compiler import compile_spec
from autogram.schema.spec import (
    CellCodec, ColumnPattern, FamilySelector, GrammarSpec, RefTemplate, RelatedTemplate,
    RoleOntology,
)


def _mini_spec(agg=("SUM",), max_degree=1, role_exclusions=()):
    onto = RoleOntology(binders=("node",), ref_roles={"node": ("a", "b")},
                        fam_roles={"node": ("fam",)}, agg_kinds=agg)
    return GrammarSpec(name="t", patterns=(), ontology=onto, ref_templates=(),
                       family_selectors=(), binder_enumerate={"node": "per_node"},
                       max_degree=max_degree, role_exclusions=role_exclusions)


def _sum_witness_spec(*, include_family: bool) -> GrammarSpec:
    columns = ("total", "a", "x", "y", "q")
    return GrammarSpec(
        name="sum-witness",
        patterns=(
            ColumnPattern(
                "values",
                "regex",
                "measurement",
                "value",
                regex=r"^(?:total|a|z|x|y|q)$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": columns},
            fam_roles={
                "record": ("parts",) if include_family else (),
            },
            agg_kinds=("SUM",),
        ),
        ref_templates=tuple(
            RefTemplate("record", column, column)
            for column in columns
        ),
        family_selectors=(
            (
                FamilySelector(
                    "record",
                    "parts",
                    "measurement",
                    columns=("a", "z"),
                ),
            )
            if include_family
            else ()
        ),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )


class _CalibrationInducer(SchemaInducer):
    def induce(self, columns, sample_rows=None) -> GrammarSpec:
        if any(str(column).startswith("flow_") for column in columns):
            from tests.test_gtib_proxies import _SyntheticInducer
            return _SyntheticInducer().induce(columns, sample_rows)
        return GrammarSpec(
            name="calibration_frame",
            patterns=(
                ColumnPattern(
                    "placeholder",
                    "regex",
                    "unused",
                    "unused",
                    regex=r"^does_not_match$",
                ),
            ),
            ontology=RoleOntology(
                binders=("network",),
                ref_roles={"network": ()},
                fam_roles={"network": ()},
            ),
            ref_templates=(),
            family_selectors=(),
            binder_enumerate={"network": "singleton"},
            cell_codec=CellCodec(kind="scalar"),
        )


def test_calibration_defaults_global_while_discovery_stays_adaptive():
    # The engine default is unchanged; calibration (config + CLI) defaults to one fixed global band.
    from autogram.cli import build_parser
    assert DiscoveryConfig().band_mode == "adaptive"
    assert CalibrationConfig().band_mode == "global"
    assert CalibrationConfig().max_capability_tiers == 5
    a = build_parser().parse_args(["calibrate", "--input", "x.pkl", "--known", "k.yaml"])
    assert a.band_mode == "global"


def test_known_split_is_disjoint_and_rejects_singleton_catalog():
    known = [
        KnownInvariant(f"i{index}", "==", f"x{index}", f"y{index}")
        for index in range(3)
    ]
    calibration, validation = _split_known(
        known,
        frac=0.99,
        seed=0,
    )

    assert calibration
    assert validation
    assert {item.name for item in calibration}.isdisjoint(
        item.name for item in validation
    )
    with pytest.raises(ValueError, match="at least two"):
        _split_known(known[:1], frac=0.3, seed=0)


def test_widen_spec_enables_all_aggs_and_degree():
    spec = _mini_spec(agg=("SUM",), max_degree=1)
    w = _widen_spec(spec, all_aggs=True, max_degree=2)
    assert set(w.ontology.agg_kinds) == {"SUM", "AVG", "MIN", "MAX"}
    assert w.max_degree == 2
    # original spec is untouched (frozen -> replace returns a copy)
    assert spec.ontology.agg_kinds == ("SUM",) and spec.max_degree == 1


def test_widen_spec_unions_without_duplicates_and_never_lowers_degree():
    spec = _mini_spec(agg=("AVG", "SUM"), max_degree=2)
    w = _widen_spec(spec, all_aggs=True, max_degree=1)   # asking for degree 1 must not lower 2
    assert w.max_degree == 2
    assert len(w.ontology.agg_kinds) == len(set(w.ontology.agg_kinds)) == 4


def test_widen_spec_can_drop_exclusions():
    spec = _mini_spec(role_exclusions=(frozenset({"a", "b"}),))
    assert _widen_spec(spec, drop_exclusions=True).role_exclusions == ()
    assert _widen_spec(spec).role_exclusions == (frozenset({"a", "b"}),)


def test_capability_tiers_are_monotone():
    tiers = _capability_tiers()
    assert tiers[0] == {}                                  # tier 0 = as induced
    assert tiers[1].get("all_aggs") is True                # tier 1 adds aggregations
    assert tiers[2].get("max_degree") == 2                 # tier 2 adds nonlinear degree


def test_knob_schedule_exercises_adaptive_first_then_global():
    base = DiscoveryConfig(seed=0, tolerance=0.05, hold_rate_threshold=0.62, band_mode="adaptive")
    ladder = _knob_schedule(base, null_floor=0.5)
    assert ladder[0].band_mode == "adaptive"               # adaptive is exercised first
    assert any(c.band_mode == "global" for c in ladder)    # global fallback is present


def test_knob_schedule_global_base_has_distinct_rungs():
    # With the default global base, the fallback rungs must genuinely loosen (wider tolerance),
    # not duplicate the base config -- otherwise calibration re-runs identical configs for nothing.
    base = DiscoveryConfig(seed=0, tolerance=0.05, hold_rate_threshold=0.62, band_mode="global")
    ladder = _knob_schedule(base, null_floor=0.5)
    keys = {(c.band_mode, round(c.tolerance, 4), round(c.hold_rate_threshold, 4)) for c in ladder}
    assert len(keys) == len(ladder)                        # no duplicate (mode, tol, thr) rungs
    assert all(c.band_mode == "global" for c in ladder)    # every rung stays in the default mode
    assert max(c.tolerance for c in ladder) > base.tolerance   # a genuinely looser rung exists


def test_knob_schedule_has_fixed_cost_at_null_floor():
    base = DiscoveryConfig(
        seed=0,
        tolerance=0.05,
        hold_rate_threshold=0.5,
        band_mode="global",
    )
    ladder = _knob_schedule(base, null_floor=0.5)
    keys = {
        (
            item.band_mode,
            round(item.tolerance, 8),
            round(item.hold_rate_threshold, 8),
        )
        for item in ladder
    }

    assert len(ladder) == 4
    assert len(keys) == 4
    assert _reachable_capability_tiers(5, 4) == [
        _capability_tiers()[0],
    ]
    assert _reachable_capability_tiers(5, 5) == (
        _capability_tiers()[:2]
    )


def test_spec_summary_reports_capabilities():
    s = _spec_summary(_mini_spec(agg=("SUM", "MIN"), max_degree=2), tier=1, caps={"all_aggs": True})
    assert s["tier"] == 1 and s["max_degree"] == 2
    assert s["agg_kinds"] == ["SUM", "MIN"] and s["n_ref_roles"] == 2


def test_calibration_config_saves_rules_by_default():
    c = CalibrationConfig()
    assert c.save_rules is True and c.rules_dir == "rules"


def test_calibrate_api_rejects_unbounded_automatic_advanced_tier(tmp_path):
    known = _write_known(tmp_path, [
        {"name": "x_nonnegative", "op": ">=", "lhs": "x", "rhs": 0},
        {"name": "y_nonnegative", "op": ">=", "lhs": "y", "rhs": 0},
    ])
    frame = profile_dataframe(pd.DataFrame({
        "x": [1.0, 2.0],
        "y": [2.0, 3.0],
    }))

    with pytest.raises(ValueError, match="finite max_rules"):
        calibrate(
            frame,
            known,
            CalibrationConfig(
                max_capability_tiers=5,
                max_rules=0,
                save_rules=False,
            ),
        )


def test_calibration_honors_configured_schema_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "autogram.calibrate.make_inducer",
        lambda backend, **kwargs: calls.append((backend, kwargs)) or object(),
    )

    _make_calibration_inducer(CalibrationConfig(backend="openai", harness="copilot"))
    _make_calibration_inducer(CalibrationConfig(backend="subagent", harness="codex"))

    assert calls == [
        ("openai", {}),
        ("subagent", {"harness": "codex"}),
    ]


def test_write_rules_dl_persists_learned_portfolio(tmp_path):
    from types import SimpleNamespace
    from autogram.discovery.export import write_rules_dl
    ev = SimpleNamespace(rule=SimpleNamespace(unparse=lambda: "[forall node] measurement ~= SUM(fam)"),
                         strictness="loose", eps=0.05, hold_rate=0.66, hold_rate_lo=0.65,
                         hold_rate_hi=0.67, support=0.9, mdl_gain=3.5)
    res = SimpleNamespace(portfolio=[ev], rounds_run=1, diagnostics=[])
    path = write_rules_dl(res, "mydata", out_dir=str(tmp_path))
    assert path is not None
    text = open(path, encoding="utf-8").read()
    assert "measurement ~= SUM(fam)" in text and "hold=0.660" in text
    # the same fields feed calibrate's JSON `learned_invariants`
    learned = {"rule": ev.rule.unparse(), "hold_rate": round(ev.hold_rate, 4), "eps": ev.eps}
    assert learned["rule"].startswith("[forall node]") and learned["hold_rate"] == 0.66


def test_calibrate_parser_exposes_save_flags():
    from autogram.cli import build_parser
    a = build_parser().parse_args(["calibrate", "--input", "x.pkl", "--known", "k.yaml"])
    assert a.rules_dir == "rules" and a.no_save_rules is False and a.name == ""


def test_merge_specs_grows_search_space_and_keeps_base_authoritative():
    # base grammar: node binder with roles a,b, a SUM family, an exclusion, and one grounding.
    base = replace(
        _mini_spec(agg=("SUM",), role_exclusions=(frozenset({"a", "b"}),)),
        patterns=(ColumnPattern(name="p_base", matcher="split", kind="low", direction="egress"),),
        ref_templates=(RefTemplate("node", "a", "low_{X}_a"),),
        family_selectors=(FamilySelector("node", "fam", "low"),),
    )
    # an independent re-proposal: a NEW binder+role+template, a new pattern, a new aggregation,
    # and a *conflicting* template for the existing (node, a) role that must NOT win.
    new = GrammarSpec(
        name="fresh",
        patterns=(ColumnPattern(name="p_new", matcher="split", kind="low", direction="ingress"),),
        ontology=RoleOntology(binders=("link",), ref_roles={"link": ("o0",), "node": ("a", "c")},
                              fam_roles={"link": ()}, agg_kinds=("AVG",)),
        ref_templates=(RefTemplate("node", "a", "SHOULD_NOT_WIN"),
                       RefTemplate("node", "c", "low_{X}_c"),
                       RefTemplate("link", "o0", "low_{X}_egress_{Y}")),
        family_selectors=(),
        binder_enumerate={"link": "per_directed_link"},
        max_degree=2,
        role_exclusions=(),
    )
    m = _merge_specs(base, new)

    # binders / roles / ops / aggs are unioned -> a superset of base
    assert set(m.ontology.binders) == {"node", "link"}
    assert set(m.ontology.ref_roles["node"]) == {"a", "b", "c"}
    assert m.ontology.ref_roles["link"] == ("o0",)
    assert set(m.ontology.agg_kinds) == {"SUM", "AVG"}
    # base stays authoritative on the shared (node, a) grounding
    node_a = [t.template for t in m.ref_templates if (t.binder, t.role) == ("node", "a")]
    assert node_a == ["low_{X}_a"]
    # novel groundings from the fresh proposal are added
    assert any((t.binder, t.role) == ("node", "c") for t in m.ref_templates)
    assert any((t.binder, t.role) == ("link", "o0") for t in m.ref_templates)
    assert {p.name for p in m.patterns} == {"p_base", "p_new"}
    assert m.binder_enumerate == {"node": "per_node", "link": "per_directed_link"}
    # base exclusions + dataset constants preserved; degree floor raised, never lowered
    assert m.role_exclusions == (frozenset({"a", "b"}),)
    assert m.max_degree == 2


def test_merge_specs_renames_nonidentical_pattern_name_collisions():
    from autogram.loader.names import NameModel
    from autogram.schema.compiler import compile_spec

    base = GrammarSpec(
        name="base",
        patterns=(
            ColumnPattern(
                name="metric",
                matcher="regex",
                kind="tabular",
                direction="a",
                regex=r"^a$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("a_role",)},
            fam_roles={"record": ()},
        ),
        ref_templates=(RefTemplate("record", "a_role", "a"),),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    new = GrammarSpec(
        name="new",
        patterns=(
            ColumnPattern(
                name="metric",
                matcher="regex",
                kind="tabular",
                direction="b",
                regex=r"^b$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("b_role",)},
            fam_roles={"record": ()},
        ),
        ref_templates=(RefTemplate("record", "b_role", "b"),),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )

    merged = _merge_specs(base, new, columns=("a", "b"))
    adapter = compile_spec(merged)
    model = NameModel.from_columns_with_adapter(("a", "b"), adapter)

    assert len({pattern.name for pattern in merged.patterns}) == 2
    assert adapter.resolve_ref("a_role", "record", {}, model) == "a"
    assert adapter.resolve_ref("b_role", "record", {}, model) == "b"


def test_merge_specs_is_identity_preserving_superset_of_base():
    # merging base with an empty proposal must reproduce base's expressive vocabulary exactly.
    base = _mini_spec(agg=("SUM", "AVG"), max_degree=2, role_exclusions=(frozenset({"a", "b"}),))
    empty = GrammarSpec(name="e", patterns=(), ontology=RoleOntology(binders=(), ref_roles={},
                        fam_roles={}, agg_kinds=()), ref_templates=(), family_selectors=(),
                        binder_enumerate={}, max_degree=1, role_exclusions=())
    m = _merge_specs(base, empty)
    assert set(m.ontology.binders) == set(base.ontology.binders)
    assert set(m.ontology.ref_roles["node"]) == set(base.ontology.ref_roles["node"])
    assert set(m.ontology.agg_kinds) == set(base.ontology.agg_kinds)
    assert m.max_degree == 2                                  # never lowered by an emptier proposal
    assert m.role_exclusions == (frozenset({"a", "b"}),)


def test_merge_specs_widens_capabilities_but_keeps_dataset_interpretation():
    base = _mini_spec()
    related = RelatedTemplate(
        binder="node",
        role="raw",
        relation="child",
        column="counter",
        mode="sum_delta",
        parent_keys=(),
        child_keys=(),
        partition_keys=("shard",),
        parent_time="timestamp",
        child_time="timestamp",
        window_seconds=60,
    )
    new = replace(
        _mini_spec(),
        time_index="timestamp",
        group_keys=("tenant",),
        condition_columns={"label": ("normal", "alert")},
        temporal_enabled=True,
        max_lag=7,
        windows=(3, 7),
        conditional_enabled=True,
        max_condition_values=6,
        related_templates=(related,),
        boolean_roles={"node": ("a",)},
        advanced_enabled=True,
        run_lengths=(3, 5),
        max_conjunction_terms=4,
        metadata_columns=("tenant",),
        band_enabled=True,
    )

    merged = _merge_specs(base, new)

    assert merged.time_index == base.time_index
    assert merged.group_keys == base.group_keys
    assert merged.condition_columns == base.condition_columns
    assert merged.temporal_enabled
    assert merged.max_lag == 7
    assert merged.windows == (3, 7)
    assert merged.conditional_enabled
    assert merged.max_condition_values == 6
    assert merged.related_templates == (related,)
    assert merged.boolean_roles == {"node": ()}
    assert merged.advanced_enabled
    assert merged.run_lengths == (3, 5)
    assert merged.max_conjunction_terms == 4
    assert merged.metadata_columns == base.metadata_columns
    assert merged.band_enabled


# --- regime derivation + calibrate wiring ---------------------------------------------------

def test_derive_regime_from_calibration_shapes():
    calib = [KnownInvariant("i1", "~=", "x", "y"), KnownInvariant("i2", ">=", "z", 0)]
    regime = _derive_regime(CalibrationConfig(), calib)
    assert {e.shape for e in regime.active_entries()} == {"offset_pair", "nonneg"}


def test_derive_regime_respects_custom_regime_unchanged():
    custom = RegimeSpec(entries=[ProxyEntry("agg_ref_balance"), ProxyEntry("two_end", active=False)])
    out = _derive_regime(CalibrationConfig(regime=custom), [KnownInvariant("i1", "~=", "x", "y")])
    assert out is custom                                        # identity: not re-derived
    assert {e.shape for e in out.active_entries()} == {"agg_ref_balance"}


def test_derive_regime_raises_when_no_shape_maps():
    # a "!=" separation is not an abstractable known-invariant form -> no proxy shape derivable
    with pytest.raises(ValueError):
        _derive_regime(CalibrationConfig(), [KnownInvariant("i1", "!=", "x", "y")])


def _write_known(tmp_path, invs):
    p = tmp_path / "known.json"
    p.write_text(json.dumps({"invariants": invs}))
    return str(p)


def test_invalid_split_inputs_fail_before_external_induction(monkeypatch, tmp_path):
    import autogram.calibrate as calibration

    known = _write_known(tmp_path, [
        {"name": "only", "op": "==", "lhs": "x", "rhs": "y"},
    ])

    def should_not_construct_inducer(_cfg):
        raise AssertionError("external inducer was constructed before local validation")

    monkeypatch.setattr(
        calibration,
        "_make_calibration_inducer",
        should_not_construct_inducer,
    )

    with pytest.raises(ValueError, match="at least two"):
        calibrate(
            pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}),
            known,
            CalibrationConfig(max_capability_tiers=1, save_rules=False),
        )


def test_calibrate_runs_real_tuning_nulls_discovery_and_report(
    monkeypatch,
    tmp_path,
):
    import autogram.calibrate as calibration

    frame = profile_dataframe(pd.DataFrame({
        "x": np.linspace(1.0, 120.0, 120),
        "y": np.linspace(2.0, 240.0, 120),
    }))
    known = _write_known(tmp_path, [
        {"name": "x_nonnegative", "op": ">=", "lhs": "x", "rhs": 0},
        {"name": "y_nonnegative", "op": ">=", "lhs": "y", "rhs": 0},
    ])
    monkeypatch.setattr(
        calibration,
        "make_inducer",
        lambda *args, **kwargs: _CalibrationInducer(),
    )

    report = calibrate(
        frame,
        known,
        CalibrationConfig(
            max_capability_tiers=1,
            max_iterations=1,
            max_complexity=6,
            max_add_arity=2,
            max_rules=500,
            save_rules=False,
            tolerance=0.05,
            hold_rate_threshold=0.72,
        ),
        name="production_path",
    )

    assert report["recall_all"] == 1.0
    assert report["recall_validation"] == 1.0
    assert report["false_discovery"] == {
        "null_equalities_accepted": 0,
        "null_temporal_accepted": 0,
        "null_definitions_accepted": 0,
    }
    assert report["n_rules_learned"] > 0
    from autogram.dsl.parser import rule_from_dict

    assert all(
        rule_from_dict(item["rule_payload"]).unparse()
        == item["rule"]
        for item in report["learned_invariants"]
    )


def _fake_calibrate_env(monkeypatch, *, recall_fn, null_fn, tune=None):
    """Patch calibrate's discovery seams so the loop runs with no LLM / no enumeration."""
    import autogram.calibrate as C
    from autogram.discovery.validate import PreparedProxy, ProxySuite

    def fake_prepare(regime, seed=0, inducer=None, **kwargs):
        pos = [PreparedProxy(e.shape, object(), object(), {}) for e in regime.active_entries()]
        return ProxySuite(positives=pos, null=PreparedProxy("null", object(), object(), {}))

    def fake_runtime(ds, grammar, search_cfg, **kwargs):
        return ProxySuite(
            positives=[],
            null=PreparedProxy("runtime_null", ds, grammar, {}),
            candidate_counts={
                "all": 0,
                "equalities": 0,
                "temporal": 0,
                "definitions": 0,
            },
        )

    def default_tune(suite, seed=0, band_mode="global", null_floor=0.5, max_expansions=3, **kw):
        shapes = [p.shape for p in suite.positives]
        return {"tolerance": 0.05, "hold_rate_threshold": 0.66, "expansions": 0,
                "selected_null_equalities": 0, "proxy_shapes": shapes,
                "per_proxy": [{"shape": s, "recovery": 0.9, "accepted": 3, "compact": True,
                               "scaled_slack": []} for s in shapes]}

    monkeypatch.setattr(C, "make_inducer", lambda *a, **k: object())
    monkeypatch.setattr(C, "prepare_proxy_suite", fake_prepare)
    monkeypatch.setattr(C, "prepare_runtime_null_controls", fake_runtime)
    monkeypatch.setattr(C, "tune_joint", tune or default_tune)
    monkeypatch.setattr(C, "induce_spec", lambda cols, inducer, *a, **k: _mini_spec())
    monkeypatch.setattr(C, "build_dataframe_grammar",
                        lambda df, spec, search_cfg=None, name="": (
                            SimpleNamespace(
                                observed=SimpleNamespace(
                                    has=lambda _column: False,
                                ),
                            ),
                            object(),
                        ))
    monkeypatch.setattr(C, "run_prepared",
                        lambda ds, G, discovery_cfg=None, search_cfg=None, proposer=None:
                        SimpleNamespace(portfolio=[], _dcfg=discovery_cfg))
    monkeypatch.setattr(C, "recover_known", lambda res, invs: {
        "recall": recall_fn(res._dcfg), "recovered": 1, "total": len(invs), "invariants": []})
    monkeypatch.setattr(C, "null_equalities_at", lambda null, dcfg, seed=0: null_fn(dcfg))


def test_calibrate_reports_selected_proxy_shapes_and_evidence(monkeypatch, tmp_path):
    _fake_calibrate_env(monkeypatch, recall_fn=lambda d: 0.5, null_fn=lambda d: 0)
    known = _write_known(tmp_path, [{"name": f"i{i}", "op": "~=", "lhs": f"x{i}", "rhs": f"y{i}"}
                                    for i in range(4)])
    df = SimpleNamespace(columns=["x0", "y0", "x1", "y1"])
    report = calibrate(df, known, CalibrationConfig(max_capability_tiers=1, save_rules=False))
    assert report["proxies"]["shapes"] == ["offset_pair"]         # derived from the ~= invariants
    ev = report["proxies"]["per_proxy"]
    assert ev and ev[0]["shape"] == "offset_pair"
    assert set(ev[0]) >= {"recovery", "accepted", "compact"}      # per-proxy tuning evidence
    assert report["proxies"]["selected_null_equalities"] == 0
    assert "null_equalities_accepted" in report["false_discovery"]
    assert set(report["provenance"]) == {
        "engine_source_sha256",
        "input_sha256",
        "known_sha256",
        "calibration_config_sha256",
    }
    assert all(
        len(value) == 64
        for value in report["provenance"].values()
    )
    assert report["induction"] == {
        "backend": "subagent",
        "harness": "copilot",
    }
    grammar_spec = report["grammar_specs"][0]
    assert len(grammar_spec["normalized_spec_sha256"]) == 64
    assert grammar_spec["normalized_spec"]["ontology"]["binders"]


def test_calibration_hashes_the_post_profile_runtime_spec(
    monkeypatch,
    tmp_path,
):
    _fake_calibrate_env(
        monkeypatch,
        recall_fn=lambda _config: 1.0,
        null_fn=lambda _config: 0,
    )
    known = _write_known(tmp_path, [
        {"name": "x_nonnegative", "op": ">=", "lhs": "x", "rhs": 0},
        {"name": "y_nonnegative", "op": ">=", "lhs": "y", "rhs": 0},
    ])
    frame = profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=20,
                freq="1min",
            ),
            "tenant": ["a"] * 20,
            "x": np.arange(20, dtype=float),
            "y": np.arange(20, dtype=float),
        }),
        time_index="timestamp",
        group_keys=("tenant",),
        temporal_windows=(3,),
        max_lag=3,
    )

    report = calibrate(
        frame,
        known,
        CalibrationConfig(
            max_capability_tiers=1,
            max_iterations=1,
            save_rules=False,
        ),
    )
    spec = report["grammar_specs"][0]["normalized_spec"]

    assert spec["ontology"]["binders"] == ["record"]
    assert spec["time_index"] == "timestamp"
    assert spec["group_keys"] == ["tenant"]
    assert spec["windows"] == [3]
    assert spec["metadata_columns"] == []

    # The published SHA-256 must be reproducible from the published normalized spec alone: a fresh
    # JSON round-trip fingerprint of the reported spec must equal the reported digest. This guards
    # against tuple/list drift between the hashed object and the serialized object (I8).
    from autogram.calibrate import _json_fingerprint, _json_normalize

    published = report["grammar_specs"][0]
    assert _json_fingerprint(_json_normalize(spec)) == published["normalized_spec_sha256"]
    assert _json_fingerprint(spec) == published["normalized_spec_sha256"]


def test_calibrate_unsafe_relaxed_rung_cannot_win(monkeypatch, tmp_path):
    # recall rises with tolerance, but the loosest rungs let the null accept an equality: an unsafe
    # relaxed rung must never become the winner even though its recall is higher.
    _fake_calibrate_env(monkeypatch,
                        recall_fn=lambda d: round(0.5 + d.tolerance, 4),
                        null_fn=lambda d: 1 if d.tolerance >= 0.1 else 0)
    known = _write_known(tmp_path, [{"name": f"i{i}", "op": "~=", "lhs": f"x{i}", "rhs": f"y{i}"}
                                    for i in range(4)])
    df = SimpleNamespace(columns=["x0", "y0"])
    report = calibrate(df, known, CalibrationConfig(max_capability_tiers=1, save_rules=False))
    assert report["config"]["tolerance"] == 0.05                  # a null-safe rung won
    assert report["recall_all"] == 0.55
    assert report["false_discovery"]["null_equalities_accepted"] == 0
    unsafe = [h for h in report["trajectory"] if h["null_equalities"] >= 1]
    assert unsafe                                                 # the relaxed rungs were tried
    assert max(h["calibration_recall"] for h in unsafe) > report["recall_all"]


def test_calibrate_continues_capability_widening_when_recall_stalls(monkeypatch, tmp_path):
    _fake_calibrate_env(
        monkeypatch,
        recall_fn=lambda _config: 0.5,
        null_fn=lambda _config: 0,
    )
    known = _write_known(
        tmp_path,
        [
            {"name": f"i{i}", "op": "~=", "lhs": f"x{i}", "rhs": f"y{i}"}
            for i in range(4)
        ],
    )
    report = calibrate(
        SimpleNamespace(columns=["x0", "y0"]),
        known,
        CalibrationConfig(max_capability_tiers=3, save_rules=False),
    )

    assert report["grammar_reinductions"] == 2
    assert {entry["grammar_tier"] for entry in report["trajectory"]} == {0, 1, 2}


def test_calibrate_split_closes_over_later_tier_runtime_witnesses(
    monkeypatch,
    tmp_path,
):
    import autogram.calibrate as calibration
    from autogram.discovery.validate import PreparedProxy, ProxySuite

    frame = pd.DataFrame({
        "total": np.full(20, 1000.05),
        "a": np.full(20, 1000.0),
        "z": np.full(20, 0.05),
        "x": np.arange(20.0),
        "y": np.arange(20.0) + 1.0,
        "q": np.arange(20.0) + 2.0,
    })
    known = _write_known(tmp_path, [
        {
            "name": "exact_padded",
            "op": "==",
            "lhs": "total",
            "rhs": {"sum": ["a", "z"]},
        },
        {
            "name": "approx_plain",
            "op": "~=",
            "lhs": "total",
            "rhs": {"sum": ["a"]},
        },
        {"name": "other", "op": "==", "lhs": "x", "rhs": "y"},
        {"name": "third", "op": ">=", "lhs": "q", "rhs": 0},
    ])
    proposals = iter((
        _sum_witness_spec(include_family=False),
        _sum_witness_spec(include_family=True),
    ))
    induced = []

    def fake_induce(columns, inducer):
        spec = next(proposals)
        induced.append(spec)
        return spec

    def fake_prepare(regime, seed=0, inducer=None, **kwargs):
        positives = [
            PreparedProxy(entry.shape, object(), object(), {})
            for entry in regime.active_entries()
        ]
        return ProxySuite(
            positives=positives,
            null=PreparedProxy("null", object(), object(), {}),
        )

    def fake_runtime(dataset, grammar, search_cfg, **kwargs):
        return ProxySuite(
            positives=[],
            null=PreparedProxy("runtime_null", dataset, grammar, {}),
            candidate_counts={
                "all": 0,
                "equalities": 0,
                "temporal": 0,
                "definitions": 0,
            },
        )

    monkeypatch.setattr(calibration, "make_inducer", lambda *args, **kwargs: object())
    monkeypatch.setattr(calibration, "induce_spec", fake_induce)
    monkeypatch.setattr(calibration, "prepare_proxy_suite", fake_prepare)
    monkeypatch.setattr(calibration, "prepare_runtime_null_controls", fake_runtime)
    monkeypatch.setattr(calibration, "tune_joint", lambda *args, **kwargs: {
        "tolerance": 0.05,
        "hold_rate_threshold": 0.66,
        "expansions": 0,
        "selected_null_equalities": 0,
        "proxy_shapes": [
            proxy.shape for proxy in args[0].positives
        ],
        "per_proxy": [],
    })
    monkeypatch.setattr(
        calibration,
        "run_prepared",
        lambda dataset, grammar, **kwargs: SimpleNamespace(
            portfolio=[],
            dataset=dataset,
            grammar=grammar,
            _dcfg=kwargs["discovery_cfg"],
        ),
    )
    monkeypatch.setattr(calibration, "recover_known", lambda result, invariants: {
        "recall": 0.5,
        "recovered": 1,
        "total": len(invariants),
        "invariants": [],
    })
    monkeypatch.setattr(calibration, "null_equalities_at", lambda *args, **kwargs: 0)
    monkeypatch.setattr(calibration, "null_temporal_at", lambda *args, **kwargs: 0)
    monkeypatch.setattr(calibration, "null_definitions_at", lambda *args, **kwargs: 0)

    original_split = calibration._split_known

    def checked_split(*args, **kwargs):
        without_witnesses = dict(kwargs)
        without_witnesses.pop("recovery_witnesses", None)
        without_witnesses["recovery_rules"] = None
        bare_calibration, _bare_validation = original_split(
            *args,
            **without_witnesses,
        )
        bare_names = {
            item.name for item in bare_calibration
        }
        assert (
            ("exact_padded" in bare_names)
            != ("approx_plain" in bare_names)
        )

        calibration_known, validation_known = original_split(
            *args,
            **kwargs,
        )
        calibration_names = {
            item.name for item in calibration_known
        }
        validation_names = {
            item.name for item in validation_known
        }
        assert {"exact_padded", "approx_plain"} <= calibration_names or {
            "exact_padded",
            "approx_plain",
        } <= validation_names
        return calibration_known, validation_known

    monkeypatch.setattr(calibration, "_split_known", checked_split)

    calibrate(
        frame,
        known,
        CalibrationConfig(
            max_capability_tiers=2,
            max_complexity=6,
            max_rules=10_000,
            validation_frac=0.5,
            save_rules=False,
        ),
        name="later_witness",
    )

    assert len(induced) == 2


def test_calibrate_applies_max_iterations_globally_across_tiers(
    monkeypatch,
    tmp_path,
):
    _fake_calibrate_env(
        monkeypatch,
        recall_fn=lambda _config: 0.5,
        null_fn=lambda _config: 0,
    )
    known = _write_known(
        tmp_path,
        [
            {"name": f"i{i}", "op": "~=", "lhs": f"x{i}", "rhs": f"y{i}"}
            for i in range(4)
        ],
    )

    report = calibrate(
        SimpleNamespace(columns=["x0", "y0"]),
        known,
        CalibrationConfig(
            max_capability_tiers=3,
            max_iterations=1,
            save_rules=False,
        ),
    )

    assert report["iterations"] == 1
    assert report["grammar_reinductions"] == 0


def test_derive_regime_rejects_custom_regime_with_no_active_entries():
    # A caller-supplied regime with nothing active would prepare no positive proxies; reject it
    # clearly before any preparation, instead of silently "calibrating" on an empty suite.
    for empty in (RegimeSpec(entries=[]),
                  RegimeSpec(entries=[ProxyEntry("offset_pair", active=False)])):
        with pytest.raises(ValueError):
            _derive_regime(CalibrationConfig(regime=empty), [KnownInvariant("i1", "~=", "x", "y")])


def test_calibrate_scores_each_runtime_grammar_with_memoized_rungs(monkeypatch, tmp_path):
    # Each widened grammar gets its own runtime-parity null, while repeated access to one tier/rung
    # remains memoized and unsafe rungs still cannot win.
    import autogram.calibrate as C
    from autogram.discovery.validate import PreparedProxy, ProxySuite

    def fake_prepare(regime, seed=0, inducer=None, **kwargs):
        pos = [PreparedProxy(e.shape, object(), object(), {}) for e in regime.active_entries()]
        return ProxySuite(positives=pos, null=PreparedProxy("null", object(), object(), {}))

    def fake_runtime(ds, grammar, search_cfg, **kwargs):
        return ProxySuite(
            positives=[],
            null=PreparedProxy("runtime_null", ds, grammar, {}),
            candidate_counts={
                "all": 0,
                "equalities": 0,
                "temporal": 0,
                "definitions": 0,
            },
        )

    def fake_tune(suite, seed=0, band_mode="global", null_floor=0.5, max_expansions=3, **kw):
        shapes = [p.shape for p in suite.positives]
        return {"tolerance": 0.05, "hold_rate_threshold": 0.66, "expansions": 0,
                "selected_null_equalities": 0, "proxy_shapes": shapes,
                "per_proxy": [{"shape": s, "recovery": 0.9, "accepted": 3, "compact": True,
                               "scaled_slack": []} for s in shapes]}

    tier_counter = {"n": -1}

    def fake_build(df, spec, search_cfg=None, name=""):
        tier_counter["n"] += 1
        return (
            SimpleNamespace(
                tier=tier_counter["n"],
                observed=SimpleNamespace(
                    has=lambda _column: False,
                ),
            ),
            object(),
        )

    def fake_run(ds, G, discovery_cfg=None, search_cfg=None, proposer=None):
        return SimpleNamespace(portfolio=[], _dcfg=discovery_cfg, _tier=ds.tier)

    def fake_recover(res, invs):
        # recall rises with the grammar tier (so no stall-break) and slightly with tolerance,
        # staying < 1.0 so all three tiers run.
        recall = round(0.3 + 0.2 * res._tier + 0.1 * res._dcfg.tolerance, 4)
        return {"recall": recall, "recovered": 1, "total": len(invs), "invariants": []}

    null_calls = []

    def fake_null(prepared_null, dcfg, seed=0):
        null_calls.append((dcfg.band_mode, dcfg.tolerance, dcfg.hold_rate_threshold))
        return 1 if dcfg.tolerance >= 0.1 else 0

    monkeypatch.setattr(C, "make_inducer", lambda *a, **k: object())
    monkeypatch.setattr(C, "prepare_proxy_suite", fake_prepare)
    monkeypatch.setattr(C, "prepare_runtime_null_controls", fake_runtime)
    monkeypatch.setattr(C, "tune_joint", fake_tune)
    monkeypatch.setattr(C, "induce_spec", lambda cols, inducer, *a, **k: _mini_spec())
    monkeypatch.setattr(C, "build_dataframe_grammar", fake_build)
    monkeypatch.setattr(C, "run_prepared", fake_run)
    monkeypatch.setattr(C, "recover_known", fake_recover)
    monkeypatch.setattr(C, "null_equalities_at", fake_null)

    known = _write_known(tmp_path, [{"name": f"i{i}", "op": "~=", "lhs": f"x{i}", "rhs": f"y{i}"}
                                    for i in range(4)])
    report = calibrate(SimpleNamespace(columns=["x0", "y0"]), known,
                       CalibrationConfig(max_capability_tiers=3, save_rules=False))

    # (a) three tiers ran and each tier scored every distinct rung exactly once
    assert report["grammar_reinductions"] == 2                    # tiers 1 and 2 re-induced
    assert len(null_calls) == 12
    assert len(set(null_calls)) == 4
    # (b) an unsafe relaxed rung (higher recall) still cannot win
    assert report["config"]["tolerance"] == 0.05
    assert report["false_discovery"]["null_equalities_accepted"] == 0
    unsafe = [h for h in report["trajectory"] if h["null_equalities"] >= 1]
    assert unsafe and max(h["calibration_recall"] for h in unsafe) > report["recall_all"]


def test_known_split_keeps_alias_relations_on_the_same_side():
    # Round-24: the split must be by canonical relation signature, not list position. "x == y" and
    # "y == x" denote the SAME relation, so placing one in calibration and the other in validation
    # would mean tuning directly on a "held-out" invariant -- the calibration/validation gap would
    # stop being an overfitting alarm.
    known = [
        KnownInvariant("forward", "==", "x", "y"),
        KnownInvariant("reverse", "==", "y", "x"),
        KnownInvariant("other", "==", "a", "b"),
        KnownInvariant("third", "==", "c", "d"),
    ]

    for seed in range(25):
        calibration, validation = _split_known(known, frac=0.5, seed=seed)
        calib_names = {item.name for item in calibration}
        valid_names = {item.name for item in validation}
        assert calib_names.isdisjoint(valid_names)
        assert calib_names | valid_names == {"forward", "reverse", "other", "third"}
        # The aliases travel together.
        assert ("forward" in calib_names) == ("reverse" in calib_names)
        # And the two halves share no canonical signature at all.
        calib_sigs = {_known_signature(item) for item in calibration}
        valid_sigs = {_known_signature(item) for item in validation}
        assert calib_sigs.isdisjoint(valid_sigs), seed


def test_known_split_rejects_a_catalog_of_only_aliases():
    # If every entry canonicalises to one relation there is nothing to hold out, and calibration
    # must fail loudly rather than report a recall figure against a split it did not really make.
    aliases = [
        KnownInvariant("forward", "==", "x", "y"),
        KnownInvariant("reverse", "==", "y", "x"),
    ]
    with pytest.raises(ValueError, match="two distinct known-invariant relations"):
        _split_known(aliases, frac=0.5, seed=0)


def test_checked_in_gtib_split_is_signature_disjoint():
    known = load_known("configs/gtib_known.yaml")
    for seed in range(5):
        calibration, validation = _split_known(known, frac=0.3, seed=seed)
        assert len(calibration) + len(validation) == len(known)
        calib_sigs = {_known_signature(item) for item in calibration}
        valid_sigs = {_known_signature(item) for item in validation}
        assert calib_sigs.isdisjoint(valid_sigs), seed


def test_known_split_keeps_exact_and_approximate_forms_of_one_relation_together():
    # Round-26: `recover_known` expands an APPROXIMATE equality so the exact rule also recovers it
    # (`known._matching_signatures`). "x ~= y" and "x == y" are therefore recovered by one and the
    # same discovered rule, so splitting them apart would put a "held-out" invariant within reach of
    # the calibration half. Exact-signature grouping missed this because the two signatures differ
    # in their strength tag.
    known = [
        KnownInvariant("approx", "~=", "x", "y"),
        KnownInvariant("exact", "==", "x", "y"),
        KnownInvariant("other", "==", "a", "b"),
        KnownInvariant("third", "==", "c", "d"),
    ]

    for seed in range(25):
        calibration, validation = _split_known(known, frac=0.5, seed=seed)
        calib = {item.name for item in calibration}
        valid = {item.name for item in validation}
        assert calib.isdisjoint(valid)
        assert calib | valid == {"approx", "exact", "other", "third"}
        assert ("approx" in calib) == ("exact" in calib), seed


def test_known_split_merges_entries_that_data_canonicalization_makes_identical():
    """Round-27: the split must apply the same data-dependent canonicalisation as recovery.

    `recover_known` drops summed members whose observed data is negligible against the anchor's
    scale, so `total == SUM(a)` and `total == SUM(a, z)` with `z` identically zero are recovered by
    one and the same rule. Comparing signatures without that transform let the pair straddle the
    split, which would put a "held-out" invariant directly within reach of the calibration half.
    """
    from autogram.calibrate import _ColumnScaleView

    df = pd.DataFrame({
        "total": np.linspace(10.0, 20.0, 50),
        "a": np.linspace(10.0, 20.0, 50),
        "z": np.zeros(50),
        "p": np.linspace(1.0, 2.0, 50),
        "q": np.linspace(3.0, 4.0, 50),
    })
    known = [
        KnownInvariant("plain", "==", "total", {"sum": ["a"]}),
        KnownInvariant("padded", "==", "total", {"sum": ["a", "z"]}),
        KnownInvariant("other", "==", "p", "q"),
        KnownInvariant("third", "==", "q", "p"),
        KnownInvariant("fourth", ">=", "p", 0),
    ]
    view = _ColumnScaleView(df)

    # Without the canonicalising view the two spellings look like different relations.
    bare_calib, bare_valid = _split_known(known, frac=0.5, seed=0)
    assert {item.name for item in bare_calib} | {item.name for item in bare_valid}

    for seed in range(25):
        calibration, validation = _split_known(known, frac=0.4, seed=seed, frame=view)
        calib = {item.name for item in calibration}
        valid = {item.name for item in validation}
        assert calib.isdisjoint(valid)
        assert ("plain" in calib) == ("padded" in calib), seed


def test_known_split_keeps_atomic_and_lag_sign_aliases_together():
    """One exact atomic sign law recovers every same-direction grounded lag."""
    known = [
        KnownInvariant("atomic", ">=", "x", 0),
        KnownInvariant("lag_1", ">=", {"lag": ["x", 1]}, 0),
        KnownInvariant("lag_2", ">=", {"lag": ["x", 2]}, 0),
        KnownInvariant("other", "==", "a", "b"),
        KnownInvariant("third", ">=", "y", 0),
    ]
    frame = profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=20,
                freq="1min",
            ),
            "x": np.arange(1.0, 21.0),
            "a": np.arange(20.0),
            "b": np.arange(20.0),
            "y": np.arange(1.0, 21.0),
        }),
        time_index="timestamp",
        max_lag=3,
    )
    split_dataset, _grammar = build_dataframe_grammar(
        frame,
        _mini_spec(),
        name="lag_split",
    )

    for seed in range(25):
        calibration, validation = _split_known(
            known,
            frac=0.4,
            seed=seed,
            frame=split_dataset.observed,
            recovery_dataset=split_dataset,
        )
        calibration_names = {item.name for item in calibration}
        validation_names = {item.name for item in validation}
        assert calibration_names.isdisjoint(validation_names)
        locations = {
            name: name in calibration_names
            for name in ("atomic", "lag_1", "lag_2")
        }
        assert len(set(locations.values())) == 1, (seed, locations)


def test_known_split_only_merges_strict_lag_when_data_is_strictly_signed():
    known = [
        KnownInvariant("atomic", ">=", "x", 0),
        KnownInvariant("lag_strict", ">", {"lag": ["x", 2]}, 0),
        KnownInvariant("other", "==", "a", "b"),
        KnownInvariant("third", ">=", "y", 0),
    ]
    with_zero = pd.DataFrame({
        "x": [1.0, 1.0, 0.0],
        "a": [1.0, 2.0, 3.0],
        "b": [3.0, 2.0, 1.0],
        "y": [1.0, 2.0, 3.0],
    })
    positive = with_zero.copy()
    positive["x"] = [1.0, 2.0, 3.0]

    def split_dataset(frame):
        profiled = profile_dataframe(
            frame.assign(
                timestamp=pd.date_range(
                    "2026-01-01",
                    periods=len(frame),
                    freq="1min",
                ),
            ),
            time_index="timestamp",
            max_lag=2,
        )
        return build_dataframe_grammar(
            profiled,
            _mini_spec(),
            name="strict_lag_split",
        )[0]
    zero_dataset = split_dataset(with_zero)
    positive_dataset = split_dataset(positive)

    split_with_zero = _split_known(
        known,
        frac=0.5,
        seed=0,
        frame=zero_dataset.observed,
        recovery_dataset=zero_dataset,
    )
    calibration_zero = {item.name for item in split_with_zero[0]}
    assert (
        ("atomic" in calibration_zero)
        != ("lag_strict" in calibration_zero)
    )

    for seed in range(10):
        calibration, _validation = _split_known(
            known,
            frac=0.5,
            seed=seed,
            frame=positive_dataset.observed,
            recovery_dataset=positive_dataset,
        )
        names = {item.name for item in calibration}
        assert (
            ("atomic" in names)
            == ("lag_strict" in names)
        ), seed


def test_known_split_does_not_merge_nonstrict_lag_on_mixed_sign_data():
    known = [
        KnownInvariant("atomic", ">=", "x", 0),
        KnownInvariant("lag", ">=", {"lag": ["x", 1]}, 0),
        KnownInvariant("other", "==", "a", "b"),
        KnownInvariant("third", ">=", "y", 0),
    ]
    frame = profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="1min",
            ),
            "x": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            "a": np.arange(6.0),
            "b": np.arange(6.0)[::-1],
            "y": np.arange(1.0, 7.0),
        }),
        time_index="timestamp",
        max_lag=1,
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _mini_spec(),
        name="mixed_sign_lag_split",
    )

    calibration, _validation = _split_known(
        known,
        frac=0.5,
        seed=0,
        frame=dataset.observed,
        recovery_dataset=dataset,
    )
    names = {item.name for item in calibration}

    assert ("atomic" in names) != ("lag" in names)


def test_known_split_keeps_bindings_of_one_quantified_rule_together():
    frame = profile_dataframe(pd.DataFrame({
        "metric_n0_source": np.arange(1.0, 81.0),
        "metric_n1_source": np.arange(2.0, 82.0),
    }))
    spec = GrammarSpec(
        name="quantified-split",
        patterns=(
            ColumnPattern(
                name="measurement",
                matcher="regex",
                kind="measurement",
                direction="source",
                regex=r"^metric_(?P<source>n\d+)_source$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("measurement_source",)},
            fam_roles={"node": ()},
        ),
        ref_templates=(
            RefTemplate(
                "node",
                "measurement_source",
                "metric_{X}_source",
            ),
        ),
        family_selectors=(),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
        temporal_enabled=True,
        max_lag=1,
        time_index="timestamp",
    )
    dataset = build_dataset(
        frame.columns,
        frame.to_numpy(dtype=float),
        compile_spec(spec),
        name="quantified_split",
        timestamps=np.arange(len(frame)),
    )
    known = [
        KnownInvariant("n0", ">=", "metric_n0_source", 0),
        KnownInvariant(
            "n1",
            ">=",
            {"lag": ["metric_n1_source", 1]},
            0,
        ),
        KnownInvariant(
            "other",
            "==",
            "metric_n0_source",
            "metric_n1_source",
        ),
    ]
    calibration, validation = _split_known(
        known,
        frac=0.5,
        seed=0,
        frame=dataset.observed,
        recovery_dataset=dataset,
    )
    calibration_names = {item.name for item in calibration}
    validation_names = {item.name for item in validation}

    assert {"n0", "n1"} <= calibration_names or {
        "n0",
        "n1",
    } <= validation_names


def test_known_split_groups_later_tier_and_parameterized_bindings():
    frame = pd.DataFrame({
        "metric_n0_source": np.arange(1.0, 81.0),
        "metric_n1_source": np.arange(2.0, 82.0),
        "alert_n0": np.zeros(80),
        "alert_n1": np.zeros(80),
    })
    spec = GrammarSpec(
        name="quantified-advanced-split",
        patterns=(
            ColumnPattern(
                "measurement",
                "regex",
                "measurement",
                "source",
                regex=r"^metric_(?P<source>n\d+)_source$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
            ColumnPattern(
                "alert",
                "regex",
                "boolean",
                "alert",
                regex=r"^alert_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={
                "node": ("measurement_source", "alert"),
            },
            fam_roles={"node": ()},
        ),
        ref_templates=(
            RefTemplate(
                "node",
                "measurement_source",
                "metric_{X}_source",
            ),
            RefTemplate("node", "alert", "alert_{X}"),
        ),
        family_selectors=(),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
        temporal_enabled=False,
        time_index="timestamp",
    )
    dataset = build_dataset(
        frame.columns,
        frame.to_numpy(dtype=float),
        compile_spec(spec),
        name="quantified_advanced_split",
        timestamps=np.arange(len(frame)),
    )
    other = KnownInvariant(
        "other",
        "==",
        "metric_n0_source",
        "metric_n1_source",
    )
    catalogues = (
        [
            KnownInvariant(
                "n0",
                ">=",
                {"delta": ["metric_n0_source", 1]},
                0,
            ),
            KnownInvariant(
                "n1",
                ">=",
                {"delta": ["metric_n1_source", 1]},
                0,
            ),
            other,
        ],
        [
            KnownInvariant(
                "n0",
                ":=",
                "alert_n0",
                {
                    "sustained": {
                        "term": "metric_n0_source",
                        "op": "<",
                        "threshold": 40.0,
                        "window": 2,
                    },
                },
            ),
            KnownInvariant(
                "n1",
                ":=",
                "alert_n1",
                {
                    "sustained": {
                        "term": "metric_n1_source",
                        "op": "<",
                        "threshold": 40.0,
                        "window": 2,
                    },
                },
            ),
            other,
        ],
    )

    for catalogue_index, known in enumerate(catalogues):
        calibration, validation = _split_known(
            known,
            frac=0.5,
            seed=0,
            frame=dataset.observed,
            recovery_dataset=dataset,
        )
        calibration_names = {item.name for item in calibration}
        validation_names = {item.name for item in validation}
        assert {"n0", "n1"} <= calibration_names or {
            "n0",
            "n1",
        } <= validation_names, catalogue_index


def test_known_split_closes_over_fitted_definition_witness():
    from autogram.dsl import ast as A

    signal = np.linspace(90.0, 112.0, 221)
    frame = profile_dataframe(
        pd.DataFrame({
            "signal": signal,
            "other": signal * 2.0,
            "alert": signal < 101.0,
        }),
        condition_columns=("alert",),
        advanced=True,
    )
    dataset, grammar = build_dataframe_grammar(
        frame,
        _mini_spec(),
        name="fitted_split_witness",
    )
    witness = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("alert"),
            A.Conjunction((
                A.Bound(A.Ref("signal"), "<", None),
            )),
        ),
    )
    known = [
        KnownInvariant(
            "lower",
            ":=",
            "alert",
            {"and": [
                {"bound": ["signal", "<", 100.0]},
            ]},
        ),
        KnownInvariant(
            "upper",
            ":=",
            "alert",
            {"and": [
                {"bound": ["signal", "<", 101.9]},
            ]},
        ),
        KnownInvariant("other", "==", "signal", "other"),
    ]

    calibration, validation = _split_known(
        known,
        frac=0.5,
        seed=0,
        frame=dataset.observed,
        recovery_dataset=dataset,
        recovery_rules=[witness],
    )
    calibration_names = {item.name for item in calibration}
    validation_names = {item.name for item in validation}

    assert {"lower", "upper"} <= calibration_names or {
        "lower",
        "upper",
    } <= validation_names


def test_known_split_matches_alternative_roles_and_family_witnesses():
    frame = pd.DataFrame({
        "x_n0": np.ones(20),
        "y_n0": np.ones(20),
        "x_n1": np.ones(20),
        "y_n1": np.ones(20),
        "total_n0": np.full(20, 3.0),
        "total_n1": np.ones(20),
        "part_n0_a": np.ones(20),
        "part_n0_b": np.full(20, 2.0),
        "part_n1_a": np.ones(20),
    })
    spec = GrammarSpec(
        name="alternative-witnesses",
        patterns=(
            ColumnPattern(
                "x",
                "regex",
                "measurement",
                "x",
                regex=r"^x_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
            ColumnPattern(
                "y",
                "regex",
                "measurement",
                "y",
                regex=r"^y_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
            ColumnPattern(
                "total",
                "regex",
                "measurement",
                "total",
                regex=r"^total_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
            ColumnPattern(
                "part",
                "regex",
                "part",
                "part",
                regex=r"^part_(?P<source>n\d+)_.+$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={
                "node": ("x", "y", "x_alias", "total"),
            },
            fam_roles={"node": ("parts",)},
            agg_kinds=("SUM",),
        ),
        ref_templates=(
            RefTemplate("node", "x", "x_{X}"),
            RefTemplate("node", "y", "y_{X}"),
            RefTemplate("node", "x_alias", "x_n0"),
            RefTemplate("node", "total", "total_{X}"),
        ),
        family_selectors=(
            FamilySelector(
                "node",
                "parts",
                "part",
                predicates=(("source", "==", "X"),),
            ),
        ),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
    )
    dataset = build_dataset(
        frame.columns,
        frame.to_numpy(dtype=float),
        compile_spec(spec),
        name="alternative_witnesses",
        timestamps=np.arange(len(frame)),
    )
    catalogues = (
        [
            KnownInvariant("n0", "~=", "x_n0", "y_n0"),
            KnownInvariant("n1", "==", "x_n1", "y_n1"),
            KnownInvariant("other", ">=", "total_n0", 0),
        ],
        [
            KnownInvariant(
                "n0",
                "==",
                "total_n0",
                {"sum": ["part_n0_a", "part_n0_b"]},
            ),
            KnownInvariant(
                "n1",
                "==",
                "total_n1",
                {"sum": ["part_n1_a"]},
            ),
            KnownInvariant("other", "==", "x_n0", "y_n1"),
        ],
        [
            KnownInvariant("n0", "==", "x_n0", "y_n0"),
            KnownInvariant(
                "n1",
                "==",
                "x_n1",
                {"sum": ["y_n1"]},
            ),
            KnownInvariant("other", ">=", "total_n0", 0),
        ],
    )

    for catalogue_index, known in enumerate(catalogues):
        calibration, validation = _split_known(
            known,
            frac=0.5,
            seed=0,
            frame=dataset.observed,
            recovery_dataset=dataset,
        )
        calibration_names = {item.name for item in calibration}
        validation_names = {item.name for item in validation}
        assert {"n0", "n1"} <= calibration_names or {
            "n0",
            "n1",
        } <= validation_names, catalogue_index


def test_merge_specs_does_not_reclassify_existing_numeric_role():
    base = replace(
        _mini_spec(),
        patterns=(
            ColumnPattern(
                "metric",
                "regex",
                "measurement",
                "value",
                regex=r"^metric_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
        ),
        ref_templates=(
            RefTemplate("node", "a", "metric_{X}"),
        ),
    )
    later = replace(
        base,
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={
                "node": ("a", "b", "boolean_alias"),
            },
            fam_roles={"node": ("fam",)},
        ),
        ref_templates=(
            *base.ref_templates,
            RefTemplate(
                "node",
                "boolean_alias",
                "metric_n0",
            ),
        ),
        boolean_roles={
            "node": ("a", "boolean_alias"),
        },
    )

    merged = _merge_specs(
        base,
        later,
        columns=("metric_n0", "metric_n1"),
    )

    assert "a" not in merged.boolean_roles.get("node", ())
    assert "boolean_alias" not in merged.boolean_roles.get(
        "node",
        (),
    )


def test_merge_specs_keeps_base_measurements_out_of_later_metadata():
    base = GrammarSpec(
        name="measurement-base",
        patterns=(
            ColumnPattern(
                "metric",
                "regex",
                "measurement",
                "value",
                regex=r"^metric_(?P<source>n\d+)$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("metric",)},
            fam_roles={"node": ()},
            agg_kinds=("SUM",),
        ),
        ref_templates=(
            RefTemplate("node", "metric", "metric_{X}"),
        ),
        family_selectors=(),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
    )
    later = replace(
        base,
        ontology=replace(
            base.ontology,
            agg_kinds=("AVG",),
        ),
        max_degree=2,
        time_index="metric_n0",
        group_keys=("metric_n1",),
        condition_columns={"metric_n0": (1.0, 2.0)},
        metadata_columns=("metric_n1",),
        temporal_enabled=True,
        conditional_enabled=True,
    )

    merged = _merge_specs(
        base,
        later,
        columns=("metric_n0", "metric_n1"),
    )
    dataset, grammar = build_dataframe_grammar(
        pd.DataFrame({
            "metric_n0": np.arange(8.0),
            "metric_n1": np.arange(8.0) + 1.0,
        }),
        merged,
        name="base_interpretation",
    )

    assert merged.time_index == base.time_index
    assert merged.group_keys == base.group_keys
    assert merged.condition_columns == base.condition_columns
    assert merged.metadata_columns == base.metadata_columns
    assert set(dataset.observed.names) == {
        "metric_n0",
        "metric_n1",
    }
    assert set(grammar.agg_kinds) == {"SUM", "AVG"}
    assert grammar.max_degree == 2
    assert grammar.temporal_enabled
    assert grammar.conditional_enabled


def test_merge_specs_admits_new_metadata_without_removing_base_measurement():
    base = GrammarSpec(
        name="metric-base",
        patterns=(
            ColumnPattern(
                "metric",
                "regex",
                "measurement",
                "value",
                regex=r"^metric$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("metric",)},
            fam_roles={"record": ()},
        ),
        ref_templates=(
            RefTemplate("record", "metric", "metric"),
        ),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    later = replace(
        base,
        patterns=(
            ColumnPattern(
                "label",
                "regex",
                "metadata",
                "label",
                regex=r"^label$",
            ),
        ),
        metadata_columns=("metric", "label"),
    )

    merged = _merge_specs(
        base,
        later,
        columns=("metric", "label"),
    )
    dataset, _ = build_dataframe_grammar(
        pd.DataFrame({
            "metric": [1.0, 2.0, 3.0],
            "label": ["red", "green", "blue"],
        }),
        merged,
        name="new_metadata",
    )

    assert merged.metadata_columns == ("label",)
    assert dataset.observed.names == ["metric"]
    np.testing.assert_array_equal(
        dataset.observed.matrix[:, 0],
        np.array([1.0, 2.0, 3.0]),
    )
    assert dataset.observed.row_context["label"].tolist() == [
        "red",
        "green",
        "blue",
    ]


def test_merge_specs_admits_new_condition_without_reclassifying_measurement():
    base = GrammarSpec(
        name="metric-base",
        patterns=(
            ColumnPattern(
                "metric",
                "regex",
                "measurement",
                "value",
                regex=r"^metric$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("metric",)},
            fam_roles={"record": ()},
        ),
        ref_templates=(
            RefTemplate("record", "metric", "metric"),
        ),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    later = replace(
        base,
        patterns=(
            ColumnPattern(
                "label",
                "regex",
                "metadata",
                "label",
                regex=r"^label$",
            ),
        ),
        condition_columns={"label": ("red", "blue")},
        conditional_enabled=True,
    )

    merged = _merge_specs(
        base,
        later,
        columns=("metric", "label"),
    )
    dataset, grammar = build_dataframe_grammar(
        pd.DataFrame({
            "metric": [1.0, 2.0, 3.0],
            "label": ["red", "blue", "red"],
        }),
        merged,
        name="new_condition",
    )

    assert merged.condition_columns == {
        "label": ("red", "blue"),
    }
    assert dataset.observed.names == ["metric"]
    assert dataset.observed.row_context["label"].tolist() == [
        "red",
        "blue",
        "red",
    ]
    assert grammar.conditional_enabled


def test_merge_specs_preserves_unpatterned_base_condition_domain():
    base = replace(
        _mini_spec(),
        condition_columns={"label": ("red", "blue")},
        metadata_columns=("label",),
        conditional_enabled=True,
    )
    later = replace(
        _mini_spec(),
        condition_columns={"label": ("red",)},
        metadata_columns=(),
        conditional_enabled=True,
    )

    merged = _merge_specs(
        base,
        later,
        columns=("label", "metric"),
    )

    assert merged.condition_columns == {
        "label": ("red", "blue"),
    }
    assert merged.metadata_columns == ("label",)


def test_merge_specs_rejects_context_column_in_numeric_family():
    base = GrammarSpec(
        name="family-base",
        patterns=(
            ColumnPattern(
                "part",
                "regex",
                "part",
                "value",
                regex=r"^part_a$",
            ),
            ColumnPattern(
                "total",
                "regex",
                "total",
                "value",
                regex=r"^total$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("total",)},
            fam_roles={"record": ("parts",)},
        ),
        ref_templates=(
            RefTemplate("record", "total", "total"),
        ),
        family_selectors=(
            FamilySelector(
                "record",
                "parts",
                match_kind="part",
            ),
        ),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )
    later = replace(
        base,
        patterns=(
            ColumnPattern(
                "part_b",
                "regex",
                "part",
                "value",
                regex=r"^part_b$",
            ),
        ),
        condition_columns={"part_b": ("red", "blue")},
        conditional_enabled=True,
    )

    with pytest.raises(
        ValueError,
        match="context columns also ground numeric grammar roles.*parts",
    ):
        _merge_specs(
            base,
            later,
            columns=("part_a", "part_b", "total"),
        )


def test_merge_specs_validates_context_grounding_for_patternless_base():
    base = GrammarSpec(
        name="empty-base",
        patterns=(),
        ontology=RoleOntology(
            binders=(),
            ref_roles={},
            fam_roles={},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={},
        cell_codec=CellCodec(kind="scalar"),
    )
    later = GrammarSpec(
        name="later",
        patterns=(
            ColumnPattern(
                "x",
                "regex",
                "part",
                "value",
                regex=r"^x$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ("parts",)},
        ),
        ref_templates=(),
        family_selectors=(
            FamilySelector(
                "record",
                "parts",
                match_kind="part",
            ),
        ),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
        condition_columns={"x": ("red", "blue")},
        conditional_enabled=True,
    )

    with pytest.raises(
        ValueError,
        match="context columns also ground numeric grammar roles.*parts",
    ):
        _merge_specs(base, later, columns=("x",))


def test_merge_specs_rejects_new_numeric_role_for_base_context():
    base = GrammarSpec(
        name="context-base",
        patterns=(
            ColumnPattern(
                "metric",
                "regex",
                "measurement",
                "value",
                regex=r"^metric$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("metric",)},
            fam_roles={"record": ()},
        ),
        ref_templates=(
            RefTemplate("record", "metric", "metric"),
        ),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
        condition_columns={"label": ("red", "blue")},
        metadata_columns=("label",),
        conditional_enabled=True,
    )
    later = replace(
        base,
        patterns=(
            ColumnPattern(
                "label",
                "regex",
                "measurement",
                "value",
                regex=r"^label$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("label",)},
            fam_roles={"record": ()},
        ),
        ref_templates=(
            RefTemplate("record", "label", "label"),
        ),
        condition_columns={},
        metadata_columns=(),
    )

    with pytest.raises(
        ValueError,
        match="context columns also ground numeric grammar roles.*label",
    ):
        _merge_specs(
            base,
            later,
            columns=("metric", "label"),
        )


def test_merge_specs_rechecks_context_after_boolean_role_demotion():
    columns = ("value_n0", "value_n1")
    base = GrammarSpec(
        name="numeric-base",
        patterns=(
            ColumnPattern(
                "value",
                "regex",
                "measurement",
                "value",
                regex=r"^value_(?P<node>n\d+)$",
                node_groups=("node",),
                token_groups=("node",),
            ),
        ),
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("value",)},
            fam_roles={"node": ()},
        ),
        ref_templates=(
            RefTemplate("node", "value", "value_{X}"),
        ),
        family_selectors=(),
        binder_enumerate={"node": "per_node"},
        cell_codec=CellCodec(kind="scalar"),
        condition_columns={"value_n1": (False, True)},
        metadata_columns=(),
        conditional_enabled=True,
    )
    later = replace(
        base,
        ontology=RoleOntology(
            binders=("node",),
            ref_roles={"node": ("alias",)},
            fam_roles={"node": ()},
        ),
        ref_templates=(
            RefTemplate("node", "alias", "value_{X}"),
        ),
        boolean_roles={"node": ("alias",)},
    )

    with pytest.raises(
        ValueError,
        match="context columns also ground numeric grammar roles.*alias",
    ):
        _merge_specs(base, later, columns=columns)


def test_runtime_tier_pins_do_not_erase_accumulated_capabilities():
    base = replace(
        _mini_spec(),
        conditional_enabled=True,
        condition_columns={"status": ("ok", "bad")},
    )
    pinned_runtime = replace(base, conditional_enabled=False)
    later = replace(
        _mini_spec(),
        conditional_enabled=False,
        condition_columns={},
    )

    accumulated = _merge_specs(base, later)

    assert not pinned_runtime.conditional_enabled
    assert accumulated.conditional_enabled
    assert accumulated.condition_columns == {
        "status": ("ok", "bad"),
    }


def test_known_split_keeps_singleton_sum_balance_aliases_together():
    known = [
        KnownInvariant(
            "ref_sum",
            "==",
            "total",
            {"sum": ["a", "b"]},
        ),
        KnownInvariant(
            "singleton_balance",
            "==",
            {"sum": ["total"]},
            {"sum": ["a", "b"]},
        ),
        KnownInvariant("other", "==", "x", "y"),
        KnownInvariant("third", ">=", "z", 0),
    ]

    for seed in range(25):
        calibration, validation = _split_known(
            known,
            frac=0.5,
            seed=seed,
        )
        calibration_names = {item.name for item in calibration}
        validation_names = {item.name for item in validation}
        assert calibration_names.isdisjoint(validation_names)
        assert (
            ("ref_sum" in calibration_names)
            == ("singleton_balance" in calibration_names)
        ), seed


def test_known_split_canonicalizes_zero_members_in_singleton_sum_aliases():
    from autogram.calibrate import _ColumnScaleView

    frame = pd.DataFrame({
        "total": np.full(60, 10.0),
        "a": np.full(60, 10.0),
        "z": np.zeros(60),
        "x": np.linspace(1.0, 2.0, 60),
        "y": np.linspace(3.0, 4.0, 60),
    })
    known = [
        KnownInvariant("ref_sum", "==", "total", {"sum": ["a"]}),
        KnownInvariant(
            "singleton_balance",
            "==",
            {"sum": ["total"]},
            {"sum": ["a", "z"]},
        ),
        KnownInvariant("other", "==", "x", "y"),
        KnownInvariant("third", ">=", "x", 0),
    ]

    for seed in range(25):
        calibration, validation = _split_known(
            known,
            frac=0.5,
            seed=seed,
            frame=_ColumnScaleView(frame),
        )
        calibration_names = {item.name for item in calibration}
        validation_names = {item.name for item in validation}
        assert calibration_names.isdisjoint(validation_names)
        assert (
            ("ref_sum" in calibration_names)
            == ("singleton_balance" in calibration_names)
        ), seed


def _crosscheck_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if str(column).startswith(("low_", "high_"))]


def test_column_scale_view_decodes_dict_cells_like_the_runtime_frame():
    """Round-29 / TODO-3: the split's view and the runtime frame must not decode differently.

    CrossCheck cells are dicts whose ``ground_truth`` key carries the datum. ``pd.to_numeric``
    coerces every one of them to ``NaN``, so the view and the runtime frame disagreed about a
    column's data -- and therefore about which summed members are negligible -- which is exactly how
    an alias pair can straddle a split that is supposed to hold the validation half out.
    """
    from autogram.calibrate import _ColumnScaleView
    from autogram.discovery.loop import build_dataframe_grammar
    from tests.test_crosscheck_golden import _crosscheck_spec

    df = pd.read_pickle("data/crosscheck-samples/abilene_sample_1000.pkl")
    columns = _crosscheck_columns(df)
    assert columns

    # The naive coercion the view used to perform loses the data entirely.
    naive = pd.to_numeric(df[columns[0]], errors="coerce").to_numpy(dtype=float)
    assert np.all(np.isnan(naive))

    dataset, _grammar = build_dataframe_grammar(
        df, _crosscheck_spec(link_demand_context=True), name="abilene_view_check",
    )
    view = _ColumnScaleView(df)

    compared = 0
    for column in columns:
        if not dataset.observed.has(column):
            continue
        assert np.array_equal(view.col(column), dataset.observed.col(column), equal_nan=True)
        compared += 1
    assert compared > 0
    assert not np.all(np.isnan(view.col(columns[0])))
    view.assert_matches_runtime(dataset.observed)     # must not raise


def test_known_split_drops_zero_members_before_sum_shape_classification():
    from autogram.calibrate import _ColumnScaleView

    frame = pd.DataFrame({
        "total": np.full(60, 10.0),
        "a": np.full(60, 4.0),
        "b": np.full(60, 6.0),
        "z": np.zeros(60),
        "x": np.arange(60.0),
    })
    known = [
        KnownInvariant(
            "plain",
            "==",
            "total",
            {"sum": ["a", "b"]},
        ),
        KnownInvariant(
            "padded",
            "==",
            {"sum": ["total", "z"]},
            {"sum": ["a", "b"]},
        ),
        KnownInvariant("other", ">=", "x", 0),
    ]

    calibration, validation = _split_known(
        known,
        frac=0.5,
        seed=0,
        frame=_ColumnScaleView(frame),
    )
    calibration_names = {item.name for item in calibration}
    validation_names = {item.name for item in validation}

    assert {"plain", "padded"} <= calibration_names or {
        "plain",
        "padded",
    } <= validation_names


def test_split_keeps_aliases_together_on_dict_valued_cells():
    """The alias pair must stay on one side of the split for CrossCheck-shaped data too.

    With ``pd.to_numeric`` every column decoded to all-``NaN``, so no member ever looked negligible
    and ``total == SUM(a)`` / ``total == SUM(a, z)`` were treated as two different relations.
    """
    from autogram.calibrate import _ColumnScaleView

    n = 60
    def cells(values):
        return [
            {"ground_truth": float(value), "hidden_ground_truth": float(value)}
            for value in values
        ]

    df = pd.DataFrame({
        "total": cells(np.linspace(10.0, 20.0, n)),
        "a": cells(np.linspace(10.0, 20.0, n)),
        "z": cells(np.zeros(n)),
        "p": cells(np.linspace(1.0, 2.0, n)),
        "q": cells(np.linspace(3.0, 4.0, n)),
    })
    known = [
        KnownInvariant("plain", "==", "total", {"sum": ["a"]}),
        KnownInvariant("padded", "==", "total", {"sum": ["a", "z"]}),
        KnownInvariant("other", "==", "p", "q"),
        KnownInvariant("third", "==", "q", "p"),
        KnownInvariant("fourth", ">=", "p", 0),
    ]

    for seed in range(25):
        calibration, validation = _split_known(
            known, frac=0.4, seed=seed, frame=_ColumnScaleView(df),
        )
        calib = {item.name for item in calibration}
        valid = {item.name for item in validation}
        assert calib.isdisjoint(valid)
        assert ("plain" in calib) == ("padded" in calib), seed


def test_column_scale_view_fails_loudly_on_an_undecodable_cell():
    from autogram.calibrate import _ColumnScaleView

    df = pd.DataFrame({"total": [{"value": 1.0}, {"value": 2.0}]})

    with pytest.raises(ValueError, match="cannot decode column"):
        _ColumnScaleView(df).col("total")


def test_column_scale_view_detects_a_runtime_decoding_mismatch():
    from autogram.calibrate import _ColumnScaleView

    class _Frame:
        def has(self, column: str) -> bool:
            return True

        def col(self, column: str) -> np.ndarray:
            return np.zeros(4)

    df = pd.DataFrame({
        "total": [{"ground_truth": float(v)} for v in (1.0, 2.0, 3.0, 4.0)],
    })
    view = _ColumnScaleView(df)
    view.col("total")

    with pytest.raises(ValueError, match="differently from the runtime frame"):
        view.assert_matches_runtime(_Frame())


def test_column_scale_view_fails_loudly_on_a_non_numeric_cell():
    from autogram.calibrate import _ColumnScaleView

    df = pd.DataFrame({"archetype": ["steady", "burst", "drain"]})

    with pytest.raises(ValueError, match="is not numeric"):
        _ColumnScaleView(df).col("archetype")
