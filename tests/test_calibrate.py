"""Fast tests for the calibration wiring: capability widening, tiers, ladder, defaults."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from autogram.config import DiscoveryConfig
from autogram.calibrate import (
    CalibrationConfig, _capability_tiers, _derive_regime, _knob_schedule, _merge_specs,
    _spec_summary, _widen_spec, calibrate,
)
from autogram.discovery.known import KnownInvariant
from autogram.discovery.regime import ProxyEntry, RegimeSpec
from autogram.schema.spec import (
    ColumnPattern, FamilySelector, GrammarSpec, RefTemplate, RoleOntology,
)


def _mini_spec(agg=("SUM",), max_degree=1, role_exclusions=()):
    onto = RoleOntology(binders=("node",), ref_roles={"node": ("a", "b")},
                        fam_roles={"node": ("fam",)}, agg_kinds=agg)
    return GrammarSpec(name="t", patterns=(), ontology=onto, ref_templates=(),
                       family_selectors=(), binder_enumerate={"node": "per_node"},
                       max_degree=max_degree, role_exclusions=role_exclusions)


def test_calibration_defaults_global_while_discovery_stays_adaptive():
    # The engine default is unchanged; calibration (config + CLI) defaults to one fixed global band.
    from autogram.cli import build_parser
    assert DiscoveryConfig().band_mode == "adaptive"
    assert CalibrationConfig().band_mode == "global"
    assert CalibrationConfig().max_capability_tiers == 3
    a = build_parser().parse_args(["calibrate", "--input", "x.pkl", "--known", "k.yaml"])
    assert a.band_mode == "global"


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


def test_spec_summary_reports_capabilities():
    s = _spec_summary(_mini_spec(agg=("SUM", "MIN"), max_degree=2), tier=1, caps={"all_aggs": True})
    assert s["tier"] == 1 and s["max_degree"] == 2
    assert s["agg_kinds"] == ["SUM", "MIN"] and s["n_ref_roles"] == 2


def test_calibration_config_saves_rules_by_default():
    c = CalibrationConfig()
    assert c.save_rules is True and c.rules_dir == "rules"


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


def _fake_calibrate_env(monkeypatch, *, recall_fn, null_fn, tune=None):
    """Patch calibrate's discovery seams so the loop runs with no LLM / no enumeration."""
    import autogram.calibrate as C
    from autogram.discovery.validate import PreparedProxy, ProxySuite

    def fake_prepare(regime, seed=0, inducer=None):
        pos = [PreparedProxy(e.shape, object(), object(), {}) for e in regime.active_entries()]
        return ProxySuite(positives=pos, null=PreparedProxy("null", object(), object(), {}))

    def default_tune(suite, seed=0, band_mode="global", null_floor=0.5, max_expansions=3, **kw):
        shapes = [p.shape for p in suite.positives]
        return {"tolerance": 0.05, "hold_rate_threshold": 0.66, "expansions": 0,
                "selected_null_equalities": 0, "proxy_shapes": shapes,
                "per_proxy": [{"shape": s, "recovery": 0.9, "accepted": 3, "compact": True,
                               "scaled_slack": []} for s in shapes]}

    monkeypatch.setattr(C, "make_inducer", lambda *a, **k: object())
    monkeypatch.setattr(C, "prepare_proxy_suite", fake_prepare)
    monkeypatch.setattr(C, "tune_joint", tune or default_tune)
    monkeypatch.setattr(C, "induce_spec", lambda cols, inducer, *a, **k: _mini_spec())
    monkeypatch.setattr(C, "build_dataframe_grammar",
                        lambda df, spec, search_cfg=None, name="": (object(), object()))
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


def test_derive_regime_rejects_custom_regime_with_no_active_entries():
    # A caller-supplied regime with nothing active would prepare no positive proxies; reject it
    # clearly before any preparation, instead of silently "calibrating" on an empty suite.
    for empty in (RegimeSpec(entries=[]),
                  RegimeSpec(entries=[ProxyEntry("offset_pair", active=False)])):
        with pytest.raises(ValueError):
            _derive_regime(CalibrationConfig(regime=empty), [KnownInvariant("i1", "~=", "x", "y")])


def test_calibrate_memoizes_null_gate_across_tiers(monkeypatch, tmp_path):
    # Across multiple grammar tiers the same relaxation-ladder rungs recur; the per-rung null gate
    # must be memoized by (band_mode, tolerance, threshold) so each unique config is scored once,
    # while unsafe rungs still cannot win.
    import autogram.calibrate as C
    from autogram.discovery.validate import PreparedProxy, ProxySuite

    def fake_prepare(regime, seed=0, inducer=None):
        pos = [PreparedProxy(e.shape, object(), object(), {}) for e in regime.active_entries()]
        return ProxySuite(positives=pos, null=PreparedProxy("null", object(), object(), {}))

    def fake_tune(suite, seed=0, band_mode="global", null_floor=0.5, max_expansions=3, **kw):
        shapes = [p.shape for p in suite.positives]
        return {"tolerance": 0.05, "hold_rate_threshold": 0.66, "expansions": 0,
                "selected_null_equalities": 0, "proxy_shapes": shapes,
                "per_proxy": [{"shape": s, "recovery": 0.9, "accepted": 3, "compact": True,
                               "scaled_slack": []} for s in shapes]}

    tier_counter = {"n": -1}

    def fake_build(df, spec, search_cfg=None, name=""):
        tier_counter["n"] += 1
        return (SimpleNamespace(tier=tier_counter["n"]), object())

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

    # (a) three tiers ran, but each unique (band_mode, tolerance, threshold) config scored once
    assert report["grammar_reinductions"] == 2                    # tiers 1 and 2 re-induced
    assert len(null_calls) == len(set(null_calls)) == 4           # 4 distinct rungs, no re-eval
    # (b) an unsafe relaxed rung (higher recall) still cannot win
    assert report["config"]["tolerance"] == 0.05
    assert report["false_discovery"]["null_equalities_accepted"] == 0
    unsafe = [h for h in report["trajectory"] if h["null_equalities"] >= 1]
    assert unsafe and max(h["calibration_recall"] for h in unsafe) > report["recall_all"]
