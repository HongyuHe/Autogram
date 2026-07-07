"""Fast tests for the calibration wiring: capability widening, tiers, ladder, defaults."""

from __future__ import annotations

from dataclasses import replace

from autogram.config import DiscoveryConfig
from autogram.calibrate import (
    CalibrationConfig, _capability_tiers, _knob_schedule, _merge_specs, _spec_summary, _widen_spec,
)
from autogram.schema.spec import (
    ColumnPattern, FamilySelector, GrammarSpec, RefTemplate, RoleOntology,
)


def _mini_spec(agg=("SUM",), max_degree=1, role_exclusions=()):
    onto = RoleOntology(binders=("node",), ref_roles={"node": ("a", "b")},
                        fam_roles={"node": ("fam",)}, agg_kinds=agg)
    return GrammarSpec(name="t", patterns=(), ontology=onto, ref_templates=(),
                       family_selectors=(), binder_enumerate={"node": "per_node"},
                       max_degree=max_degree, role_exclusions=role_exclusions)


def test_default_band_mode_is_adaptive():
    assert DiscoveryConfig().band_mode == "adaptive"
    assert CalibrationConfig().band_mode == "adaptive"
    assert CalibrationConfig().max_capability_tiers == 3


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
