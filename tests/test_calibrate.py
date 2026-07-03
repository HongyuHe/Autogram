"""Fast tests for the calibration wiring: capability widening, tiers, ladder, defaults."""

from __future__ import annotations

from autogram.config import DiscoveryConfig
from autogram.calibrate import (
    CalibrationConfig, _capability_tiers, _knob_schedule, _spec_summary, _widen_spec,
)
from autogram.schema.spec import GrammarSpec, RoleOntology


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
