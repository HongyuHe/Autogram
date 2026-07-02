"""v2 proxy validation harness."""

from __future__ import annotations

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.cli import build_parser
from autogram.discovery import synth
from autogram.discovery import validate as V
from autogram.discovery.loop import discover
from autogram.dsl import ast as A


def test_plant_and_recover_across_noise():
    pr = V.plant_and_recover(noise_levels=(0.0, 0.02, 0.05), n_entities=4, n_snapshots=120, seed=0)
    assert set(pr.recovered) >= {
        "row_sum", "col_sum", "two_end", "self_zero",
        "offset_pair", "agg_ref_balance", "presence_pair",
    }
    assert any(any(by_noise.values()) for by_noise in pr.recovered.values())


def test_systematic_offset_family_recovers_near_two_thirds_hold_rate():
    data = synth.make_synthetic(
        n_entities=4, n_snapshots=180, noise=0.0, seed=0,
        families=("offset_pair",), offset_hold_rate=0.67, offset_factor=0.98,
    )
    res = discover(
        data.columns, data.matrix,
        discovery_cfg=DiscoveryConfig(seed=0, tolerance=0.01, hold_rate_threshold=0.62),
        search_cfg=SearchConfig(seed=0, max_complexity=8),
        name="offset", timestamps=data.timestamps,
    )
    rec = V.score_recovery(res, data.planted)
    assert rec.offset_pair >= 0.8
    offset_rules = [
        e for e in res.portfolio
        if e.rule.atom.op == "~=" and e.rule.atom.left == A.Ref("o0_rev") and e.rule.atom.right == A.Ref("o1")
    ]
    assert offset_rules
    assert 0.64 <= offset_rules[0].hold_rate <= 0.70


def test_agg_ref_balance_family_needs_mixed_add_terms():
    data = synth.make_synthetic(
        n_entities=4, n_snapshots=120, noise=0.0, seed=0,
        families=("agg_ref_balance",),
    )
    res = discover(
        data.columns, data.matrix,
        discovery_cfg=DiscoveryConfig(seed=0, tolerance=0.01, hold_rate_threshold=0.95),
        search_cfg=SearchConfig(seed=0, max_complexity=10, max_add_arity=2),
        name="agg_ref", timestamps=data.timestamps,
    )
    rec = V.score_recovery(res, data.planted)
    assert rec.agg_ref_balance >= 0.8
    assert any(
        isinstance(e.rule.atom.left, A.Add) and isinstance(e.rule.atom.right, A.Add)
        and any(isinstance(t, A.Agg) for t in e.rule.atom.left.terms + e.rule.atom.right.terms)
        and any(isinstance(t, A.Ref) for t in e.rule.atom.left.terms + e.rule.atom.right.terms)
        for e in res.portfolio
    )


def test_default_synthetic_highest_arity_family_is_admissible_at_cli_default_bound():
    args = build_parser().parse_args(["discover"])
    data = synth.make_synthetic(n_entities=4, n_snapshots=120, noise=0.0, seed=0)

    assert "agg_ref_balance" in data.planted

    res = discover(
        data.columns, data.matrix,
        discovery_cfg=DiscoveryConfig(
            seed=args.seed,
            tolerance=args.tolerance,
            hold_rate_threshold=args.hold_rate,
            ci_alpha=args.ci_alpha,
        ),
        search_cfg=SearchConfig(
            seed=args.seed,
            max_complexity=args.max_complexity,
            max_add_arity=args.max_add_arity,
        ),
        name="default_synthetic", timestamps=data.timestamps,
    )
    rec = V.score_recovery(res, data.planted)
    assert rec.agg_ref_balance >= 0.8


def test_presence_pairing_family_uses_existence_operator():
    data = synth.make_synthetic(
        n_entities=4, n_snapshots=160, noise=0.0, seed=0,
        families=("presence_pair",), presence_rate=0.55,
    )
    res = discover(
        data.columns, data.matrix,
        discovery_cfg=DiscoveryConfig(seed=0, hold_rate_threshold=0.95),
        search_cfg=SearchConfig(seed=0, max_complexity=8),
        name="presence", timestamps=data.timestamps,
    )
    rec = V.score_recovery(res, data.planted)
    assert rec.presence_pair >= 0.8
    assert any(e.rule.atom.op == "<|>" for e in res.portfolio)


def test_null_dataset_has_no_equalities():
    assert V.null_accepted(n_entities=4, n_snapshots=120, seed=0) == 0


def test_structural_families_reports_v2_classes():
    d = synth.make_synthetic(n_entities=4, n_snapshots=120, noise=0.02, seed=0)
    res = discover(d.columns, d.matrix,
                   discovery_cfg=DiscoveryConfig(seed=0, hold_rate_threshold=0.9),
                   search_cfg=SearchConfig(seed=0, max_complexity=8),
                   name="families", timestamps=d.timestamps)
    fams = V.structural_families(res)
    assert "one-sided nonnegativity/bound" in fams
    assert "aggregate sum conservation" in fams or "pairwise equality/order" in fams


def test_portfolio_quality_uses_hold_rate_only():
    q = V.portfolio_quality(seed=0)
    assert q["ok"]
    assert q["accepted"] >= 1
    assert q["accepted"] < 250
    assert not q["bad_hold_rate_rules"]
    assert not q["scaled_slack_rules"]


def test_run_all_reports_proxy_phase():
    report = V.run_all(seed=0)
    assert "proxy_ok" in report
    assert "synthetic_recovery" in report
    assert report["portfolio_quality"]["ok"]


def test_proxy_tune_validates_returned_runtime_config():
    tuned = V.proxy_tune(seed=0)
    runtime = tuned["runtime_recovery"]
    assert tuned["runtime_discovery"].tolerance == tuned["discovery"].tolerance
    assert tuned["runtime_discovery"].hold_rate_threshold == tuned["discovery"].hold_rate_threshold
    assert set(runtime) >= {"row_sum", "col_sum", "two_end", "self_zero", "agg_ref_balance", "presence_pair"}
    assert all(runtime.values())
