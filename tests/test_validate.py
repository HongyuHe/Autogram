"""Adversarial validation harness: the proof of discovery without ground truth."""

from __future__ import annotations

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery import validate as V
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.induce import HeuristicInducer, induce_spec
from autogram.discovery.loop import discover
from autogram.dsl import ast as A
from autogram.dsl.grammar import grammar_from_adapter
from autogram.loader.loader import build_dataset
from autogram.schema.compiler import compile_spec


def test_plant_and_recover_across_noise():
    pr = V.plant_and_recover(noise_levels=(0.0, 0.05), n_entities=4,
                             n_snapshots=180, seed=0)
    assert set(pr.recovered) >= {"row_sum", "col_sum", "two_end", "self_zero"}
    for family, by_noise in pr.recovered.items():
        assert by_noise[0.0], family
        assert all(by_noise[nz] for nz in (0.0, 0.05)), family


def test_null_dataset_fdr_control():
    assert V.null_accepted(n_entities=4, n_snapshots=160, seed=0) == 0


def test_tautology_rejection():
    tc = V.tautology_check(seed=0)
    assert tc.self_comparison_admissible is False
    assert tc.nonneg_accepted is False


def test_rename_invariance():
    ri = V.rename_invariance(n_entities=4, n_snapshots=160, seed=0)
    assert ri.invariant and ri.overlap >= 0.9


def test_drop_name_induction_fails_on_renamed_schema():
    v2 = synth.Vocab(meas="signal", demand="route", src="out", dst="inn",
                     to="unto", frm="fro", entity_prefix="z")
    d = synth.make_synthetic(n_entities=4, n_snapshots=140, noise=0.02, seed=0, vocab=v2)
    res = discover(d.columns, d.matrix, inducer=V.FixedVocabInducer(),
                   discovery_cfg=DiscoveryConfig(n_perm=8, seed=0),
                   search_cfg=SearchConfig(rounds=3, proposals_per_round=60, seed=0),
                   name="fixed", timestamps=d.timestamps)
    assert len(res.portfolio) == 0


def test_drop_lift_admits_spurious_on_null():
    """Disabling the lift/null guard admits spurious null-correlated rules.

    With the guard ENABLED, independent columns yield ~no accepted rules (FDR control); DISABLING
    it (alpha=1, no lift test) admits substantially more spurious rules.  This is the precise,
    demonstrated claim -- not literal-tautology admission (self-comparisons stay structurally
    inadmissible regardless of lift).
    """
    d = synth.make_null(n_entities=4, n_snapshots=160, seed=0)
    base = discover(d.columns, d.matrix,
                    discovery_cfg=DiscoveryConfig(n_perm=8, seed=0),
                    search_cfg=SearchConfig(rounds=3, proposals_per_round=80, seed=0),
                    name="b", timestamps=d.timestamps)
    nolift = discover(d.columns, d.matrix,
                      discovery_cfg=DiscoveryConfig(n_perm=8, alpha=1.0,
                                                    require_lift=False,
                                                    require_null_support=False,
                                                    require_parsimony=False,
                                                    seed=0),
                      search_cfg=SearchConfig(rounds=3, proposals_per_round=80, seed=0),
                      name="nl", timestamps=d.timestamps)
    assert len(base.portfolio) == 0
    assert len(nolift.portfolio) > len(base.portfolio)
    # the admitted rules are spurious: their lift percentile sits near the middle of the null.
    assert any(e.lift_percentile > 0.2 for e in nolift.portfolio)


def _regime_unstable_rule(seed=0, unstable_frac=0.2):
    """Build the regime dataset and return (evaluator_factory, the unstable two-end rule).

    The two-end pairing (``to_ij == from_ji``) is the planted regime trap: it agrees on the first
    ``1-unstable_frac`` of rows and shifts regime afterwards.
    """
    real = synth.make_synthetic(n_entities=5, n_snapshots=300, noise=0.05, seed=seed,
                                unstable_frac=unstable_frac, regime_factor=1.8)
    spec = induce_spec(real.columns, HeuristicInducer(), None)
    adapter = compile_spec(spec)
    ds = build_dataset(real.columns, real.matrix, adapter, "regime", real.timestamps)
    G = grammar_from_adapter(adapter, 12, 3)

    # find the two-end pairing (highest-lift admissible link Ref-vs-Ref equality)
    ev = DataOnlyEvaluator(ds, DiscoveryConfig(n_perm=16, seed=seed))
    best, best_ev = None, None
    refs = G.refs_for("link")
    for a in refs:
        for b in refs:
            if a >= b:
                continue
            r = A.Rule("link", A.Compare(A.Ref(a), "==", A.Ref(b)))
            e = ev.evaluate(r)
            if e.lift > 3 and (best_ev is None or e.lift_percentile < best_ev.lift_percentile):
                best, best_ev = r, e
    return ds, best, best_ev


def test_stability_gate_rejects_unstable_rule_admitted_by_loose():
    """The stability gate (and nothing else) rejects a high-support, high-lift but UNSTABLE rule;
    dropping stability admits it.  This pins the corrected drop-stability ablation semantics."""
    ds, rule, es = _regime_unstable_rule(seed=0)
    assert rule is not None and es is not None
    # passes support + the name-permutation lift test
    assert es.lift > 1.0 and es.lift_percentile <= 0.05
    assert es.n_bindings >= 2 and es.coverage_lo > 0.0
    # genuinely unstable: held-out coverage falls to the null/by-chance level on a split
    assert es.stability_margin <= 0.0
    # rejected ONLY by the stability gate
    assert not es.accepted and "stable" in es.reason
    # dropping the stability gate admits it
    loose = DataOnlyEvaluator(ds, DiscoveryConfig(n_perm=16, require_stability=False,
                                                  seed=0))
    el = loose.evaluate(rule)
    assert el.accepted


def test_unstable_frac_default_is_a_noop():
    """unstable_frac=0 leaves the directed two-end agreement exact (no regression to defaults)."""
    d0 = synth.make_synthetic(n_entities=4, n_snapshots=50, noise=0.0, seed=0)
    d1 = synth.make_synthetic(n_entities=4, n_snapshots=50, noise=0.0, seed=0, unstable_frac=0.0)
    assert (d0.matrix == d1.matrix).all()


def test_ablations_report_is_real():
    """The full ablation report is honest: drop-stability strictly admits more (unstable) rules,
    drop-lift admits spurious null rules from a controlled base."""
    abl = V.ablations(seed=0)
    assert abl.drop_stability_more_overfit
    assert abl.drop_lift_admits_spurious
    assert abl.drop_induction_fails
    ds = abl.detail["drop_stability"]
    assert ds["loose_unstable_count"] > ds["strict_unstable_count"]
    assert ds["strict_unstable_count"] == 0
    assert len(ds["overfit_admitted_by_loose"]) >= 1
    assert all(x["stability_margin"] <= 0.0 for x in ds["overfit_admitted_by_loose"])
    dl = abl.detail["drop_lift"]
    assert dl["base_null_accepted"] == 0
    assert dl["pre_mdl_no_lift_spurious"] > dl["base_null_accepted"]


def test_portfolio_quality_rejects_subpar_proxy_metrics():
    q = V.portfolio_quality(seed=0)
    assert q["ok"]
    assert q["accepted"] >= 1
    assert not q["negative_mdl_rules"]
    assert not q["lift_fail_rules"]
