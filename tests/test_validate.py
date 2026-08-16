"""v2 proxy validation harness."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.cli import build_parser
from autogram.discovery import synth
from autogram.discovery import regime as R
from autogram.discovery import validate as V
from autogram.discovery.loop import discover
from autogram.dsl import ast as A
from autogram.dsl.evaluate import typed_group_key


def test_runtime_null_covers_each_typed_condition_domain_when_product_is_too_large():
    adapter = SimpleNamespace(condition_columns={
        f"c{column}": tuple(
            f"c{column}-v{value}"
            for value in range(10)
        )
        for column in range(3)
    })

    context = V._runtime_condition_context(
        adapter,
        100,
        np.random.default_rng(0),
        randomize=True,
    )

    for name, domain in adapter.condition_columns.items():
        assert {
            typed_group_key(value)
            for value in context[name].tolist()
        } == {
            typed_group_key(value)
            for value in domain
        }


def test_nonneg_proxy_plants_only_nonnegativity():
    data = synth.make_synthetic(n_entities=3, n_snapshots=80, noise=0.0, seed=0,
                                families=("nonneg",))
    assert set(data.planted) == {"nonneg"}                 # no pair/sum/zero/balance/presence
    assert data.planted["nonneg"] == frozenset(data.columns)
    assert (data.matrix >= 0).all()
    assert (data.matrix == 0).any()                        # dropout breaks trivial presence pairing


def test_nonpos_proxy_plants_only_nonpositivity():
    data = synth.make_synthetic(n_entities=3, n_snapshots=80, noise=0.0, seed=0,
                                families=("nonpos",))
    assert set(data.planted) == {"nonpos"}
    assert data.planted["nonpos"] == frozenset(data.columns)
    assert (data.matrix <= 0).all()
    assert (data.matrix == 0).any()


def test_regime_generates_one_sided_proxies():
    for shape, sign in (("nonneg", 1.0), ("nonpos", -1.0)):
        d = R.generate(R.ProxyEntry(shape, n_entities=3, n_snapshots=60))
        assert set(d.planted) == {shape}
        assert (d.matrix * sign >= 0).all()


def test_algebraic_null_is_sign_balanced_per_column():
    data = synth.make_null(
        n_entities=4,
        n_snapshots=160,
        seed=0,
    )

    assert np.all(np.any(data.matrix > 0.0, axis=0))
    assert np.all(np.any(data.matrix < 0.0, axis=0))


def test_runtime_numeric_null_avoids_tied_temporal_differences():
    generated = V._balanced_null_numeric(
        np.resize(np.array([0.0, 1.0]), 1_000),
        np.random.default_rng(0),
        binary=False,
    )
    nonnegative_differences = np.mean(np.diff(generated) >= 0.0)

    assert np.unique(generated).size > 900
    assert 0.45 <= nonnegative_differences <= 0.55


def test_score_recovery_exposes_numeric_one_sided_families(monkeypatch):
    # nonneg/nonpos are surfaced through the same uniform numeric recovery interface joint tuning
    # uses (getattr(rec, shape) >= 0.8), i.e. coverage of the planted one-sided columns.
    from types import SimpleNamespace
    result = SimpleNamespace(portfolio=[])
    monkeypatch.setattr(V, "_portfolio_one_sided_columns",
                        lambda res, op: ({"a", "b", "c"} if op == ">=" else {"p"}),
                        raising=False)
    planted = {"nonneg": frozenset({"a", "b", "c", "d"}), "nonpos": frozenset({"p", "q"})}
    rec = V.score_recovery(result, planted)
    assert isinstance(rec.nonneg, float) and isinstance(rec.nonpos, float)
    assert rec.nonneg == 0.75          # 3 of 4 nonneg columns covered by a >= 0 rule
    assert rec.nonpos == 0.5           # 1 of 2 nonpos columns covered by a <= 0 rule
    assert rec.recovered is False      # neither family reaches the 0.8 target


def test_approximate_pair_does_not_recover_exact_pair_proxy(monkeypatch):
    pair = frozenset({"left", "right"})
    result = SimpleNamespace(portfolio=[])
    monkeypatch.setattr(
        V,
        "portfolio_relations",
        lambda _result: {
            (
                "equality",
                "approximate",
                ("pair", pair),
            )
        },
    )

    recovery = V.score_recovery(
        result,
        {
            "two_end": {pair},
            "offset_pair": {pair},
        },
    )

    assert recovery.two_end == 0.0
    assert recovery.offset_pair == 1.0


def test_relation_signature_matching_is_type_sensitive_for_categoricals():
    # Booleans and strings are exact categorical identities: True must not match 1, False must not
    # match 0, and a numeric threshold declared as an integer still matches a fitted float.
    assert not V.relation_signature_matches(True, 1)
    assert not V.relation_signature_matches(False, 0)
    assert not V.relation_signature_matches("1", 1)
    assert V.relation_signature_matches(True, True)
    assert V.relation_signature_matches(
        ("bound", ("ref", "x"), "<", 0),
        ("bound", ("ref", "x"), "<", 0.0),
    )
    assert V.relation_signature_matches(
        ("code", "in", (1, 2)),
        ("code", "in", (1, 2)),
    )
    assert not V.relation_signature_matches(
        ("flag", "==", (True,)),
        ("flag", "==", (1,)),
    )


def test_relation_signature_matching_keeps_structural_integers_exact():
    assert not V.relation_signature_matches(
        ("lag_bound", ("x", 99, ">=")),
        ("lag_bound", ("x", 100, ">=")),
    )
    assert V.relation_signature_matches(
        ("bound", ("ref", "x"), "<", 0.004),
        ("bound", ("ref", "x"), "<", 0.00400001),
    )


def _outcome(shape, recovery, accepted=5, compact=True, slack=None):
    return V.ProxyOutcome(shape=shape, recovery=recovery, accepted=accepted,
                          compact=compact, scaled_slack=list(slack or []))


def test_joint_candidate_rejected_when_any_proxy_below_target():
    c = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.85), _outcome("row_sum", 0.7)], 0)
    assert c.recovery_ok is False
    assert c.eligible is False
    assert V.select_candidate([c]) is None


def test_joint_candidate_rejected_with_one_null_equality():
    c = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.9)], null_equalities=1)
    assert c.null_safe is False
    assert c.eligible is False
    assert V.select_candidate([c]) is None


def test_joint_candidate_rejected_with_one_temporal_null_acceptance():
    c = V.GridCandidate(
        0.01,
        0.66,
        [_outcome("monotone", 0.9)],
        null_equalities=0,
        null_temporal=1,
    )
    assert c.null_safe is False
    assert c.eligible is False
    assert V.select_candidate([c]) is None


def test_joint_candidate_rejected_with_one_definition_null_acceptance():
    c = V.GridCandidate(
        0.01,
        0.66,
        [_outcome("sustained", 0.9)],
        null_equalities=0,
        null_temporal=0,
        null_definitions=1,
    )
    assert c.null_safe is False
    assert c.eligible is False


def test_joint_candidate_rejected_when_not_compact_or_has_scaled_slack():
    big = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.9, accepted=200, compact=False)], 0)
    slack = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.9, slack=["r <= -1*x"])], 0)
    assert big.eligible is False and slack.eligible is False


def test_joint_selection_prefers_strictest_eligible():
    # all eligible -> prefer smallest tolerance, then highest threshold
    loose = V.GridCandidate(0.05, 0.66, [_outcome("offset_pair", 1.0, accepted=5)], 0)
    strict = V.GridCandidate(0.01, 0.62, [_outcome("offset_pair", 0.9, accepted=9)], 0)
    stricter_thr = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.9, accepted=20)], 0)
    best = V.select_candidate([loose, strict, stricter_thr])
    assert (best.tolerance, best.hold_rate_threshold) == (0.01, 0.66)


def test_joint_selection_tie_breaks_on_portfolio_then_recovery():
    a = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.9, accepted=30)], 0)
    b = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.95, accepted=10)], 0)   # smaller total
    assert V.select_candidate([a, b]) is b
    c = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.85, accepted=10)], 0)
    d = V.GridCandidate(0.01, 0.66, [_outcome("offset_pair", 0.95, accepted=10)], 0)   # stronger recovery
    assert V.select_candidate([c, d]) is d


def _fake_suite(shapes=("offset_pair",)):
    # tune_joint only forwards the suite to `evaluate`, so a tiny stand-in with the shapes is enough.
    from types import SimpleNamespace
    positives = [SimpleNamespace(shape=s) for s in shapes]
    return SimpleNamespace(positives=positives, null=SimpleNamespace(shape="null"))


def test_tune_joint_selects_strictest_on_base_grid():
    def fake_eval(suite, tol, thr, seed=0, band_mode="global", ci_alpha=0.05):
        rec = 0.9 if tol >= 0.01 else 0.5          # 0.005 is too tight to recover the proxy
        return V.GridCandidate(tol, thr, [V.ProxyOutcome("offset_pair", rec, 5, True, [])], 0)
    out = V.tune_joint(_fake_suite(), evaluate=fake_eval, band_mode="global")
    assert out["expansions"] == 0
    assert out["tolerance"] == 0.01                # smallest eligible tolerance
    assert out["hold_rate_threshold"] == 0.72      # then the highest threshold
    assert out["proxy_shapes"] == ["offset_pair"]
    assert out["per_proxy"][0]["recovery"] == 0.9


def test_tune_joint_stops_after_strictest_eligible_grid_cell():
    calls = []

    def fake_eval(suite, tol, thr, seed=0, band_mode="global", ci_alpha=0.05):
        calls.append((tol, thr))
        rec = 0.9 if tol >= 0.01 and thr <= 0.7 else 0.5
        return V.GridCandidate(
            tol,
            thr,
            [V.ProxyOutcome("offset_pair", rec, 5, True, [])],
            0,
        )

    out = V.tune_joint(
        _fake_suite(),
        evaluate=fake_eval,
        thresholds=[0.6, 0.7],
        tolerances=[0.005, 0.01, 0.02],
        max_expansions=0,
    )

    assert (out["tolerance"], out["hold_rate_threshold"]) == (0.01, 0.7)
    assert calls == [(0.005, 0.7), (0.005, 0.6), (0.01, 0.7)]


def test_tune_joint_accepts_an_eligible_explicit_starting_cell_first():
    calls = []

    def fake_eval(
        suite,
        tolerance,
        threshold,
        seed=0,
        band_mode="global",
        ci_alpha=0.05,
    ):
        calls.append((tolerance, threshold))
        return V.GridCandidate(
            tolerance,
            threshold,
            [V.ProxyOutcome("offset_pair", 1.0, 5, True, [])],
            0,
        )

    result = V.tune_joint(
        _fake_suite(),
        evaluate=fake_eval,
        initial_tolerance=0.05,
        initial_threshold=0.62,
    )

    assert calls == [(0.05, 0.62)]
    assert result["tolerance"] == 0.05
    assert result["hold_rate_threshold"] == 0.62


def test_tune_joint_forwards_confidence_alpha_to_grid_evaluation():
    calls = []

    def fake_eval(
        suite,
        tol,
        thr,
        seed=0,
        band_mode="global",
        ci_alpha=0.05,
    ):
        calls.append(ci_alpha)
        return V.GridCandidate(
            tol,
            thr,
            [V.ProxyOutcome("offset_pair", 0.9, 5, True, [])],
            0,
        )

    V.tune_joint(
        _fake_suite(),
        evaluate=fake_eval,
        thresholds=[0.7],
        tolerances=[0.01],
        max_expansions=0,
        ci_alpha=0.1,
    )

    assert calls == [0.1]


def test_tune_joint_expands_grid_until_eligible():
    def eval_wide(suite, tol, thr, seed=0, band_mode="global", ci_alpha=0.05):
        rec = 0.9 if tol >= 0.1 else 0.5           # eligible only beyond the base grid max (0.05)
        return V.GridCandidate(tol, thr, [V.ProxyOutcome("offset_pair", rec, 5, True, [])], 0)
    out = V.tune_joint(_fake_suite(), evaluate=eval_wide, max_expansions=3, null_floor=0.5)
    assert out["expansions"] >= 1
    assert out["tolerance"] >= 0.1


def test_tune_joint_fails_loudly_with_per_proxy_evidence():
    def never(suite, tol, thr, seed=0, band_mode="global", ci_alpha=0.05):
        return V.GridCandidate(tol, thr, [V.ProxyOutcome("offset_pair", 0.4, 5, True, [])], 0)
    with pytest.raises(V.CalibrationGridError) as ei:
        V.tune_joint(_fake_suite(), evaluate=never, max_expansions=2)
    assert "offset_pair" in str(ei.value)          # message carries per-proxy evidence


def _boom(*a, **k):
    raise AssertionError("reached generation/preparation before regime shape validation")


def test_prepare_proxy_suite_rejects_unknown_regime_shape(monkeypatch):
    # A custom regime built directly (bypassing RegimeSpec.add) with an unknown shape must fail
    # loudly BEFORE any generation/preparation, naming the bad shape and the allowed shapes.
    monkeypatch.setattr(V, "make_inducer", _boom)
    monkeypatch.setattr(R, "generate", _boom)
    bad = R.RegimeSpec(entries=[R.ProxyEntry("unknown"), R.ProxyEntry("offset_pair")])
    with pytest.raises(ValueError) as ei:
        V.prepare_proxy_suite(bad)
    msg = str(ei.value)
    assert "unknown" in msg                        # names the offending shape
    assert "offset_pair" in msg                    # lists the allowed KNOWN_SHAPES


def test_tune_joint_memoizes_grid_cells_across_expansions():
    # Forcing two expansions must evaluate each unique (tolerance, threshold) cell exactly once --
    # growing grids re-list earlier cells, so without memoization they would be re-evaluated.
    calls = []

    def rec_eval(suite, tol, thr, seed=0, band_mode="global", ci_alpha=0.05):
        calls.append((tol, thr))
        return V.GridCandidate(tol, thr, [V.ProxyOutcome("offset_pair", 0.4, 5, True, [])], 0)

    with pytest.raises(V.CalibrationGridError):
        V.tune_joint(_fake_suite(), evaluate=rec_eval, max_expansions=2, null_floor=0.5)
    assert len(calls) == len(set(calls))           # no cell evaluated twice
    final_thresholds = {0.5, 0.54, 0.58, 0.62, 0.66, 0.72}
    final_tolerances = {0.005, 0.01, 0.02, 0.05, 0.1, 0.2}
    assert set(calls) == {(tol, thr) for thr in final_thresholds for tol in final_tolerances}


def test_live_one_sided_proxies_recover_without_wrong_sign():
    # LIVE end-to-end (real schema induction): closes the disclosed one-sided coverage gap.  Prepare
    # nonneg + nonpos proxies once, then score them through evaluate_grid_candidate and the prepared
    # path.  Each must hit its recovery target with a compact, scaled-slack-free portfolio, and
    # neither may recover the WRONG sign.  N/T kept small (3 entities, 80 snapshots).
    from autogram.config import DiscoveryConfig
    from autogram.discovery.loop import run_prepared
    regime = R.RegimeSpec(entries=[
        R.ProxyEntry("nonneg", n_entities=3, n_snapshots=80),
        R.ProxyEntry("nonpos", n_entities=3, n_snapshots=80),
    ])
    suite = V.prepare_proxy_suite(regime, seed=0)
    cand = V.evaluate_grid_candidate(suite, tolerance=0.05, hold_rate_threshold=0.62, seed=0)
    by = {p.shape: p for p in cand.proxies}
    assert set(by) == {"nonneg", "nonpos"}
    for shape in ("nonneg", "nonpos"):
        assert by[shape].recovery >= 0.8, (shape, cand.evidence())
        assert by[shape].compact and not by[shape].scaled_slack
    # zero wrong-sign recovery: score each prepared proxy's own portfolio against the OPPOSITE
    # one-sided target -> must be exactly 0.0 (no nonpos rules on nonneg data, and vice versa).
    dcfg = DiscoveryConfig(seed=0, tolerance=0.05, hold_rate_threshold=0.62, band_mode="global")
    for p in suite.positives:
        res = run_prepared(p.ds, p.G, discovery_cfg=dcfg)
        other = "nonpos" if p.shape == "nonneg" else "nonneg"
        wrong = V.score_recovery(res, {other: p.planted[p.shape]})
        assert getattr(wrong, other) == 0.0, (p.shape, "wrong-sign leak")


def test_plant_and_recover_across_noise():
    pr = V.plant_and_recover(noise_levels=(0.0, 0.02, 0.05), n_entities=4, n_snapshots=120, seed=0)
    assert set(pr.recovered) >= {
        "row_sum", "col_sum", "two_end", "self_zero",
        "offset_pair", "agg_ref_balance", "presence_pair",
    }
    assert any(any(by_noise.values()) for by_noise in pr.recovered.values())


def test_deployed_validation_covers_every_supported_proxy_shape():
    from autogram.discovery.regime import KNOWN_SHAPES

    assert set(V._PLANT_FAMILIES) == set(KNOWN_SHAPES)


def test_run_all_reports_all_null_classes(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        V,
        "proxy_tune",
        lambda seed=0: {
            "ok": True,
            "family_ok": {"ratio": True},
            "runtime_recovery": {"ratio": True},
            "tuned_threshold_tolerance": {},
            "runtime_discovery": DiscoveryConfig(),
            "runtime_search": SearchConfig(),
        },
    )
    monkeypatch.setattr(
        V,
        "portfolio_quality",
        lambda seed=0: {"ok": True},
    )
    monkeypatch.setattr(V, "null_accepted", lambda seed=0: 0)
    suite = SimpleNamespace(
        null=object(),
        temporal_null=object(),
        definition_null=object(),
    )
    monkeypatch.setattr(
        V,
        "prepare_proxy_suite",
        lambda *args, **kwargs: suite,
    )
    monkeypatch.setattr(V, "null_equalities_at", lambda *args, **kwargs: 0)
    monkeypatch.setattr(V, "null_temporal_at", lambda *args, **kwargs: 0)
    monkeypatch.setattr(V, "null_definitions_at", lambda *args, **kwargs: 0)

    report = V.run_all(seed=0)

    assert report["null_equalities_accepted"] == 0
    assert report["null_temporal_accepted"] == 0
    assert report["null_definitions_accepted"] == 0


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
        if e.rule.atom.op in ("~=", "==")
        and e.rule.atom.left == A.Ref("o0_rev")
        and e.rule.atom.right == A.Ref("o1")
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
