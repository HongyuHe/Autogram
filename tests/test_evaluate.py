"""Guarantees-first evaluator: solver gates + hold-rate statistic."""

from __future__ import annotations

from autogram.config import DiscoveryConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.dsl import ast as A


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right))


def test_accepts_two_end_agreement_by_hold_rate(dataset):
    # Isolates the hold-rate acceptance path; pin the flat threshold so the per-rule precision
    # policy (covered by test_per_rule_threshold_*) does not interact with this thin margin.
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9,
                                                    threshold_policy="global"))
    res = ev.evaluate(_rule("link", A.Ref("o1"), "~=", A.Ref("o0_rev")))
    assert res.accepted
    assert res.hold_rate_lo >= 0.9
    assert res.statistic == "hold_rate"
    assert res.strictness in ("exact", "soft", "loose")


def test_per_rule_threshold_raises_bar_for_low_support(dataset):
    rule = _rule("link", A.Ref("o1"), "~=", A.Ref("o0_rev"))
    # global policy uses the flat bar verbatim, so the razor-thin margin clears 0.9.
    g = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9,
                                                   threshold_policy="global")).evaluate(rule)
    assert g.threshold == 0.9 and g.accepted
    # precision-aware (per-rule policy) raises the per-rule bar for this fragile (low-support)
    # grounding, so the same near-miss hold-rate no longer clears it -- a per-law precision gate.
    p = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9,
                                                   threshold_policy="per_rule")).evaluate(rule)
    assert p.threshold > g.threshold
    assert not p.accepted


def test_global_threshold_policy_matches_flat_bar(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.62,
                                                    threshold_policy="global"))
    res = ev.evaluate(_rule("node", A.Agg("SUM", "demand_row"), "~=", A.Ref("measurement_source")))
    assert res.threshold == 0.62


def test_accepts_origination_row_sum(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    res = ev.evaluate(_rule("node", A.Agg("SUM", "demand_row"), "~=", A.Ref("measurement_source")))
    assert res.accepted


def test_rejects_empty_aggregate_family_as_degenerate(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    res = ev.evaluate(_rule("node", A.Agg("SUM", "no_such_family"), "~=", A.Const(0)))
    assert not res.accepted
    assert "degenerate" in res.reason


def test_rejects_spurious_pairing(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    res = ev.evaluate(_rule("link", A.Ref("o1"), "~=", A.Ref("o0")))
    assert not res.accepted
    assert "hold-rate" in res.reason


def test_accepts_same_family_separation(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    res = ev.evaluate(_rule("link", A.Ref("o1"), "!=", A.Ref("o1_rev")))
    assert res.accepted
    assert res.strictness == "separation"


def test_accepts_one_sided_nonnegativity(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.99))
    res = ev.evaluate(_rule("node", A.Ref("measurement_source"), ">=", A.Const(0)))
    assert res.accepted
    assert res.hold_rate == 1.0
    assert res.strictness == "one-sided"


def test_solver_rejects_tautology(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0))
    res = ev.evaluate(_rule("node", A.Ref("measurement_source"), ">=", A.Ref("measurement_source")))
    assert not res.accepted
    assert "solver-trivial" in res.reason


def test_mdl_is_not_acceptance_gate(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    res = ev.evaluate(_rule("node", A.Ref("measurement_source"), ">=", A.Const(0)))
    assert res.accepted
    assert isinstance(res.mdl_gain, float)
