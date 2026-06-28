"""Data-only evaluator: accept planted invariants, reject tautologies/spurious (P3, P5)."""

from __future__ import annotations

from autogram.config import DiscoveryConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.dsl import ast as A


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right))


def test_accepts_two_end_agreement(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=12, seed=0))
    res = ev.evaluate(_rule("link", A.Ref("o1"), "~=", A.Ref("o0_rev")))
    assert res.accepted
    assert res.lift > 5.0
    assert res.support_margin > 0.0
    assert res.stability_margin > 0.0
    assert res.strictness in ("exact", "soft")


def test_accepts_origination_row_sum(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=12, seed=0))
    res = ev.evaluate(_rule("node", A.Agg("SUM", "demand_row"), "~=", A.Ref("m_src")))
    assert res.accepted


def test_rejects_spurious_pairing(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=12, seed=0))
    res = ev.evaluate(_rule("link", A.Ref("o1"), "~=", A.Ref("o0")))
    assert not res.accepted
    assert "lift" in res.reason or "stable" in res.reason or "plateau" in res.reason


def test_rejects_anti_invariant_for_acceptance(dataset):
    # separation forms are admissible but the conservative data-only grader does not certify them
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=12, seed=0))
    res = ev.evaluate(_rule("link", A.Ref("o1"), "!=", A.Ref("o1_rev")))
    assert not res.accepted


def test_strictness_is_descriptive_label(dataset):
    # the strictness is a LABEL read off the fitted eps + operator, not a gate: a literally-zero
    # column compared to 0 has eps ~ 0 -> "exact" regardless of whether it is finally accepted.
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=12, seed=0))
    res = ev.evaluate(_rule("node", A.Ref("demand_self"), "==", A.Const(0)))
    assert res.strictness == "exact"
    assert res.eps == 0.0


def test_discovery_config_has_no_fixed_acceptance_thresholds():
    cfg = DiscoveryConfig()
    for name in (
        "stability_tol", "plateau_lo", "min_points", "min_bindings",
        "exact_label_eps", "soft_label_eps",
    ):
        assert not hasattr(cfg, name)


def test_rejects_negative_mdl_even_when_other_proxy_gates_pass(dataset):
    # This seed-0 edge rule previously entered the delivered portfolio despite increasing
    # description length. It may have support/lift/stability, but parsimony is a required proxy.
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=12, seed=0))
    res = ev.evaluate(_rule("node", A.Scale(0.5, A.Ref("m_src")), "==",
                            A.Agg("SUM", "fam_from")))
    assert res.mdl_gain < 0.0
    assert not res.accepted
    assert "parsimonious" in res.reason
