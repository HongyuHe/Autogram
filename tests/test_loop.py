"""The discovery loop: non-empty parsimonious portfolio, determinism, null FDR control."""

from __future__ import annotations

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.loop import discover
from autogram.dsl import ast as A


def _run(d, seed=0, rounds=5, proposals=100):
    return discover(d.columns, d.matrix,
                    discovery_cfg=DiscoveryConfig(n_perm=10, seed=seed),
                    search_cfg=SearchConfig(rounds=rounds, proposals_per_round=proposals,
                                            seed=seed),
                    name="t", timestamps=d.timestamps)


def test_discovers_planted_invariants():
    d = synth.make_synthetic(n_entities=5, n_snapshots=250, noise=0.02, seed=0)
    res = _run(d)
    assert res.portfolio
    # every accepted rule lifts above the null and is stable
    for ev in res.portfolio:
        assert ev.lift > 1.0 and ev.lift_percentile <= 0.05
        assert ev.support_margin > 0.0
        assert ev.stability_margin > 0.0
        assert ev.mdl_gain > 0.0
    # the progress trace is non-decreasing in its running maximum
    running = 0.0
    for p in res.progress_history:
        running = max(running, p)
    assert running > 0


def test_discovery_is_deterministic():
    d = synth.make_synthetic(n_entities=4, n_snapshots=180, noise=0.02, seed=0)
    a = _run(d, rounds=4, proposals=80)
    b = _run(d, rounds=4, proposals=80)
    assert [e.rule.signature() for e in a.portfolio] == [e.rule.signature() for e in b.portfolio]


def test_null_dataset_yields_no_invariants():
    d = synth.make_null(n_entities=5, n_snapshots=250, seed=0)
    res = _run(d)
    assert len(res.portfolio) == 0


class _FakeProposer:
    def __init__(self):
        self.calls = []

    def propose(self, n, seeds, rng):
        self.calls.append([s.signature() for s in seeds])
        return [A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev")))]


def test_discover_uses_injected_proposer_and_feeds_own_elites():
    d = synth.make_synthetic(n_entities=4, n_snapshots=160, noise=0.02, seed=0)
    fake = _FakeProposer()
    res = discover(d.columns, d.matrix,
                   discovery_cfg=DiscoveryConfig(n_perm=8, seed=0),
                   search_cfg=SearchConfig(rounds=2, proposals_per_round=1, seed=0),
                   proposer=fake, name="fake", timestamps=d.timestamps)
    assert res.portfolio
    assert len(fake.calls) == 2
    assert fake.calls[0] == []
    assert fake.calls[1]


def test_llm_portfolio_option_invokes_responder():
    d = synth.make_synthetic(n_entities=4, n_snapshots=160, noise=0.02, seed=0)
    calls = {"n": 0}

    def responder(_prompt):
        calls["n"] += 1
        return (
            '[{"binder":"link","op":"~=",'
            '"left":{"k":"Ref","role":"o1"},'
            '"right":{"k":"Ref","role":"o0_rev"}}]'
        )

    res = discover(d.columns, d.matrix,
                   discovery_cfg=DiscoveryConfig(n_perm=8, seed=0),
                   search_cfg=SearchConfig(rounds=1, proposals_per_round=2, seed=0,
                                           proposer="portfolio"),
                   llm_responder=responder, name="llm", timestamps=d.timestamps)
    assert calls["n"] >= 1
    assert res.portfolio
