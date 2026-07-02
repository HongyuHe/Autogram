"""Discovery loop: exhaustive enumeration and deterministic portfolios."""

from __future__ import annotations

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.loop import discover
from autogram.dsl import ast as A


def _run(d, seed=0):
    return discover(d.columns, d.matrix,
                    discovery_cfg=DiscoveryConfig(seed=seed, hold_rate_threshold=0.9),
                    search_cfg=SearchConfig(seed=seed, max_complexity=8),
                    name="t", timestamps=d.timestamps)


def test_discovers_planted_invariants():
    d = synth.make_synthetic(n_entities=5, n_snapshots=180, noise=0.02, seed=0)
    res = _run(d)
    assert res.portfolio
    for ev in res.portfolio:
        assert ev.hold_rate_lo >= 0.9
        assert ev.statistic == "hold_rate"
    assert res.progress_history[0] > 0


def test_discovery_is_deterministic():
    d = synth.make_synthetic(n_entities=4, n_snapshots=120, noise=0.02, seed=0)
    a = _run(d)
    b = _run(d)
    assert [e.rule.signature() for e in a.portfolio] == [e.rule.signature() for e in b.portfolio]


def test_null_dataset_yields_only_structural_nonnegativity_not_equalities():
    d = synth.make_null(n_entities=4, n_snapshots=120, seed=0)
    res = _run(d)
    assert not [e for e in res.portfolio if e.rule.atom.op in ("~=", "==")]


class _FakeProposer:
    def __init__(self):
        self.calls = 0

    def propose(self, n=0, seeds=(), rng=None):
        self.calls += 1
        return [A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev")))]


def test_discover_uses_injected_proposer():
    d = synth.make_synthetic(n_entities=4, n_snapshots=120, noise=0.02, seed=0)
    fake = _FakeProposer()
    res = discover(d.columns, d.matrix,
                   discovery_cfg=DiscoveryConfig(seed=0, hold_rate_threshold=0.9),
                   search_cfg=SearchConfig(seed=0), proposer=fake,
                   name="fake", timestamps=d.timestamps)
    assert res.portfolio
    assert fake.calls == 1
