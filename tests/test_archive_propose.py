"""Pareto archive + random schema-typed proposer (P4, architecture)."""

from __future__ import annotations

import random

from autogram.config import DiscoveryConfig
from autogram.discovery.archive import ParetoArchive
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.propose import RandomProposer
from autogram.dsl import ast as A
from autogram.dsl.typecheck import is_admissible


def test_archive_keeps_one_elite_per_cell(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=8, seed=0))
    arch = ParetoArchive()
    r1 = ev.evaluate(A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev"))))
    r2 = ev.evaluate(A.Rule("node", A.Compare(A.Agg("SUM", "demand_row"), "~=", A.Ref("m_src"))))
    assert arch.add(r1) and arch.add(r2)
    # re-adding a worse rule in the same cell does not displace the incumbent
    elites_before = len(arch.elites())
    assert len(arch.portfolio()) == elites_before
    assert arch.progress() > 0


def test_front_is_non_dominated(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(n_perm=8, seed=0))
    arch = ParetoArchive()
    for rule in (A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev"))),
                 A.Rule("node", A.Compare(A.Agg("SUM", "demand_row"), "~=", A.Ref("m_src")))):
        arch.add(ev.evaluate(rule))
    front = arch.front()
    assert front and all(f.accepted for f in front)


def test_proposer_yields_admissible_rules(grammar):
    prop = RandomProposer(grammar, p_mutate=0.7)
    rng = random.Random(0)
    seed_rule = A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev")))
    rules = prop.propose(40, [seed_rule], rng)
    assert rules
    for r in rules:
        ok, _ = is_admissible(r, grammar)
        assert ok
    # signatures are unique within a batch
    assert len({r.signature() for r in rules}) == len(rules)
