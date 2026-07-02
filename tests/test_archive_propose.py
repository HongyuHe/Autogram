"""Archive + exhaustive enumeration proposer."""

from __future__ import annotations

from autogram.config import DiscoveryConfig
from autogram.discovery.archive import ParetoArchive
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.propose import EnumerationProposer
from autogram.dsl import ast as A
from autogram.dsl.typecheck import is_admissible


def test_archive_keeps_solver_distinct_representatives(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    arch = ParetoArchive()
    r1 = ev.evaluate(A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev"))))
    r2 = ev.evaluate(A.Rule("node", A.Compare(A.Agg("SUM", "demand_row"), "~=", A.Ref("measurement_source"))))
    assert arch.add(r1) and arch.add(r2)
    assert len(arch.portfolio()) == 2
    assert arch.progress() > 0


def test_front_is_non_dominated(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(seed=0, hold_rate_threshold=0.9))
    arch = ParetoArchive()
    for rule in (A.Rule("link", A.Compare(A.Ref("o1"), "~=", A.Ref("o0_rev"))),
                 A.Rule("node", A.Compare(A.Agg("SUM", "demand_row"), "~=", A.Ref("measurement_source")))):
        arch.add(ev.evaluate(rule))
    front = arch.front()
    assert front and all(f.accepted for f in front)


def test_enumeration_yields_admissible_unique_rules(grammar):
    prop = EnumerationProposer(grammar)
    rules = prop.propose()
    assert rules
    for r in rules:
        ok, _ = is_admissible(r, grammar)
        assert ok
    assert len({r.signature() for r in rules}) == len(rules)
    rendered = {r.unparse() for r in rules}
    assert "[forall node] measurement_source >= 0" in rendered
    assert "[forall link] o0 != o0_rev" in rendered


def test_enumeration_includes_bounded_scale_and_add_forms(grammar):
    rules = EnumerationProposer(grammar).propose()
    atoms = [r.atom for r in rules]
    assert any(isinstance(a.left, A.Scale) or isinstance(a.right, A.Scale) for a in atoms)
    assert any(isinstance(a.left, A.Add) or isinstance(a.right, A.Add) for a in atoms)
