"""v2 guarantees-first behavior: LLM schema backends, enumeration, solver gates."""

from __future__ import annotations

import pandas as pd

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.induce import available_inducer_backends, induce_adapter, make_inducer
from autogram.discovery.loop import discover
from autogram.discovery.propose import EnumerationProposer
from autogram.dsl import ast as A
from autogram.logic.solver import equivalent, is_tautology, subsumes
from autogram.loader.loader import load_dataframe


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right))


def test_schema_induction_exposes_only_subagent_and_openai_backends(data):
    assert available_inducer_backends() == ("subagent", "openai")
    adapter = induce_adapter(data.columns, inducer=make_inducer("subagent"))
    assert "link" in adapter.binders
    assert adapter.refs_for("node")


def test_openai_backend_accepts_injected_real_responder(data):
    calls = {"n": 0}

    def responder(_prompt: str) -> str:
        calls["n"] += 1
        return make_inducer("subagent").to_json_spec(data.columns)

    adapter = induce_adapter(data.columns, inducer=make_inducer("openai", responder=responder))
    assert calls["n"] == 1
    assert "node" in adapter.binders


def test_enumeration_emits_all_v2_rule_families(grammar):
    rules = EnumerationProposer(grammar).propose(10_000, [], None)
    rendered = {r.unparse() for r in rules}

    assert "[forall node] measurement_source >= 0" in rendered
    assert "[forall node] measurement_source <= SUM(demand_row)" in rendered
    assert "[forall link] o0 != o0_rev" in rendered
    assert "[forall node] SUM(demand_row) ~= measurement_source" in rendered
    assert "[forall node] measurement_destination + SUM(demand_row) ~= measurement_source + SUM(demand_col)" in rendered
    assert "[forall link] o0_rev <|> o1" in rendered
    assert len(rendered) == len(rules)


def test_solver_decides_tautology_equivalence_and_subsumption():
    ge = _rule("node", A.Ref("measurement_source"), ">=", A.Const(0))
    le = _rule("node", A.Const(0), "<=", A.Ref("measurement_source"))
    eq = _rule("node", A.Ref("measurement_source"), "==", A.Const(0))
    taut = _rule("node", A.Ref("measurement_source"), ">=", A.Ref("measurement_source"))

    assert is_tautology(taut)
    assert equivalent(ge, le)
    assert subsumes(eq, ge)
    assert not subsumes(ge, eq)


def test_evaluator_accepts_one_sided_nonnegativity_from_hold_rate_only(dataset):
    ev = DataOnlyEvaluator(dataset, DiscoveryConfig(hold_rate_threshold=0.99, seed=0))
    res = ev.evaluate(_rule("node", A.Ref("measurement_source"), ">=", A.Const(0)))
    assert res.accepted
    assert res.hold_rate == 1.0
    assert res.hold_rate_lo > 0.99
    assert res.statistic == "hold_rate"


def test_discover_uses_enumeration_mode_and_finds_nonnegative_rules():
    data = synth.make_synthetic(n_entities=4, n_snapshots=120, noise=0.02, seed=0)
    res = discover(
        data.columns,
        data.matrix,
        discovery_cfg=DiscoveryConfig(hold_rate_threshold=0.98, seed=0),
        search_cfg=SearchConfig(proposer="enumeration", max_complexity=8, seed=0),
        name="v2",
        timestamps=data.timestamps,
    )
    assert any(e.rule.atom.op == ">=" and isinstance(e.rule.atom.right, A.Const)
               for e in res.portfolio)
    assert all(e.statistic == "hold_rate" for e in res.portfolio)


def test_dataframe_loader_reads_observed_ground_truth_not_hidden_clean():
    df = pd.DataFrame({
        "low_a_origination": [{"ground_truth": 7.0, "hidden_ground_truth": 70.0}],
        "low_a_termination": [{"ground_truth": 5.0, "hidden_ground_truth": 50.0}],
        "high_a_a": [{"ground_truth": 0.0, "hidden_ground_truth": 999.0}],
    })
    adapter = induce_adapter(list(df.columns), inducer=make_inducer("subagent"))
    ds = load_dataframe(df, adapter, "observed-only")

    assert float(ds.observed.col("low_a_origination")[0]) == 7.0
    assert float(ds.observed.col("low_a_termination")[0]) == 5.0
    assert float(ds.observed.col("high_a_a")[0]) == 0.0
