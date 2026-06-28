"""E3 (poc-eval recommendations 1 & 3): per-candidate trace + result bundle.

These tests run a *tiny* scripted (no-network) learning loop on the shared Abilene fixture and
assert that (a) the loop emits a well-formed per-candidate trace, and (b) the result-bundle
builder/writers produce JSON-parseable artifacts.  They never touch a live backend or the
ground-truth catalogue.
"""

from __future__ import annotations

import json

import pytest

from autogram.config import RunConfig
from autogram.proposer import make_proposer
from autogram.search.loop import learn
from autogram.search.recall import score_recall
from autogram.search.result_bundle import (
    build_result_bundle,
    git_provenance,
    write_result_json,
    write_trace_jsonl,
)

_TRACE_KEYS = {
    "round", "phase", "iter", "island", "origin", "parent_sig", "improved",
    "sig", "rule", "binder", "op", "complexity", "verdict", "accepted",
    "combined_score", "eps", "kappa_hat", "support", "lift", "delta",
}


def _tiny_run(abilene):
    """A small but real scripted run; returns (RunConfig, RunResult)."""
    rc = RunConfig(dataset="abilene")
    rc.search.iterations = 40
    rc.search.bootstrap_random = 6
    rc.seed = 0
    rc.reseed()
    res = learn(abilene, rc, make_proposer("scripted"))
    return rc, res


def test_trace_is_populated_and_well_formed(abilene):
    _, res = _tiny_run(abilene)
    assert res.trace, "the loop should emit a non-empty trace"
    phases = {row["phase"] for row in res.trace}
    assert phases == {"seed", "search"}
    for row in res.trace:
        assert _TRACE_KEYS.issubset(row.keys())
        assert isinstance(row["improved"], bool)
        # every row must be JSON-serialisable (the writer relies on this)
        json.dumps(row)


def test_seed_origins_include_grammar_and_anti_seeds(abilene):
    # anti_seeds defaults on, so the separation niche must be represented in the seed phase.
    _, res = _tiny_run(abilene)
    seed_origins = {row["origin"] for row in res.trace if row["phase"] == "seed"}
    assert "anti_seed" in seed_origins
    # the scripted proposer enumerates name-semantic forms, surfaced as "proposer" seeds.
    assert "proposer" in seed_origins


def test_search_rows_have_parent_lineage_for_mutations(abilene):
    _, res = _tiny_run(abilene)
    search = [r for r in res.trace if r["phase"] == "search"]
    assert search, "a 40-iteration run should produce search rows"
    muts = [r for r in search if r["origin"] in ("mutation", "seed_mutation")]
    # mutation rows must name a concrete parent signature (lineage auditing).
    assert all(r["parent_sig"] for r in muts)
    # random rows carry no parent.
    rands = [r for r in search if r["origin"] == "random"]
    assert all(r["parent_sig"] is None for r in rands)


def test_inadmissible_child_row_tolerates_missing_result(abilene):
    # Not all variations yield an admissible child; such rows record verdict INADMISSIBLE
    # with null metrics and must still be JSON-serialisable.
    _, res = _tiny_run(abilene)
    inadmissible = [r for r in res.trace if r["verdict"] == "INADMISSIBLE"]
    for row in inadmissible:
        assert row["sig"] is None and row["combined_score"] is None
        json.dumps(row)


def test_write_trace_jsonl_roundtrips(abilene, tmp_path):
    _, res = _tiny_run(abilene)
    path = tmp_path / "trace_abilene.jsonl"
    n = write_trace_jsonl(str(path), res.trace)
    assert n == len(res.trace)
    lines = path.read_text(encoding="ascii").splitlines()
    assert len(lines) == n
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["phase"] == "seed"


def test_result_bundle_is_serialisable_and_complete(abilene, tmp_path):
    rc, res = _tiny_run(abilene)
    rep = score_recall(res.portfolio)
    bundle = build_result_bundle(
        rc, "data/crosscheck-samples/abilene_sample_1000.pkl", abilene, res, rep,
        used_real_subagent=None, proposer_notes="",
    )
    # top-level contract
    for key in ("schema", "dataset", "knobs", "counts", "recall", "portfolio",
                "provenance", "seed", "proposer"):
        assert key in bundle
    assert bundle["schema"] == "autogram.result/v1"
    assert bundle["dataset"] == "abilene"
    assert bundle["knobs"]["search"]["anti_seeds"] is True
    assert bundle["counts"]["portfolio_size"] == len(res.portfolio)
    assert bundle["recall"]["n_targets"] == rep.n_targets
    assert len(bundle["portfolio"]) == len(res.portfolio)
    # the whole bundle must round-trip through JSON
    path = tmp_path / "result_abilene.json"
    write_result_json(str(path), bundle)
    reloaded = json.loads(path.read_text(encoding="ascii"))
    assert reloaded["schema"] == "autogram.result/v1"


def test_git_provenance_shape():
    prov = git_provenance()
    assert set(prov.keys()) == {"git_commit", "git_dirty"}
