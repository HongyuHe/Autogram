"""E4 (poc-eval recommendation 7): run/score split + multi-seed bench + provenance.

These tests exercise the building blocks behind the ``score`` and ``bench`` subcommands using
the offline *scripted* proposer and the shared Abilene fixture.  They never touch a live
backend.  The central guarantee under test is that grading a *saved, unscored* bundle (the
``score`` path) reproduces the verdicts -- and hence the recall -- of the original ``run``,
because each portfolio entry carries a self-contained serialised AST (``rule_dict``) and the
bundle records the exact inner-loop evaluator knobs.
"""

from __future__ import annotations

import json

from autogram.cli import _apply_section, load_config
from autogram.config import RunConfig
from autogram.dsl.parser import rule_from_dict, rule_to_dict
from autogram.evaluator.evaluator import evaluate
from autogram.proposer import make_proposer
from autogram.search.loop import learn
from autogram.search.recall import TESTABLE_TARGETS, score_recall
from autogram.search.result_bundle import (
    build_bench_bundle,
    build_result_bundle,
    recall_record,
    write_result_json,
)


def _tiny_run(dataset_obj, seed=0):
    """A small but real scripted run; returns (RunConfig, RunResult)."""
    rc = RunConfig(dataset=dataset_obj.name)
    rc.search.iterations = 40
    rc.search.bootstrap_random = 6
    rc.seed = seed
    rc.reseed()
    res = learn(dataset_obj, rc, make_proposer("scripted"))
    return rc, res


# --------------------------------------------------------------------------- run --no-score

def test_unscored_bundle_has_null_recall_and_self_contained_rules(abilene, tmp_path):
    rc, res = _tiny_run(abilene)
    bundle = build_result_bundle(
        rc, "data/crosscheck-samples/abilene_sample_1000.pkl", abilene, res, rep=None,
        used_real_subagent=None, proposer_notes="",
    )
    # an unscored (--no-score) bundle never imported the oracle grader.
    assert bundle["recall"] is None
    # every portfolio entry must carry a serialised AST so it can be re-scored later.
    assert bundle["portfolio"], "scripted run should learn a non-empty portfolio"
    for entry in bundle["portfolio"]:
        assert "rule_dict" in entry and isinstance(entry["rule_dict"], dict)
    # whole bundle round-trips through JSON.
    path = tmp_path / "result_abilene.json"
    write_result_json(str(path), bundle)
    assert json.loads(path.read_text(encoding="ascii"))["recall"] is None


# --------------------------------------------------------------------------- score round-trip

def test_score_path_reproduces_run_verdicts_and_recall(abilene, tmp_path):
    """The core E4 guarantee: reconstructing rules from the bundle and re-evaluating with the
    bundle's recorded knobs reproduces the original run's per-rule verdicts and recall."""
    rc, res = _tiny_run(abilene)
    rep_original = score_recall(res.portfolio)

    # Write an UNSCORED bundle (as `run --no-score` would).
    bundle = build_result_bundle(
        rc, "data/crosscheck-samples/abilene_sample_1000.pkl", abilene, res, rep=None,
    )
    path = tmp_path / "result_abilene.json"
    write_result_json(str(path), bundle)
    reloaded = json.loads(path.read_text(encoding="ascii"))

    # Reconstruct exactly as `cmd_score` does: fresh config + bundle's eval knobs.
    rc2 = load_config(None, "abilene")
    _apply_section(rc2.eval, reloaded["knobs"]["eval"])
    rescored = [evaluate(rule_from_dict(e["rule_dict"]), abilene, rc2.eval)
                for e in reloaded["portfolio"]]

    # Per-rule verdicts must match the original portfolio one-for-one.
    assert [r.summary() for r in rescored] == [r.summary() for r in res.portfolio]

    rep_rescored = score_recall(rescored)
    assert rep_rescored.strict_recall == rep_original.strict_recall
    assert rep_rescored.recall == rep_original.recall


def test_rule_dict_roundtrip_is_faithful(abilene):
    rc, res = _tiny_run(abilene)
    for r in res.portfolio:
        rebuilt = rule_from_dict(rule_to_dict(r.rule))
        again = evaluate(rebuilt, abilene, rc.eval)
        assert again.summary() == r.summary()


def test_recall_record_graft_shape(abilene):
    _, res = _tiny_run(abilene)
    rep = score_recall(res.portfolio)
    rec = recall_record(rep)
    for key in ("n_targets", "n_recovered", "n_full", "strict_recall", "recall", "matches"):
        assert key in rec
    assert rec["n_targets"] == rep.n_targets
    assert isinstance(rec["matches"], list)


# --------------------------------------------------------------------------- bench aggregation

def test_build_bench_bundle_aggregates_correctly(tmp_path):
    targets = [t.tid for t in TESTABLE_TARGETS]
    # Two synthetic seeds with known recalls so the aggregates are checkable by hand.
    per_seed = [
        {"seed": 0, "strict_recall": 1.0, "recall": 1.0, "hits": targets},
        {"seed": 1, "strict_recall": 0.5, "recall": 0.75, "hits": targets[:1]},
    ]
    bundle = build_bench_bundle(
        "abilene", "data/x.pkl", [0, 1], per_seed,
        proposer="scripted", iters=40, targets=targets)
    assert bundle["schema"] == "autogram.bench/v1"
    assert bundle["n_runs"] == 2
    sr = bundle["strict_recall"]
    assert sr["mean"] == 0.75 and sr["worst"] == 0.5 and sr["best"] == 1.0
    # variance of {1.0, 0.5} about mean 0.75 is 0.0625.
    assert abs(sr["var"] - 0.0625) < 1e-9
    assert bundle["soft_recall"]["mean"] == 0.875
    htr = bundle["per_target_hit_rate"]
    # the first target was hit by both seeds; the rest only by seed 0.
    assert htr[targets[0]] == 1.0
    assert htr[targets[1]] == 0.5
    # provenance + JSON round-trip.
    assert set(bundle["provenance"]).issuperset({"git_commit", "git_dirty", "timestamp_utc"})
    path = tmp_path / "bench_abilene.json"
    write_result_json(str(path), bundle)
    assert json.loads(path.read_text(encoding="ascii"))["n_runs"] == 2


def test_bench_endtoend_two_seeds_is_wellformed(abilene):
    """A light integration check: two real scripted seeds aggregate into a valid bundle."""
    targets = [t.tid for t in TESTABLE_TARGETS]
    per_seed = []
    for s in (0, 1):
        rc, res = _tiny_run(abilene, seed=s)
        rep = score_recall(res.portfolio)
        per_seed.append({
            "seed": s,
            "strict_recall": round(rep.strict_recall, 6),
            "recall": round(rep.recall, 6),
            "hits": [m.tid for m in rep.matches if m.status == "FULL"],
        })
    bundle = build_bench_bundle(
        "abilene", "data/crosscheck-samples/abilene_sample_1000.pkl", [0, 1], per_seed,
        proposer="scripted", iters=40, targets=targets)
    assert bundle["n_runs"] == 2
    assert 0.0 <= bundle["strict_recall"]["mean"] <= 1.0
    assert set(bundle["per_target_hit_rate"]) == set(targets)
