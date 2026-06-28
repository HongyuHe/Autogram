"""Command-line entry point for the P3 invariant-learning engine.

Two subcommands:

* ``run``          -- load a config, run the three-level learning loop on a dataset, then
  print the assembled invariant portfolio, the recall scorecard against the known catalogue,
  and the exact knob settings used (for reproducibility).
* ``dump-prompt``  -- render *only* the leakage-safe proposer prompt for a dataset and write
  it to a unique ``<work-dir>/prompts/prompt_*`` directory.  This supports the leakage-free
  headline workflow: an externally-spawned, isolated subagent receives this prompt (and
  nothing else -- no repo / no catalogue), returns JSON to
  ``<work-dir>/subagent_response_<dataset>.json``, and a subsequent ``run --proposer
  subagent`` consumes that reply.

All console output is ASCII (the Windows console is cp1252).

Examples
--------
    uv run autogram run --dataset abilene --proposer scripted --iters 100 --seed 0
    uv run autogram run --config configs/geant.yaml
    uv run autogram dump-prompt --dataset abilene
    uv run autogram run --dataset abilene --proposer subagent  # reads response file
    uv run autogram clean                                      # remove generated run artifacts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from dataclasses import fields as dc_fields, replace
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .config import EvalConfig, GrammarConfig, RunConfig, SearchConfig
from .loader.loader import load_dataset
from .dsl.parser import rule_from_dict
from .evaluator.evaluator import evaluate
from .proposer import build_context, make_proposer
from .proposer.base import render_proposal_prompt
from .search.loop import learn, start_grammar, ceiling_grammar
from .search.recall import TESTABLE_TARGETS, format_report, score_recall
from .search.result_bundle import (
    build_bench_bundle,
    build_result_bundle,
    recall_record,
    write_result_json,
    write_rules_file,
    write_trace_jsonl,
)

_DEFAULT_PATHS = {
    "abilene": "data/crosscheck-samples/abilene_sample_1000.pkl",
    "geant": "data/crosscheck-samples/geant_sample_1000.pkl",
}


# --------------------------------------------------------------------------- config loading

def _apply_section(section, values: dict) -> None:
    known = {f.name for f in dc_fields(section)}
    for k, v in values.items():
        if k in known:
            setattr(section, k, v)
        else:
            print(f"  [warn] unknown config key ignored: {k}", file=sys.stderr)


def load_config(path: Optional[str], dataset: Optional[str]) -> RunConfig:
    """Build a :class:`RunConfig` from an optional YAML file (CLI flags override later)."""
    rc = RunConfig()
    if path:
        import yaml  # local import so the package has no hard yaml dependency at import time
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        for key in ("dataset", "data_path", "seed"):
            if key in raw:
                setattr(rc, key, raw[key])
        if "eval" in raw:
            _apply_section(rc.eval, raw["eval"])
        if "search" in raw:
            _apply_section(rc.search, raw["search"])
        if "grammar" in raw:
            _apply_section(rc.grammar, raw["grammar"])
    if dataset:
        rc.dataset = dataset
    return rc


def _resolve_path(rc: RunConfig) -> str:
    if rc.data_path:
        return rc.data_path
    if rc.dataset in _DEFAULT_PATHS:
        return _DEFAULT_PATHS[rc.dataset]
    raise SystemExit(f"no data path for dataset {rc.dataset!r}; pass --data-path")


def _slug(value: object) -> str:
    """Return a conservative filesystem slug for run-directory names."""
    text = str(value).strip().lower()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    return slug or "unknown"


def _artifact_dir(work_dir: str, category: str, stem: str, dataset: Optional[str] = None,
                  seed: Optional[int] = None) -> str:
    """Create a unique artifact directory under ``work_dir`` so runs never overwrite."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [_slug(stem)]
    if dataset:
        parts.append(_slug(dataset))
    if seed is not None:
        parts.append(f"seed{seed}")
    parts.extend([stamp, uuid4().hex[:8]])
    path = os.path.join(work_dir, category, "_".join(parts))
    os.makedirs(path, exist_ok=False)
    return path


def _latest_result_path(work_dir: str, dataset: str) -> str:
    """Resolve the newest result bundle for ``score --dataset``, including run-scoped paths."""
    filename = f"result_{dataset}.json"
    legacy = os.path.join(work_dir, filename)
    candidates = []
    if os.path.exists(legacy):
        candidates.append(legacy)
    candidates.extend(glob.glob(os.path.join(work_dir, "**", filename), recursive=True))
    seen, uniq = set(), []
    for path in candidates:
        n = os.path.normpath(path)
        if n not in seen and os.path.isfile(n):
            seen.add(n)
            uniq.append(n)
    if not uniq:
        return legacy
    return max(uniq, key=os.path.getmtime)


# --------------------------------------------------------------------------------- commands

def cmd_run(args: argparse.Namespace) -> int:
    rc = load_config(args.config, args.dataset)
    # CLI flag overrides (only when explicitly supplied).
    if args.proposer is not None:
        rc.grammar.proposer = args.proposer
    if args.iters is not None:
        rc.search.iterations = args.iters
    if args.rounds is not None:
        rc.grammar.rounds = args.rounds
    if getattr(args, "start", None) is not None:
        rc.grammar.start = args.start
    if args.seed is not None:
        rc.seed = args.seed
    if args.data_path is not None:
        rc.data_path = args.data_path
    if getattr(args, "deployed", False):
        rc.eval.deployed = True
    if getattr(args, "rel_noise", None) is not None:
        rc.eval.rel_noise = args.rel_noise
    rc.reseed()

    path = _resolve_path(rc)
    ds = load_dataset(path, name=rc.dataset)

    proposer = make_proposer(
        rc.grammar.proposer, work_dir=args.work_dir, dataset=rc.dataset,
        model=args.model, api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"== P3 invariant learner :: dataset={rc.dataset} proposer={rc.grammar.proposer} ==")
    print(f"   data        : {path}  ({ds.n_snapshots} snapshots, {len(ds.observed.names)} cols)")
    res = learn(ds, rc, proposer)

    print()
    print(f"Learned portfolio ({len(res.portfolio)} invariants; "
          f"{res.n_evaluated} candidates evaluated, {res.n_unique} unique):")
    print("-" * 78)
    for r in res.portfolio:
        print("  " + r.summary())
    print("-" * 78)

    print()
    do_score = not getattr(args, "no_score", False)
    rep = None
    if do_score:
        rep = score_recall(res.portfolio)
        print(format_report(rep))
    else:
        print("Scoring skipped (--no-score): the portfolio was learned without grading it")
        print("  against the ground-truth catalogue (oracle-free).  Grade it later with:")
        print(f"    uv run autogram score --dataset {rc.dataset} --work-dir {args.work_dir}")

    used_real = getattr(proposer, "used_real_subagent", None)
    if rc.grammar.proposer == "subagent":
        print()
        print(f"  subagent: used_real_subagent={used_real}  "
              f"(prompt -> {os.path.join(args.work_dir, f'subagent_prompt_{rc.dataset}.txt')})")
        if not used_real:
            print("  subagent: NO external reply found; portfolio came from blind random "
                  "search only.\n           Run 'dump-prompt', answer with an isolated "
                  "subagent, then re-run.")

    notes = getattr(proposer, "last_notes", "")
    if notes:
        print()
        print(f"  proposer feedback: {notes}")

    # Outer-loop grammar widening: report every extension the proposer actually applied
    # (empty in the default start="full", where start already equals the ceiling).
    if res.extensions_applied:
        print()
        print(f"  grammar extensions applied ({len(res.extensions_applied)} round(s)):")
        for rec in res.extensions_applied:
            adds = []
            for key, label in (("enabled_binders", "binders"),
                                ("enabled_ref_roles", "ref_roles"),
                                ("enabled_fam_roles", "fam_roles"),
                                ("enabled_ops", "ops"),
                                ("enabled_agg_kinds", "agg_kinds")):
                if rec.get(key):
                    adds.append(f"{label}+{rec[key]}")
            adds.append(f"max_complexity={rec['max_complexity']}")
            adds.append(f"max_add_arity={rec['max_add_arity']}")
            note = (rec.get("note") or "").strip()
            tail = f"  ({note})" if note else ""
            print(f"    round {rec['round']}: " + "  ".join(adds) + tail)

    print()
    print("Knob settings (reproducibility):")
    print(f"  seed={rc.seed}  proposer={rc.grammar.proposer}  rounds={rc.grammar.rounds}  "
          f"start={getattr(rc.grammar, 'start', 'full')}  "
          f"max_complexity={rc.grammar.max_complexity}")
    print(f"  iterations={rc.search.iterations}  islands={rc.search.islands}  "
          f"thompson={rc.search.thompson}  bootstrap_random={rc.search.bootstrap_random}  "
          f"seed_from_grammar={rc.search.seed_from_grammar}  anti_seeds={rc.search.anti_seeds}")
    print(f"  target_coverage={rc.eval.target_coverage}  gate_k={rc.eval.gate_k}  "
          f"eps_exact={rc.eval.eps_exact}  eps_max={rc.eval.eps_max}  lift_min={rc.eval.lift_min}")

    # ---- human-readable rule file + machine-readable run artifacts --------------------
    want_artifacts = not getattr(args, "no_artifacts", False)
    want_rules = not getattr(args, "no_rules", False)
    if want_artifacts or want_rules:
        bundle = build_result_bundle(
            rc, path, ds, res, rep,
            used_real_subagent=used_real,
            proposer_notes=notes,
        )
        if want_rules:
            rules_path = write_rules_file(getattr(args, "rules_dir", "rules"), bundle)
            n_rules = len(bundle.get("portfolio", []) or [])
            print()
            print("Learned rules:")
            print(f"  rules  -> {rules_path}  ({n_rules} invariant(s))")
        if want_artifacts:
            out_dir = _artifact_dir(args.work_dir, "runs", "run", rc.dataset, rc.seed)
            trace_path = os.path.join(out_dir, f"trace_{rc.dataset}.jsonl")
            result_path = os.path.join(out_dir, f"result_{rc.dataset}.json")
            n_rows = write_trace_jsonl(trace_path, res.trace)
            write_result_json(result_path, bundle)
            print()
            print("Run artifacts:")
            print(f"  trace  -> {trace_path}  ({n_rows} candidate records)")
            recall_state = "recall=null (unscored)" if rep is None else "recall + provenance"
            print(f"  result -> {result_path}  (knobs + portfolio + {recall_state})")
    return 0


def cmd_dump_prompt(args: argparse.Namespace) -> int:
    rc = load_config(args.config, args.dataset)
    if args.seed is not None:
        rc.seed = args.seed
    if args.data_path is not None:
        rc.data_path = args.data_path
    path = _resolve_path(rc)
    ds = load_dataset(path, name=rc.dataset)
    G = start_grammar(rc.grammar)
    ceiling = ceiling_grammar(rc.grammar)
    ctx = build_context(ds, G, ceiling=ceiling)
    prompt = render_proposal_prompt(ctx)        # raises if anything leaks

    out_dir = _artifact_dir(args.work_dir, "prompts", "prompt", rc.dataset, rc.seed)
    out = os.path.join(out_dir, f"subagent_prompt_{rc.dataset}.txt")
    with open(out, "w", encoding="ascii", errors="replace") as fh:
        fh.write(prompt)
    print(f"wrote leakage-safe proposer prompt -> {out}  ({len(prompt)} chars)")
    print("Hand this file (and nothing else) to an isolated subagent; save its JSON reply to")
    print(f"  {os.path.join(out_dir, f'subagent_response_{rc.dataset}.json')}")
    print("Then run with the same prompt directory as --work-dir so the response is consumed:")
    run_target = f"--config {args.config}" if args.config else f"--dataset {rc.dataset}"
    print(f"  uv run autogram run {run_target} --work-dir {out_dir}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Grade a previously-learned portfolio against the catalogue (run/score split).

    Scoring half of poc-eval recommendation 8.  Loads a ``result_<dataset>.json`` bundle from
    ``run`` (typically ``run --no-score``), rebuilds every portfolio rule from its serialised
    AST (``rule_dict``), *re-evaluates* each through the same evaluator the learner used, then
    runs the recall grader.  Because learning and grading are separate invocations, an
    unscored bundle is auditable evidence that the portfolio was produced without the oracle.
    """
    if args.result is not None:
        result_path = args.result
    elif args.dataset is not None:
        result_path = _latest_result_path(args.work_dir, args.dataset)
    else:
        print("score: provide --dataset or --result", file=sys.stderr)
        return 2
    if not os.path.exists(result_path):
        print(f"score: result bundle not found: {result_path}", file=sys.stderr)
        return 2
    with open(result_path, "r", encoding="ascii", errors="replace") as fh:
        bundle = json.load(fh)
    dataset = bundle.get("dataset", args.dataset)

    data_path = args.data_path or bundle.get("data_path")
    if not data_path or not os.path.exists(data_path):
        print(f"score: dataset file not found: {data_path}", file=sys.stderr)
        return 2
    ds = load_dataset(data_path, name=dataset)

    # Reconstruct the inner-loop evaluator config so verdicts reproduce exactly.
    rc = load_config(None, dataset)
    _apply_section(rc.eval, bundle.get("knobs", {}).get("eval", {}))

    results = []
    n_skipped = 0
    for entry in bundle.get("portfolio", []):
        rd = entry.get("rule_dict")
        if rd is None:                       # pre-E4 bundle without serialised AST
            n_skipped += 1
            continue
        results.append(evaluate(rule_from_dict(rd), ds, rc.eval))

    extra = f", {n_skipped} unscorable" if n_skipped else ""
    print(f"== P3 score :: dataset={dataset}  ({len(results)} rules re-evaluated{extra}) ==")
    print(f"   bundle : {result_path}")
    print(f"   data   : {data_path}  ({ds.n_snapshots} snapshots, {len(ds.observed.names)} cols)")
    print()
    print("Re-evaluated portfolio:")
    print("-" * 78)
    for r in results:
        print("  " + r.summary())
    print("-" * 78)
    print()
    rep = score_recall(results)
    print(format_report(rep))

    # Graft the scorecard onto a copy of the bundle; leave the unscored original intact.
    bundle["recall"] = recall_record(rep)
    bundle["scored_provenance"] = {
        "scored_from": os.path.basename(result_path),
        "n_rules_scored": len(results),
        "n_unscorable": n_skipped,
    }
    out_path = args.out or os.path.join(os.path.dirname(result_path),
                                        f"result_{dataset}.scored.json")
    write_result_json(out_path, bundle)
    print()
    print(f"Scored bundle -> {out_path}  (recall grafted; original left unscored)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Multi-seed recall benchmark (poc-eval recommendation 7).

    Runs the learner once per seed, scores each run, and reports per-target hit-rate plus
    mean / variance / worst-case strict recall -- so stochastic search behaviour is summarised
    rather than reported from a single (possibly lucky) seed.

    Caveat: the ``subagent`` backend reads a *fixed* response file, so multi-seed varies only
    the search RNG (bootstrap / mutation / Thompson) -- which is the stochastic part here.
    Varying the proposer *sample* needs multiple response files or a live backend; this is
    recorded in the bench output and bundle.
    """
    seeds = [int(t) for t in str(args.seeds).split(",") if t.strip()]
    if not seeds:
        print("bench: --seeds must list >=1 integer (e.g. --seeds 0,1,2)", file=sys.stderr)
        return 2

    def _mk_cfg(seed: int) -> RunConfig:
        rc = load_config(args.config, args.dataset)
        if args.proposer is not None:
            rc.grammar.proposer = args.proposer
        if args.iters is not None:
            rc.search.iterations = args.iters
        if args.data_path is not None:
            rc.data_path = args.data_path
        rc.seed = seed
        rc.reseed()
        return rc

    base = _mk_cfg(seeds[0])
    dataset = base.dataset
    path = _resolve_path(base)
    ds = load_dataset(path, name=dataset)
    target_ids = [t.tid for t in TESTABLE_TARGETS]

    print(f"== P3 bench :: dataset={dataset}  proposer={base.grammar.proposer}  "
          f"seeds={seeds}  iters={base.search.iterations} ==")
    print(f"   data : {path}  ({ds.n_snapshots} snapshots, {len(ds.observed.names)} cols)")
    if base.grammar.proposer == "subagent":
        print("   note : subagent reply is fixed; seeds vary only the search RNG.")
    print("-" * 78)

    per_seed = []
    for s in seeds:
        rc = _mk_cfg(s)
        proposer = make_proposer(
            rc.grammar.proposer, work_dir=args.work_dir, dataset=rc.dataset,
            model=args.model, api_key=os.environ.get("OPENAI_API_KEY"))
        res = learn(ds, rc, proposer)
        rep = score_recall(res.portfolio)
        per_seed.append({
            "seed": s,
            "strict_recall": round(rep.strict_recall, 6),
            "recall": round(rep.recall, 6),
            "n_full": rep.n_full,
            "n_recovered": rep.n_recovered,
            "portfolio_size": len(res.portfolio),
            "hits": [m.tid for m in rep.matches if m.status == "FULL"],
            "recovered": [m.tid for m in rep.matches if m.status in ("FULL", "PARTIAL")],
        })
        print(f"  seed {s:>3} : strict={rep.strict_recall:6.2%}  soft={rep.recall:6.2%}  "
              f"full={rep.n_full}/{rep.n_targets}  portfolio={len(res.portfolio)}")
    print("-" * 78)

    bundle = build_bench_bundle(
        dataset, path, seeds, per_seed,
        proposer=base.grammar.proposer, iters=base.search.iterations, targets=target_ids)
    sr = bundle["strict_recall"]
    print(f"  strict recall over {len(seeds)} seed(s): mean={sr['mean']:.4f}  "
          f"var={sr['var']:.4f}  worst={sr['worst']:.4f}  best={sr['best']:.4f}")
    print("  per-target strict hit-rate:")
    htr = bundle["per_target_hit_rate"]
    for t in target_ids:
        print(f"    {t:<4} {htr[t]:6.2%}")

    if not getattr(args, "no_artifacts", False):
        out_dir = _artifact_dir(args.work_dir, "benches", "bench", dataset)
        out_path = os.path.join(out_dir, f"bench_{dataset}.json")
        write_result_json(out_path, bundle)
        print()
        print(f"Bench artifact -> {out_path}  (per-seed + aggregate + provenance)")
    return 0


# ------------------------------------------------------------------- deployed-mode comparison

def _verdict_by_target(results) -> dict:
    """Map each testable target id -> the verdict of its best form-matching portfolio rule.

    Reuses the recall grader's form matcher so the verdict column lines up exactly with the
    FULL/PARTIAL/MISSED status it reports.  ``MISSED`` if no rule matches the target's form.
    """
    from .search.recall import _matches_form, _rule_fingerprint
    fps = [(_rule_fingerprint(r), r) for r in results]
    out = {}
    for t in TESTABLE_TARGETS:
        verdict = "MISSED"
        for (binder, kind, keys), r in fps:
            if _matches_form(t, binder, kind, keys):
                verdict = r.verdict.value
                break
        out[t.tid] = verdict
    return out


def cmd_compare(args: argparse.Namespace) -> int:
    """Oracle-gated vs observed-only (deployed) comparison -- the E5 honesty check.

    The default evaluator gate measures the injected noise *exactly* from the hidden clean
    frame (Sec. 5.5).  At deployment that frame is gone, so the engine must estimate noise from
    a modelled relative level ``eta`` (``--rel-noise``).  This command quantifies what that
    costs by reporting three gradings on the *same* dataset:

      (1) oracle  learn + grade               -- the development-time reference (uses ds.clean);
      (2) oracle-learned forms, observed re-grade -- *controlled*: identical forms, only the
                                                 gate changes, isolating the gate's effect;
      (3) observed learn + grade              -- full end-to-end deployment (never reads ds.clean).

    The expected, honest finding: sub-noise soft-structural laws (I5/I6, ~1.9% < ~2% noise)
    can no longer be separated from noise, so they weaken EXACT -> the *strict* recall drops
    while plain recall (form recovery) is unaffected.
    """
    def _mk(deployed: bool) -> RunConfig:
        rc = load_config(args.config, args.dataset)
        if args.proposer is not None:
            rc.grammar.proposer = args.proposer
        if args.iters is not None:
            rc.search.iterations = args.iters
        if args.data_path is not None:
            rc.data_path = args.data_path
        if args.rel_noise is not None:
            rc.eval.rel_noise = args.rel_noise
        rc.eval.deployed = deployed
        rc.reseed()
        return rc

    rc = _mk(False)
    dataset = rc.dataset
    path = _resolve_path(rc)
    ds = load_dataset(path, name=dataset)

    def _proposer():
        return make_proposer(
            rc.grammar.proposer, work_dir=args.work_dir, dataset=dataset,
            model=args.model, api_key=os.environ.get("OPENAI_API_KEY"))

    dep_eval = replace(rc.eval, deployed=True)

    print(f"== P3 compare :: dataset={dataset}  proposer={rc.grammar.proposer}  "
          f"iters={rc.search.iterations}  eta={rc.eval.rel_noise} ==")
    print(f"   data : {path}  ({ds.n_snapshots} snapshots, {len(ds.observed.names)} cols)")
    print(f"   gate : oracle uses clean frame; deployed uses eta={rc.eval.rel_noise} "
          f"(gate_k={rc.eval.gate_k})")
    print("-" * 78)

    # (1) oracle learn + grade
    res_o = learn(ds, rc, _proposer())
    rep_o = score_recall(res_o.portfolio)
    v_oracle = _verdict_by_target(res_o.portfolio)

    # (2) controlled re-grade of the SAME forms under the observed-only gate
    res_regraded = [evaluate(r.rule, ds, dep_eval) for r in res_o.portfolio]
    rep_r = score_recall(res_regraded)
    v_regraded = _verdict_by_target(res_regraded)

    # (3) full end-to-end observed-only learn + grade
    rc_dep = _mk(True)
    res_d = learn(ds, rc_dep, _proposer())
    rep_d = score_recall(res_d.portfolio)
    v_deployed = _verdict_by_target(res_d.portfolio)

    print("Per-target verdict (form-matched rule); columns: (1) oracle  (2) obs re-grade  "
          "(3) obs learn")
    print(f"  {'tid':<5}{'expected':<30}{'oracle':<16}{'regrade':<16}{'deployed':<16}")
    for t in TESTABLE_TARGETS:
        exp = "|".join(v.value if hasattr(v, "value") else str(v) for v in t.expect) \
            if getattr(t, "expect", None) else ("single>=" if getattr(t, "wildcard_single_ge", False) else "-")
        print(f"  {t.tid:<5}{exp:<30}{v_oracle[t.tid]:<16}{v_regraded[t.tid]:<16}{v_deployed[t.tid]:<16}")
    print("-" * 78)

    def _line(tag, rep):
        print(f"  {tag:<26} strict={rep.strict_recall:6.2%}  recall(form)={rep.recall:6.2%}  "
              f"full={rep.n_full}/{rep.n_targets}  recovered={rep.n_recovered}/{rep.n_targets}")

    _line("(1) oracle learn+grade", rep_o)
    _line("(2) oracle forms, obs gate", rep_r)
    _line("(3) observed learn+grade", rep_d)
    print("-" * 78)
    d_strict = rep_o.strict_recall - rep_d.strict_recall
    d_recall = rep_o.recall - rep_d.recall
    print(f"  deployment cost: strict_recall {rep_o.strict_recall:.3f} -> {rep_d.strict_recall:.3f} "
          f"(drop {d_strict:+.3f});  form recall {rep_o.recall:.3f} -> {rep_d.recall:.3f} "
          f"(drop {d_recall:+.3f})")
    weakened = [t.tid for t in TESTABLE_TARGETS
                if v_oracle[t.tid] != v_regraded[t.tid]]
    if weakened:
        print(f"  forms whose verdict changed under the observed-only gate: {', '.join(weakened)}")
        print("  reading: form recall is controlled, so strictness changes isolate the price of")
        print("           losing the clean oracle under the observed-only noise gate.")
    elif d_strict or d_recall:
        print("  reading: the deployment cost comes from forms found under one learning regime but")
        print("           not the other, not from a verdict change in shared forms.")
    else:
        print("  reading: no deployed-gate penalty is visible because oracle and observed-only")
        print("           learning recovered the same form set with the same verdicts.")

    if not getattr(args, "no_artifacts", False):
        out_dir = _artifact_dir(args.work_dir, "comparisons", "compare", dataset)
        out_path = os.path.join(out_dir, f"compare_{dataset}.json")
        bundle = {
            "kind": "deployed_comparison",
            "dataset": dataset,
            "data_path": path,
            "eta": rc.eval.rel_noise,
            "gate_k": rc.eval.gate_k,
            "proposer": rc.grammar.proposer,
            "iters": rc.search.iterations,
            "oracle": recall_record(rep_o),
            "oracle_forms_observed_gate": recall_record(rep_r),
            "observed": recall_record(rep_d),
            "verdicts": {
                t.tid: {
                    "oracle": v_oracle[t.tid],
                    "regraded": v_regraded[t.tid],
                    "deployed": v_deployed[t.tid],
                } for t in TESTABLE_TARGETS
            },
            "weakened_forms": weakened,
        }
        write_result_json(out_path, bundle)
        print()
        print(f"Compare artifact -> {out_path}  (three gradings + per-target verdicts)")
    return 0


# ----------------------------------------------------------------------------------- cleaning

# Generated by dump-prompt/run; safe to delete and regenerate.
_PROMPT_GLOB = "subagent_prompt_*.txt"
# Machine-readable run artifacts live in unique run-scoped subdirectories and are safe to delete.
_ARTIFACT_DIR_GLOBS = (
    os.path.join("runs", "run_*"),
    os.path.join("benches", "bench_*"),
    os.path.join("comparisons", "compare_*"),
    os.path.join("benchmarks", "benchmark2_*"),
)
_ARTIFACT_FILE_GLOBS = ("trace_*.jsonl", "result_*.json", "bench_*.json", "compare_*.json")
# Human-readable learned-rule files at the repo-root rules/ dir; pure regenerable run outputs.
_RULES_GLOB = os.path.join("rules", "*.pl")
# Committed genuine subagent replies; reproducibility inputs, kept unless --all.
_RESPONSE_GLOB = "subagent_response_*.json"
# Repo-level caches / build outputs, only touched with --all.
_CACHE_DIR_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache",
                    ".mypy_cache", ".ipynb_checkpoints")
_BUILD_GLOBS = ("build", "dist", "*.egg-info")
_SKIP_DIR_NAMES = (".venv", "venv", ".git")


def _clean_targets(work_dir: str, include_all: bool) -> list:
    """Collect existing filesystem paths the clean command would remove."""
    targets = list(glob.glob(os.path.join(work_dir, "**", _PROMPT_GLOB), recursive=True))
    for pat in _ARTIFACT_DIR_GLOBS:
        targets += glob.glob(os.path.join(work_dir, "**", pat), recursive=True)
    for pat in _ARTIFACT_FILE_GLOBS:
        targets += glob.glob(os.path.join(work_dir, pat))
    # Learned-rule files live at the repo-root rules/ dir (independent of work_dir).
    targets += glob.glob(_RULES_GLOB)
    if include_all:
        targets += glob.glob(os.path.join(work_dir, "**", _RESPONSE_GLOB), recursive=True)
        for root, dirs, _files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIR_NAMES]
            for d in list(dirs):
                if d in _CACHE_DIR_NAMES:
                    targets.append(os.path.join(root, d))
        for pat in _BUILD_GLOBS:
            targets += glob.glob(pat)
    seen, uniq = set(), []
    for t in targets:
        n = os.path.normpath(t)
        if n not in seen and os.path.exists(n):
            seen.add(n)
            uniq.append(n)
    return uniq


def cmd_benchmark2(args: argparse.Namespace) -> int:
    """Run the structurally-different second benchmark (schema-generality proof).

    Unlike ``run`` (which loads a hardcoded CrossCheck ``.pkl``), this command synthesises a
    schema whose column syntax, demand encoding, and cell format are all different from
    CrossCheck, compiles a declarative :class:`SchemaSpec` into an adapter, and drives the
    *same* engine through it -- proving the four CrossCheck seams (name parser, role grounding,
    cell codec) are *induced from the spec*, not hardcoded.  The planted ground truth is graded
    with the same form-and-strictness matcher (:func:`score_against`).
    """
    from .schema.benchmark2 import run_benchmark2, BENCH2_TARGETS, benchmark2_config

    rep, portfolio, ds, n_eval, n_uniq = run_benchmark2(
        n_snapshots=args.snapshots, n_nodes=args.nodes, seed=args.seed,
        iterations=args.iters)

    print(f"== P3 schema-generality benchmark :: schema=benchmark2 proposer=scripted ==")
    print(f"   synthetic   : {ds.n_snapshots} snapshots, {len(ds.observed.names)} cols "
          f"({len(ds.name_model.low_cols)} low, {len(ds.name_model.high_cols)} high, "
          f"{len(ds.name_model.nodes)} nodes)")
    print(f"   note        : tx_/rx_/if_/dem[] syntax + scalar cells + regex demand matcher;")
    print(f"                 schema is compiled from a SchemaSpec, NOT the hardcoded CrossCheck path.")
    print()
    print(f"Learned portfolio ({len(portfolio)} invariants; {n_eval} candidates evaluated, "
          f"{n_uniq} unique):")
    print("-" * 78)
    for r in portfolio:
        print("  " + r.summary())
    print("-" * 78)
    print()
    print(format_report(rep))

    if not getattr(args, "no_artifacts", False):
        out_dir = _artifact_dir(args.work_dir, "benchmarks", "benchmark2", seed=args.seed)
        result_path = os.path.join(out_dir, "result_benchmark2.json")
        rc = benchmark2_config(n_nodes=args.nodes, iterations=args.iters, seed=args.seed)
        bundle = {
            "schema": "benchmark2",
            "kind": "schema-generality benchmark (induced schema, scripted proposer)",
            "snapshots": ds.n_snapshots,
            "nodes": sorted(ds.name_model.nodes),
            "n_cols": len(ds.observed.names),
            "n_low": len(ds.name_model.low_cols),
            "n_high": len(ds.name_model.high_cols),
            "n_evaluated": n_eval,
            "n_unique": n_uniq,
            "recall": rep.recall,
            "strict_recall": rep.strict_recall,
            "n_targets": rep.n_targets,
            "n_full": rep.n_full,
            "n_recovered": rep.n_recovered,
            "matches": [
                {"tid": m.tid, "status": m.status, "rule": m.rule_summary, "note": m.note}
                for m in rep.matches
            ],
            "portfolio": [r.summary() for r in portfolio],
            "knobs": {
                "seed": rc.seed, "iterations": rc.search.iterations,
                "anti_seeds": rc.search.anti_seeds, "proposer": rc.grammar.proposer,
            },
        }
        write_result_json(result_path, bundle)
        print()
        print("Run artifacts:")
        print(f"  result -> {result_path}  (induced-schema recall + portfolio)")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    targets = _clean_targets(args.work_dir, args.all)
    if not targets:
        print("clean: nothing to remove.")
        return 0
    verb = "would remove" if args.dry_run else "removing"
    for t in targets:
        print(f"  {verb}: {t}")
        if not args.dry_run:
            if os.path.isdir(t):
                shutil.rmtree(t, ignore_errors=True)
            else:
                try:
                    os.remove(t)
                except OSError:
                    pass
    note = "" if args.all else \
        "  (kept committed subagent_response_*.json; use --all to remove)"
    prefix = "dry-run, " if args.dry_run else ""
    print(f"clean: {prefix}{len(targets)} item(s).{note}")
    return 0


# ------------------------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autogram",
                                description="P3 soft/expressive invariant learner")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="learn invariants and report recall")
    pr.add_argument("--config", default=None, help="YAML config path")
    pr.add_argument("--dataset", default=None, help="abilene | geant")
    pr.add_argument("--data-path", default=None, help="override .pkl path")
    pr.add_argument("--proposer", default=None, choices=["scripted", "openai", "subagent"])
    pr.add_argument("--iters", type=int, default=None, help="middle-loop iterations")
    pr.add_argument("--rounds", type=int, default=None, help="outer grammar rounds")
    pr.add_argument("--start", default=None, choices=["full", "narrow"],
                    help="starting grammar: 'full' (default, start == ceiling, no widening) or "
                         "'narrow' (restricted; the proposer must widen G toward the ceiling)")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (openai backend)")
    pr.add_argument("--work-dir", default="artifacts",
                    help="prompt/response exchange dir and artifact root")
    pr.add_argument("--no-artifacts", action="store_true",
                    help="skip writing run-scoped trace/result artifacts")
    pr.add_argument("--rules-dir", default="rules",
                    help="directory for the human-readable learned-rule file (default rules/)")
    pr.add_argument("--no-rules", action="store_true",
                    help="skip writing the human-readable rules/<dataset>_<time>.pl file")
    pr.add_argument("--no-score", action="store_true",
                    help="learn without grading against the catalogue (oracle-free); "
                         "score later with `autogram score`")
    pr.add_argument("--deployed", action="store_true",
                    help="observed-only gate: estimate noise from --rel-noise instead of the "
                         "clean oracle (deployment setting, Sec. 5.5)")
    pr.add_argument("--rel-noise", type=float, default=None,
                    help="modelled relative per-cell noise eta for --deployed (default 0.02)")
    pr.set_defaults(func=cmd_run)

    psc = sub.add_parser("score",
                         help="grade a learned portfolio bundle against the catalogue")
    psc.add_argument("--dataset", default=None,
                     help="abilene | geant (locates the newest result_<dataset>.json below --work-dir)")
    psc.add_argument("--result", default=None,
                     help="explicit result bundle path (overrides --dataset lookup)")
    psc.add_argument("--data-path", default=None,
                     help="override dataset .pkl (else taken from the bundle)")
    psc.add_argument("--work-dir", default="artifacts", help="artifact root holding result bundles")
    psc.add_argument("--out", default=None,
                     help="scored-bundle output path (default: next to the selected result bundle)")
    psc.set_defaults(func=cmd_score)

    pb = sub.add_parser("bench",
                        help="multi-seed recall benchmark (mean/var/worst-case + hit-rate)")
    pb.add_argument("--config", default=None, help="YAML config path")
    pb.add_argument("--dataset", default=None, help="abilene | geant")
    pb.add_argument("--data-path", default=None, help="override .pkl path")
    pb.add_argument("--proposer", default=None, choices=["scripted", "openai", "subagent"])
    pb.add_argument("--iters", type=int, default=None, help="middle-loop iterations")
    pb.add_argument("--seeds", default="0,1,2",
                    help="comma-separated seeds, e.g. 0,1,2,3,4 (default: 0,1,2)")
    pb.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (openai backend)")
    pb.add_argument("--work-dir", default="artifacts",
                    help="prompt/response exchange dir and artifact root")
    pb.add_argument("--no-artifacts", action="store_true",
                    help="skip writing run-scoped bench artifact")
    pb.set_defaults(func=cmd_bench)

    pc = sub.add_parser("compare",
                        help="oracle-gated vs observed-only (deployed) recall comparison")
    pc.add_argument("--config", default=None, help="YAML config path")
    pc.add_argument("--dataset", default=None, help="abilene | geant")
    pc.add_argument("--data-path", default=None, help="override .pkl path")
    pc.add_argument("--proposer", default=None, choices=["scripted", "openai", "subagent"])
    pc.add_argument("--iters", type=int, default=None, help="middle-loop iterations")
    pc.add_argument("--rel-noise", type=float, default=None,
                    help="modelled relative per-cell noise eta (default from config: 0.02)")
    pc.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (openai backend)")
    pc.add_argument("--work-dir", default="artifacts",
                    help="prompt/response exchange dir and artifact root")
    pc.add_argument("--no-artifacts", action="store_true",
                    help="skip writing run-scoped compare artifact")
    pc.set_defaults(func=cmd_compare)

    pd = sub.add_parser("dump-prompt", help="write the leakage-safe subagent prompt only")
    pd.add_argument("--config", default=None)
    pd.add_argument("--dataset", default=None, help="abilene | geant")
    pd.add_argument("--data-path", default=None)
    pd.add_argument("--seed", type=int, default=None)
    pd.add_argument("--work-dir", default="artifacts", help="artifact root for unique prompt dirs")
    pd.set_defaults(func=cmd_dump_prompt)

    pbm = sub.add_parser("benchmark2",
                         help="schema-generality proof: run a structurally-different induced schema")
    pbm.add_argument("--snapshots", type=int, default=400, help="synthetic snapshots (default 400)")
    pbm.add_argument("--nodes", type=int, default=4, help="synthetic node count (default 4)")
    pbm.add_argument("--iters", type=int, default=200, help="middle-loop iterations (default 200)")
    pbm.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    pbm.add_argument("--work-dir", default="artifacts",
                     help="artifact root for run-scoped result_benchmark2.json")
    pbm.add_argument("--no-artifacts", action="store_true",
                     help="skip writing result_benchmark2.json")
    pbm.set_defaults(func=cmd_benchmark2)

    pcl = sub.add_parser("clean",
                         help="remove generated run artifacts (--all also responses + caches)")
    pcl.add_argument("--work-dir", default="artifacts",
                     help="run-artifact dir to clean (default: artifacts)")
    pcl.add_argument("--all", action="store_true",
                     help="also remove committed subagent_response_*.json and repo caches/build dirs")
    pcl.add_argument("--dry-run", action="store_true",
                     help="list what would be removed without deleting")
    pcl.set_defaults(func=cmd_clean)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
