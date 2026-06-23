"""Command-line entry point for the P3 invariant-learning engine.

Two subcommands:

* ``run``          -- load a config, run the three-level learning loop on a dataset, then
  print the assembled invariant portfolio, the recall scorecard against the known catalogue,
  and the exact knob settings used (for reproducibility).
* ``dump-prompt``  -- render *only* the leakage-safe proposer prompt for a dataset and write
  it to ``<work-dir>/subagent_prompt_<dataset>.txt``.  This supports the leakage-free
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
import os
import shutil
import sys
from dataclasses import fields as dc_fields
from typing import Optional

from .config import EvalConfig, GrammarConfig, RunConfig, SearchConfig
from .loader.loader import load_dataset
from .dsl.grammar import Grammar, default_grammar
from .proposer import build_context, make_proposer
from .proposer.base import render_proposal_prompt
from .search.loop import learn
from .search.recall import format_report, score_recall

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


def _capped_grammar(max_complexity: int) -> Grammar:
    G = default_grammar()
    return Grammar(binders=G.binders, ops=G.ops, ref_roles=G.ref_roles,
                   fam_roles=G.fam_roles, agg_kinds=G.agg_kinds,
                   max_complexity=max_complexity, max_add_arity=G.max_add_arity)


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
    if args.seed is not None:
        rc.seed = args.seed
    if args.data_path is not None:
        rc.data_path = args.data_path
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
    rep = score_recall(res.portfolio)
    print(format_report(rep))

    used_real = getattr(proposer, "used_real_subagent", None)
    if rc.grammar.proposer == "subagent":
        print()
        print(f"  subagent: used_real_subagent={used_real}  "
              f"(prompt -> {os.path.join(args.work_dir, f'subagent_prompt_{rc.dataset}.txt')})")
        if not used_real:
            print("  subagent: NO external reply found; portfolio came from blind random "
                  "search only.\n           Run 'dump-prompt', answer with an isolated "
                  "subagent, then re-run.")

    print()
    print("Knob settings (reproducibility):")
    print(f"  seed={rc.seed}  proposer={rc.grammar.proposer}  rounds={rc.grammar.rounds}  "
          f"max_complexity={rc.grammar.max_complexity}")
    print(f"  iterations={rc.search.iterations}  islands={rc.search.islands}  "
          f"thompson={rc.search.thompson}  bootstrap_random={rc.search.bootstrap_random}  "
          f"seed_from_grammar={rc.search.seed_from_grammar}")
    print(f"  target_coverage={rc.eval.target_coverage}  gate_k={rc.eval.gate_k}  "
          f"eps_exact={rc.eval.eps_exact}  eps_max={rc.eval.eps_max}  lift_min={rc.eval.lift_min}")
    return 0


def cmd_dump_prompt(args: argparse.Namespace) -> int:
    rc = load_config(args.config, args.dataset)
    if args.seed is not None:
        rc.seed = args.seed
    if args.data_path is not None:
        rc.data_path = args.data_path
    path = _resolve_path(rc)
    ds = load_dataset(path, name=rc.dataset)
    G = _capped_grammar(rc.grammar.max_complexity)
    ctx = build_context(ds, G)
    prompt = render_proposal_prompt(ctx)        # raises if anything leaks

    os.makedirs(args.work_dir, exist_ok=True)
    out = os.path.join(args.work_dir, f"subagent_prompt_{rc.dataset}.txt")
    with open(out, "w", encoding="ascii", errors="replace") as fh:
        fh.write(prompt)
    print(f"wrote leakage-safe proposer prompt -> {out}  ({len(prompt)} chars)")
    print("Hand this file (and nothing else) to an isolated subagent; save its JSON reply to")
    print(f"  {os.path.join(args.work_dir, f'subagent_response_{rc.dataset}.json')}")
    return 0


# ----------------------------------------------------------------------------------- cleaning

# Generated every run; safe to delete and regenerate.
_PROMPT_GLOB = "subagent_prompt_*.txt"
# Committed genuine subagent replies; reproducibility inputs, kept unless --all.
_RESPONSE_GLOB = "subagent_response_*.json"
# Repo-level caches / build outputs, only touched with --all.
_CACHE_DIR_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache",
                    ".mypy_cache", ".ipynb_checkpoints")
_BUILD_GLOBS = ("build", "dist", "*.egg-info")
_SKIP_DIR_NAMES = (".venv", "venv", ".git")


def _clean_targets(work_dir: str, include_all: bool) -> list:
    """Collect existing filesystem paths the clean command would remove."""
    targets = list(glob.glob(os.path.join(work_dir, _PROMPT_GLOB)))
    if include_all:
        targets += glob.glob(os.path.join(work_dir, _RESPONSE_GLOB))
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
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (openai backend)")
    pr.add_argument("--work-dir", default="artifacts", help="prompt/response exchange dir")
    pr.set_defaults(func=cmd_run)

    pd = sub.add_parser("dump-prompt", help="write the leakage-safe subagent prompt only")
    pd.add_argument("--config", default=None)
    pd.add_argument("--dataset", default=None, help="abilene | geant")
    pd.add_argument("--data-path", default=None)
    pd.add_argument("--seed", type=int, default=None)
    pd.add_argument("--work-dir", default="artifacts")
    pd.set_defaults(func=cmd_dump_prompt)

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
