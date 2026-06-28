"""Command-line interface for the open-ended discovery engine.

Subcommands:

* ``discover``  -- induce a schema and discover invariants on a synthetic (or seeded) dataset,
  printing a stable, parsimonious portfolio judged from data alone and saving it to ``./rules``
  as a timestamped ``.pl`` file (disable with ``--no-save-rules``).
* ``validate``  -- run the adversarial sanity checks (plant-and-recover noise sweep, null
  dataset, tautology rejection, rename invariance, ablations) and proxy signals.
* ``clean``     -- remove generated artifacts.

There is no catalogue-bound subcommand: success is not recall, it is stable + lifts +
parsimonious invariants confirmed on held-out data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Optional

from .config import DiscoveryConfig, SearchConfig
from .discovery import synth
from .discovery.loop import discover
from .discovery.validate import run_all


def _git_short() -> str:
    """Best-effort short commit SHA for rule-file provenance (``unknown`` if unavailable)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _vocab_from_args(args) -> synth.Vocab:
    return synth.Vocab()


def cmd_discover(args: argparse.Namespace) -> int:
    data = synth.make_synthetic(
        n_entities=args.entities, n_snapshots=args.snapshots,
        noise=args.noise, seed=args.seed)
    dcfg = DiscoveryConfig(n_perm=args.permutations, seed=args.seed)
    scfg = SearchConfig(rounds=args.rounds, proposals_per_round=args.proposals,
                        max_complexity=args.max_complexity, proposer=args.proposer,
                        seed=args.seed)
    res = discover(data.columns, data.matrix, discovery_cfg=dcfg, search_cfg=scfg,
                   name=args.name, timestamps=data.timestamps)
    print(res.report())
    if args.json:
        payload = {
            "dataset": res.dataset.name,
            "rounds": res.rounds_run,
            "reinductions": res.reinductions,
            "progress": res.progress_history,
            "portfolio": [
                {
                    "rule": ev.rule.unparse(),
                    "strictness": ev.strictness,
                    "coverage": ev.coverage,
                    "coverage_ci": [ev.coverage_lo, ev.coverage_hi],
                    "operating_coverage": ev.operating_cov,
                    "lift": ev.lift,
                    "lift_percentile": ev.lift_percentile,
                    "stability_std": ev.stability_std,
                    "support_margin": ev.support_margin,
                    "stability_margin": ev.stability_margin,
                    "mdl_gain": ev.mdl_gain,
                    "support": ev.support,
                    "n_bindings": ev.n_bindings,
                }
                for ev in res.portfolio
            ],
        }
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    if not args.no_save_rules:
        from .discovery.export import write_rules_pl
        pl_path = write_rules_pl(res, args.name, out_dir=args.rules_dir,
                                 seed=args.seed, proposer=args.proposer, git=_git_short())
        if pl_path:
            print(f"wrote {pl_path}")
    return 0 if res.portfolio else 1


def cmd_validate(args: argparse.Namespace) -> int:
    report = run_all(seed=args.seed)
    print(json.dumps(report, indent=2))
    ok = (report["plant_ok"]
          and report["portfolio_quality"]["ok"]
          and report["null_accepted"] == 0
          and not report["tautology"]["self_comparison_admissible"]
          and not report["tautology"]["nonneg_accepted"]
          and report["rename_invariance"]["invariant"]
          and all(report["ablations"].values()))
    print("\nVALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_clean(args: argparse.Namespace) -> int:
    removed = []
    for path in (args.out, "artifacts/discovery"):
        if path and os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(path)
    print("removed:", removed if removed else "(nothing)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autogram", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("discover", help="induce a schema and discover invariants from data")
    pd.add_argument("--entities", type=int, default=6)
    pd.add_argument("--snapshots", type=int, default=400)
    pd.add_argument("--noise", type=float, default=0.02)
    pd.add_argument("--rounds", type=int, default=6)
    pd.add_argument("--proposals", type=int, default=160)
    pd.add_argument("--permutations", type=int, default=16)
    pd.add_argument("--max-complexity", dest="max_complexity", type=int, default=12)
    pd.add_argument("--proposer", choices=("random", "portfolio", "llm"), default="random",
                    help="proposal source: deterministic random or LLM+random portfolio")
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--name", default="synthetic")
    pd.add_argument("--json", default="", help="optional path to write the portfolio as JSON")
    pd.add_argument("--rules-dir", dest="rules_dir", default="rules",
                    help="directory to save discovered invariants as a .pl file (default: rules)")
    pd.add_argument("--no-save-rules", dest="no_save_rules", action="store_true",
                    help="do not write the .pl rules file to the rules directory")
    pd.set_defaults(func=cmd_discover)

    pv = sub.add_parser("validate", help="run adversarial sanity checks + proxy signals")
    pv.add_argument("--seed", type=int, default=0)
    pv.set_defaults(func=cmd_validate)

    pc = sub.add_parser("clean", help="remove generated artifacts")
    pc.add_argument("--out", default="artifacts/discovery")
    pc.set_defaults(func=cmd_clean)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
