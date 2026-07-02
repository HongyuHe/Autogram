"""Command-line interface for Autogram v2."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from typing import Optional

from .config import DiscoveryConfig, SearchConfig
from .discovery import synth
from .discovery.induce import available_inducer_backends, make_inducer
from .discovery.subagent import HARNESSES
from .discovery.loop import discover, discover_dataframe
from .discovery.validate import run_all


def _git_short() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _portfolio_payload(res) -> dict:
    return {
        "dataset": res.dataset.name,
        "rounds": res.rounds_run,
        "progress": res.progress_history,
        "statistic": "hold_rate_wilson_ci",
        "diagnostics": list(getattr(res, "diagnostics", ())),
        "portfolio": [
            {
                "rule": ev.rule.unparse(),
                "strictness": ev.strictness,
                "hold_rate": ev.hold_rate,
                "hold_rate_ci": [ev.hold_rate_lo, ev.hold_rate_hi],
                "eps": ev.eps,
                "mdl_gain": ev.mdl_gain,
                "support": ev.support,
                "n_bindings": ev.n_bindings,
            }
            for ev in res.portfolio
        ],
    }


def _load_pickle_dataframe(path: str):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not hasattr(obj, "columns"):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    return obj


def cmd_discover(args: argparse.Namespace) -> int:
    dcfg = DiscoveryConfig(tolerance=args.tolerance, hold_rate_threshold=args.hold_rate,
                           ci_alpha=args.ci_alpha, seed=args.seed)
    scfg = SearchConfig(max_complexity=args.max_complexity, max_add_arity=args.max_add_arity,
                        proposer="enumeration", seed=args.seed)
    if args.schema_backend == "subagent":
        inducer = make_inducer("subagent", harness=args.harness)
    else:
        inducer = make_inducer(args.schema_backend)
    if args.input:
        df = _load_pickle_dataframe(args.input)
        name = args.name or os.path.splitext(os.path.basename(args.input))[0]
        res = discover_dataframe(df, inducer=inducer, discovery_cfg=dcfg, search_cfg=scfg, name=name)
    else:
        data = synth.make_synthetic(n_entities=args.entities, n_snapshots=args.snapshots,
                                    noise=args.noise, seed=args.seed)
        name = args.name or "synthetic"
        res = discover(data.columns, data.matrix, inducer=inducer, discovery_cfg=dcfg,
                       search_cfg=scfg, name=name, timestamps=data.timestamps)
    print(res.report())
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_portfolio_payload(res), fh, indent=2)
        print(f"\nwrote {args.json}")
    if not args.no_save_rules:
        from .discovery.export import write_rules_dl
        dl_path = write_rules_dl(res, res.dataset.name, out_dir=args.rules_dir,
                                 seed=args.seed, proposer="enumeration", git=_git_short())
        if dl_path:
            print(f"wrote {dl_path}")
    return 0 if res.portfolio else 1


def cmd_validate(args: argparse.Namespace) -> int:
    report = run_all(seed=args.seed)
    print(json.dumps(report, indent=2))
    ok = bool(report.get("proxy_ok") and report.get("synthetic_recovery", {}).get("ok"))
    print("\nVALIDATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_clean(args: argparse.Namespace) -> int:
    removed = []
    for path in (args.out, r"artifacts\discovery"):
        if path and os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(path)
    print("removed:", removed if removed else "(nothing)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autogram", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    discovery_defaults = DiscoveryConfig()
    search_defaults = SearchConfig()

    pd = sub.add_parser("discover", help="induce schema and enumerate invariants from observed data")
    pd.add_argument("--input", default="", help="optional pickle DataFrame path (CrossCheck sample)")
    pd.add_argument("--schema-backend", choices=available_inducer_backends(), default="subagent")
    pd.add_argument("--harness", choices=sorted(HARNESSES), default="copilot",
                    help="agentic CLI harness for the subagent schema backend (copilot|codex|claude)")
    pd.add_argument("--entities", type=int, default=6)
    pd.add_argument("--snapshots", type=int, default=400)
    pd.add_argument("--noise", type=float, default=0.02)
    pd.add_argument("--tolerance", type=float, default=discovery_defaults.tolerance)
    pd.add_argument("--hold-rate", dest="hold_rate", type=float,
                    default=discovery_defaults.hold_rate_threshold)
    pd.add_argument("--ci-alpha", dest="ci_alpha", type=float, default=discovery_defaults.ci_alpha)
    pd.add_argument("--max-complexity", dest="max_complexity", type=int,
                    default=search_defaults.max_complexity)
    pd.add_argument("--max-add-arity", dest="max_add_arity", type=int,
                    default=search_defaults.max_add_arity)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--name", default="")
    pd.add_argument("--json", default="", help="optional path to write JSON report")
    pd.add_argument("--rules-dir", dest="rules_dir", default="rules")
    pd.add_argument("--no-save-rules", dest="no_save_rules", action="store_true")
    pd.set_defaults(func=cmd_discover)

    pv = sub.add_parser("validate", help="run synthetic proxy validation")
    pv.add_argument("--seed", type=int, default=0)
    pv.set_defaults(func=cmd_validate)

    pc = sub.add_parser("clean", help="remove generated artifacts")
    pc.add_argument("--out", default=r"artifacts\discovery")
    pc.set_defaults(func=cmd_clean)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
