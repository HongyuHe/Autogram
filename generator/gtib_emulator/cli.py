"""Command-line interface for the gTIB byte-completeness emulator.

Examples
--------
Generate a small default dataset into ``./output`` and validate invariants::

    python -m gtib_emulator generate

Use a config file and override a couple of knobs::

    python -m gtib_emulator generate --config config.yaml --n-consumers 20 --duration-hours 24

Only print the effective configuration (defaults + file + overrides)::

    python -m gtib_emulator describe --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_config
from .generate import run, write_outputs
from .invariants import check_all


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate common CLI flags into a nested config overlay."""

    ov: dict[str, Any] = {}
    if args.seed is not None:
        ov["seed"] = args.seed
    if args.duration_hours is not None:
        ov.setdefault("time", {})["duration_hours"] = args.duration_hours
    if args.n_consumers is not None:
        ov.setdefault("scale", {})["n_consumers"] = args.n_consumers
    if args.fmt is not None:
        ov.setdefault("output", {})["fmt"] = args.fmt
    if args.out is not None:
        ov.setdefault("output", {})["directory"] = args.out
    if args.no_raw:
        ov.setdefault("output", {})["write_raw"] = False
    return ov


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides_from_args(args))
    result = run(cfg)
    written = write_outputs(result)

    if not args.quiet:
        _print_summary(result, written)

    if not args.no_validate:
        results = check_all(cfg, result.records, result.events)
        hard_failed = _print_invariants(results, quiet=args.quiet)
        if hard_failed:
            print("\nFAIL: one or more HARD invariants were violated.", file=sys.stderr)
            return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides_from_args(args))
    result = run(cfg)
    results = check_all(cfg, result.records, result.events)
    hard_failed = _print_invariants(results, quiet=False)
    return 1 if hard_failed else 0


def _cmd_describe(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, _overrides_from_args(args))
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    return 0


def _print_summary(result: Any, written: dict[str, str]) -> None:
    m = result.manifest
    sc = m["scale"]
    ev = m["evaluation"]
    print("=" * 68)
    print("gTIB byte-completeness emulator -- run summary")
    print("=" * 68)
    print(f"seed={m['seed']}  consumers={sc['n_consumers']}  shards={sc['total_shards']}  "
          f"minutes={sc['n_minutes']}")
    print(f"archetypes: {sc['archetypes']}")
    print(f"events: {m['event_counts']}")
    print(f"normal ratio_1h median={m['summary_stats']['normal_ratio_1h_median']}  "
          f"max ratio_1m={m['summary_stats']['max_completeness_ratio_1m']}")
    sr, tr = ev["static_rule"], ev["trajectory_rule"]
    print("-" * 68)
    print("static rule   :", {k: sr[k] for k in ("tp", "fp", "fn", "precision", "recall",
                                                 "false_positives_on_benign_minutes")})
    print("trajectory rule:", {k: tr[k] for k in ("tp", "fp", "fn", "precision", "recall",
                                                  "false_positives_on_benign_minutes")})
    print("-" * 68)
    for name, path in written.items():
        print(f"wrote {name:9s} -> {path}")
    print("=" * 68)


def _print_invariants(results: list[Any], quiet: bool) -> bool:
    hard_failed = False
    if not quiet:
        print("\ninvariant checks:")
    for r in results:
        if not r.passed and r.tier == "hard":
            hard_failed = True
        if not quiet:
            mark = "ok " if r.passed else "XX "
            print(f"  [{mark}][{r.tier:4s}] {r.name}: {r.detail}")
    return hard_failed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gtib_emulator",
                                description="Synthetic-data emulator for gTIB Consumer Byte Completeness.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", default=None, help="Path to a YAML config file (optional).")
        sp.add_argument("--seed", type=int, default=None, help="Override the RNG seed.")
        sp.add_argument("--duration-hours", type=float, default=None, dest="duration_hours")
        sp.add_argument("--n-consumers", type=int, default=None, dest="n_consumers")
        sp.add_argument("--format", choices=("csv", "parquet"), default=None, dest="fmt")
        sp.add_argument("--out", default=None, help="Output directory.")
        sp.add_argument("--no-raw", action="store_true", help="Skip the per-shard raw counter table.")

    g = sub.add_parser("generate", help="Generate a labelled synthetic dataset.")
    add_common(g)
    g.add_argument("--no-validate", action="store_true", help="Skip invariant checks after generation.")
    g.add_argument("--quiet", action="store_true", help="Suppress the run summary.")
    g.set_defaults(func=_cmd_generate)

    v = sub.add_parser("validate", help="Generate in-memory and only run invariant checks.")
    add_common(v)
    v.set_defaults(func=_cmd_validate)

    d = sub.add_parser("describe", help="Print the effective configuration and exit.")
    add_common(d)
    d.set_defaults(func=_cmd_describe)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
