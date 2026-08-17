"""Command-line interface for Autogram v2."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from typing import Optional

from .calibrate import CalibrationConfig
from .config import DiscoveryConfig, SearchConfig
from .discovery import synth
from .discovery.induce import available_inducer_backends, make_inducer
from .discovery.subagent import HARNESSES, configured_harness
from .discovery.loop import discover, discover_dataframe
from .discovery.validate import run_all


def _git_short() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _portfolio_payload(res, *, provenance=None, normalized_spec=None, effective=None) -> dict:
    from .discovery.export import _adapter_of
    from .dsl.parser import rule_to_dict
    from .dsl.render import render_rule
    adapter = _adapter_of(res)
    payload = {
        "dataset": res.dataset.name,
        "rounds": res.rounds_run,
        "progress": res.progress_history,
        "statistic": "hold_rate_wilson_ci",
        "diagnostics": list(getattr(res, "diagnostics", ())),
        "portfolio": [
            {
                "rule": ev.rule.unparse(),
                "rule_explicit": render_rule(ev.rule, adapter),
                "rule_payload": rule_to_dict(ev.rule),
                "strictness": ev.strictness,
                "hold_rate": ev.hold_rate,
                "hold_rate_ci": [ev.hold_rate_lo, ev.hold_rate_hi],
                "eps": ev.eps,
                "mdl_gain": ev.mdl_gain,
                "support": ev.support,
                "n_bindings": ev.n_bindings,
                "parameters": dict(getattr(ev, "parameters", {})),
            }
            for ev in res.portfolio
        ],
    }
    if effective is not None:
        payload["effective_settings"] = effective
    if normalized_spec is not None:
        from .calibrate import _json_fingerprint, _json_normalize
        payload["normalized_spec"] = _json_normalize(normalized_spec)
        payload["normalized_spec_sha256"] = _json_fingerprint(payload["normalized_spec"])
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


def _load_dataframe(path: str, raw_path: str = ""):
    from pathlib import Path

    from .loader.gtib import infer_tabular_profile, prepare_gtib_files, prepare_gtib_raw

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        # An explicit --raw-input is authoritative: the GTIB derived+raw join must happen even when
        # the derived CSV has been renamed away from the conventional `timeseries_derived.csv`
        # sibling-detection filename. Only fall back to filename-based auto-detection when no raw
        # input was supplied.
        if raw_path:
            return prepare_gtib_files(source, raw_path)
        if source.name == "timeseries_derived.csv":
            return prepare_gtib_files(source, None)
        import pandas as pd
        frame = pd.read_csv(source)
        if source.name == "timeseries_raw.csv":
            return prepare_gtib_raw(frame)
        return infer_tabular_profile(frame)
    if suffix not in (".pkl", ".pickle"):
        raise ValueError(f"unsupported input format {suffix!r}; expected .csv, .pkl, or .pickle")
    with source.open("rb") as fh:
        obj = pickle.load(fh)
    if not hasattr(obj, "columns"):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    return obj


def _load_pickle_dataframe(path: str):
    """Backward-compatible alias for callers that only used pickle inputs."""

    return _load_dataframe(path)


def _configure_dataframe_profile(df, args):
    from .loader.gtib import AUTOGRAM_PROFILE_ATTR, profile_dataframe

    profile = dict(df.attrs.get(AUTOGRAM_PROFILE_ATTR, {}))
    explicit = any((
        getattr(args, "time_index", ""),
        getattr(args, "group_keys", []),
        getattr(args, "condition_columns", []),
        getattr(args, "windows", []),
        getattr(args, "max_lag", None) is not None,
        getattr(args, "run_lengths", []),
        getattr(args, "max_conjunction_terms", None) is not None,
        getattr(args, "advanced", False),
        getattr(args, "max_degree", 0),
        getattr(args, "proportional", False),
        getattr(args, "agg_kinds", []),
    ))
    if not explicit:
        return df
    return profile_dataframe(
        df,
        time_index=getattr(args, "time_index", "") or profile.get("time_index") or None,
        group_keys=getattr(args, "group_keys", []) or profile.get("group_keys", ()),
        condition_columns=(
            getattr(args, "condition_columns", [])
            or profile.get("condition_columns", ())
        ),
        families=profile.get("families", {}),
        related_frames=profile.get("related_frames", {}),
        temporal_windows=getattr(args, "windows", []) or profile.get("temporal_windows", ()),
        max_lag=(
            getattr(args, "max_lag", None)
            if getattr(args, "max_lag", None) is not None
            else profile.get("max_lag", 0)
        ),
        related_aggregates=profile.get("related_aggregates", {}),
        run_lengths=getattr(args, "run_lengths", []) or profile.get("run_lengths", ()),
        advanced=bool(getattr(args, "advanced", False) or profile.get("advanced", False)),
        max_conjunction_terms=int(
            getattr(args, "max_conjunction_terms", None)
            if getattr(args, "max_conjunction_terms", None) is not None
            else profile.get("max_conjunction_terms", 3)
        ),
        metadata_columns=profile.get("metadata_columns", ()),
        band_enabled=bool(profile.get("band_enabled", False)),
        max_degree=int(
            getattr(args, "max_degree", 0)
            or profile.get("max_degree", 0)
        ),
        proportional=bool(
            getattr(args, "proportional", False)
            or profile.get("proportional", False)
        ),
        agg_kinds=getattr(args, "agg_kinds", []) or profile.get("agg_kinds", ()),
    )


def _apply_config_file(args: argparse.Namespace, argv=None) -> argparse.Namespace:
    path = getattr(args, "config", "")
    if not path:
        return args
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config {path!r} must contain a top-level mapping")
    explicit = {
        token.split("=", 1)[0]
        for token in (argv or ())
        if isinstance(token, str) and token.startswith("--")
    }

    def assign(attribute, value, flag):
        if value is not None and flag not in explicit and hasattr(args, attribute):
            setattr(args, attribute, value)

    assign("input", config.get("data_path"), "--input")
    assign("raw_input", config.get("raw_path"), "--raw-input")
    assign("known", config.get("known_path"), "--known")
    assign("name", config.get("dataset"), "--name")
    assign("seed", config.get("seed"), "--seed")

    schema = config.get("schema", {}) or {}
    assign("schema_backend", schema.get("backend"), "--schema-backend")
    assign("harness", schema.get("harness"), "--harness")

    profile = config.get("profile", {}) or {}
    assign("time_index", profile.get("time_index"), "--time-index")
    assign("group_keys", profile.get("group_keys"), "--group-key")
    assign("condition_columns", profile.get("condition_columns"), "--condition-column")
    assign("windows", profile.get("windows"), "--window")
    assign("max_lag", profile.get("max_lag"), "--max-lag")
    assign("run_lengths", profile.get("run_lengths"), "--run-length")
    assign(
        "max_conjunction_terms",
        profile.get("max_conjunction_terms"),
        "--max-conjunction-terms",
    )
    assign("advanced", profile.get("advanced"), "--advanced")
    assign("max_degree", profile.get("max_degree"), "--max-degree")
    assign("proportional", profile.get("proportional"), "--proportional")
    assign("agg_kinds", profile.get("agg_kinds"), "--aggregation")

    evaluation = config.get("eval", {}) or {}
    assign("tolerance", evaluation.get("tolerance"), "--tolerance")
    assign("hold_rate", evaluation.get("hold_rate_threshold"), "--hold-rate")
    assign("ci_alpha", evaluation.get("ci_alpha"), "--ci-alpha")
    assign("band_mode", evaluation.get("band_mode"), "--band-mode")

    search = config.get("search", {}) or {}
    assign("max_complexity", search.get("max_complexity"), "--max-complexity")
    assign("max_add_arity", search.get("max_add_arity"), "--max-add-arity")
    assign("max_rules", search.get("max_rules"), "--max-rules")
    assign(
        "max_nonlinear_leaves",
        search.get("max_nonlinear_leaves"),
        "--max-nonlinear-leaves",
    )
    assign(
        "max_linear_leaves",
        search.get("max_linear_leaves"),
        "--max-linear-leaves",
    )
    assign(
        "max_conditioned_rules",
        search.get("max_conditioned_rules"),
        "--max-conditioned-rules",
    )
    assign(
        "max_capability_tiers",
        search.get("max_capability_tiers"),
        "--max-capability-tiers",
    )
    return args


def cmd_discover(args: argparse.Namespace) -> int:
    dcfg = DiscoveryConfig(tolerance=args.tolerance, hold_rate_threshold=args.hold_rate,
                           ci_alpha=args.ci_alpha, seed=args.seed,
                           band_mode=args.band_mode)
    scfg = SearchConfig(
        max_complexity=args.max_complexity,
        max_add_arity=args.max_add_arity,
        max_rules=args.max_rules,
        max_nonlinear_leaves=args.max_nonlinear_leaves,
        max_linear_leaves=args.max_linear_leaves,
        max_conditioned_rules=args.max_conditioned_rules,
        proposer="enumeration",
        seed=args.seed,
        max_lag=int(args.max_lag or 0),
        windows=tuple(args.windows),
    )
    if args.schema_backend == "subagent":
        inducer = make_inducer("subagent", harness=args.harness)
    else:
        inducer = make_inducer(args.schema_backend)
    normalized_spec = None
    input_frame = None
    if args.input:
        input_frame = _configure_dataframe_profile(
            _load_dataframe(args.input, getattr(args, "raw_input", "")),
            args,
        )
        name = args.name or os.path.splitext(os.path.basename(args.input))[0]
        from .discovery.loop import prepare_dataframe, run_prepared
        from .discovery.induce import _spec_to_json
        ds, G, runtime_spec = prepare_dataframe(
            input_frame, inducer=inducer, search_cfg=scfg, name=name
        )
        if G.advanced_enabled and (
            scfg.max_rules <= 0
            or scfg.max_rules > 500_000
        ):
            raise ValueError(
                "effective advanced discovery requires --max-rules "
                "between 1 and 500000"
            )
        res = run_prepared(ds, G, discovery_cfg=dcfg, search_cfg=scfg)
        normalized_spec = _spec_to_json(runtime_spec)
    else:
        data = synth.make_synthetic(n_entities=args.entities, n_snapshots=args.snapshots,
                                    noise=args.noise, seed=args.seed)
        name = args.name or "synthetic"
        res = discover(data.columns, data.matrix, inducer=inducer, discovery_cfg=dcfg,
                       search_cfg=scfg, name=name, timestamps=data.timestamps)
    print(res.report())
    if args.json:
        from .calibrate import _engine_source_fingerprint, _fingerprint
        from dataclasses import asdict
        provenance = {
            "engine_source_sha256": _engine_source_fingerprint(),
            "input_sha256": (
                _fingerprint(input_frame) if input_frame is not None else None
            ),
            "discovery_config_sha256": _fingerprint(dcfg),
            "search_config_sha256": _fingerprint(scfg),
        }
        effective = {
            "discovery": asdict(dcfg),
            "search": asdict(scfg),
        }
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                _portfolio_payload(
                    res,
                    provenance=provenance,
                    normalized_spec=normalized_spec,
                    effective=effective,
                ),
                fh,
                indent=2,
            )
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
    null_safe = all(
        report.get(field) == 0
        for field in (
            "null_equalities_accepted",
            "null_temporal_accepted",
            "null_definitions_accepted",
        )
    )
    ok = bool(
        report.get("proxy_ok")
        and report.get("synthetic_recovery", {}).get("ok")
        and null_safe
    )
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


def cmd_precheck(args: argparse.Namespace) -> int:
    from .calibrate import precheck
    rep = precheck(harness=args.harness, backend=args.schema_backend)
    print(json.dumps(rep, indent=2))
    print("\nPRECHECK:", "OK" if rep["ok"] else "FAIL")
    return 0 if rep["ok"] else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibrate import CalibrationConfig, calibrate
    if not args.input or not args.known:
        raise ValueError("calibrate requires --input and --known, directly or through --config")
    df = _configure_dataframe_profile(
        _load_dataframe(args.input, getattr(args, "raw_input", "")),
        args,
    )
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    cfg = CalibrationConfig(seed=args.seed, max_iterations=args.max_iterations,
                            validation_frac=args.validation_frac, harness=args.harness,
                            backend=args.schema_backend, band_mode=args.band_mode,
                            ci_alpha=args.ci_alpha,
                            tolerance=args.tolerance,
                            hold_rate_threshold=args.hold_rate,
                            max_capability_tiers=args.max_capability_tiers,
                            save_rules=not args.no_save_rules, rules_dir=args.rules_dir)
    cfg.max_complexity = args.max_complexity
    cfg.max_add_arity = args.max_add_arity
    cfg.max_rules = args.max_rules
    cfg.max_nonlinear_leaves = args.max_nonlinear_leaves
    cfg.max_linear_leaves = args.max_linear_leaves
    cfg.max_conditioned_rules = args.max_conditioned_rules
    cfg.max_lag = int(args.max_lag or 0)
    cfg.windows = tuple(args.windows)
    report = calibrate(df, args.known, cfg, name=name)
    print(json.dumps(report, indent=2))
    if report.get("rules_file"):
        print(f"\nlearned invariants written to {report['rules_file']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autogram", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    discovery_defaults = DiscoveryConfig()
    search_defaults = SearchConfig()
    harness_default = configured_harness()

    def add_profile_args(parser):
        parser.add_argument("--time-index", dest="time_index", default="",
                            help="ordered timestamp column for temporal terms")
        parser.add_argument("--group-key", dest="group_keys", action="append", default=[],
                            help="row grouping column; repeat for composite groups")
        parser.add_argument("--condition-column", dest="condition_columns",
                            action="append", default=[],
                            help="categorical or Boolean condition column; repeat as needed")
        parser.add_argument("--window", dest="windows", action="append", type=int, default=[],
                            help="allowed rolling window in rows; repeat as needed")
        parser.add_argument("--max-lag", dest="max_lag", type=int, default=None,
                            help="maximum temporal lag in rows")
        parser.add_argument("--run-length", dest="run_lengths", action="append",
                            type=int, default=[],
                            help="allowed sustained-predicate window; repeat as needed")
        parser.add_argument("--max-conjunction-terms", dest="max_conjunction_terms",
                            type=int, default=None)
        parser.add_argument("--advanced", action="store_true",
                            help="enable sustained, conjunction, and categorical definitions")
        parser.add_argument("--max-degree", dest="max_degree", type=int, default=0,
                            help="minimum nonlinear polynomial degree for profiled data")
        parser.add_argument("--proportional", action="store_true",
                            help="enable robust proportional equality")
        parser.add_argument("--aggregation", dest="agg_kinds", action="append", default=[],
                            choices=["SUM", "MIN", "MAX", "AVG"],
                            help="allowed family aggregation; repeat as needed")
        parser.add_argument("--config", default="",
                            help="optional YAML dataset/profile configuration")

    pd = sub.add_parser("discover", help="induce schema and enumerate invariants from observed data")
    pd.add_argument("--input", default="", help="optional CSV or pickle DataFrame path")
    pd.add_argument("--raw-input", dest="raw_input", default="",
                    help="optional related raw CSV for a derived GTIB input")
    add_profile_args(pd)
    pd.add_argument("--schema-backend", choices=available_inducer_backends(), default="subagent")
    pd.add_argument("--harness", choices=sorted(HARNESSES), default=harness_default,
                    help="agentic CLI harness for the subagent schema backend (copilot|codex|claude)")
    pd.add_argument("--entities", type=int, default=6)
    pd.add_argument("--snapshots", type=int, default=400)
    pd.add_argument("--noise", type=float, default=0.02)
    pd.add_argument("--tolerance", type=float, default=discovery_defaults.tolerance)
    pd.add_argument("--hold-rate", dest="hold_rate", type=float,
                    default=discovery_defaults.hold_rate_threshold)
    pd.add_argument("--ci-alpha", dest="ci_alpha", type=float, default=discovery_defaults.ci_alpha)
    pd.add_argument("--band-mode", dest="band_mode", choices=["adaptive", "global"],
                    default=discovery_defaults.band_mode)
    pd.add_argument("--max-complexity", dest="max_complexity", type=int,
                    default=search_defaults.max_complexity)
    pd.add_argument("--max-add-arity", dest="max_add_arity", type=int,
                    default=search_defaults.max_add_arity)
    pd.add_argument("--max-rules", dest="max_rules", type=int,
                    default=search_defaults.max_rules)
    pd.add_argument("--max-nonlinear-leaves", dest="max_nonlinear_leaves",
                    type=int, default=search_defaults.max_nonlinear_leaves,
                    help="fail-loud nonlinear leaf ceiling; 0 allows every declared leaf")
    pd.add_argument("--max-linear-leaves", dest="max_linear_leaves",
                    type=int, default=search_defaults.max_linear_leaves,
                    help="fail-loud scaled/additive leaf ceiling; 0 allows every declared leaf")
    pd.add_argument("--max-conditioned-rules", dest="max_conditioned_rules",
                    type=int, default=search_defaults.max_conditioned_rules,
                    help="explicit conditioned temporal rule cap; 0 is exhaustive")
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

    pp = sub.add_parser("precheck", help="preflight: verify the harness/subagent/key can induce schemas")
    pp.add_argument("--harness", choices=sorted(HARNESSES), default=harness_default)
    pp.add_argument("--schema-backend", choices=available_inducer_backends(), default="subagent")
    pp.set_defaults(func=cmd_precheck)

    pcal = sub.add_parser("calibrate",
                          help="tune knobs on proxies, discover on your dataset, report known-invariant recall")
    pcal.add_argument("--input", default="", help="CSV or pickle DataFrame path for your dataset")
    pcal.add_argument("--raw-input", dest="raw_input", default="",
                      help="optional related raw CSV for a derived GTIB input")
    add_profile_args(pcal)
    pcal.add_argument("--known", default="", help="known_invariants.yaml or .json path")
    pcal.add_argument("--max-complexity", dest="max_complexity", type=int,
                      default=search_defaults.max_complexity)
    pcal.add_argument("--max-add-arity", dest="max_add_arity", type=int,
                      default=search_defaults.max_add_arity)
    pcal.add_argument("--max-rules", dest="max_rules", type=int,
                      default=CalibrationConfig().max_rules)
    pcal.add_argument("--max-nonlinear-leaves", dest="max_nonlinear_leaves",
                      type=int, default=search_defaults.max_nonlinear_leaves,
                      help="fail-loud nonlinear leaf ceiling; 0 allows every declared leaf")
    pcal.add_argument("--max-linear-leaves", dest="max_linear_leaves",
                      type=int, default=search_defaults.max_linear_leaves,
                      help="fail-loud scaled/additive leaf ceiling; 0 allows every declared leaf")
    pcal.add_argument("--max-conditioned-rules", dest="max_conditioned_rules",
                      type=int, default=search_defaults.max_conditioned_rules,
                      help="explicit conditioned temporal rule cap; 0 is exhaustive")
    pcal.add_argument("--harness", choices=sorted(HARNESSES), default=harness_default)
    pcal.add_argument("--schema-backend", choices=available_inducer_backends(), default="subagent")
    pcal.add_argument("--seed", type=int, default=0)
    pcal.add_argument("--max-iterations", dest="max_iterations", type=int, default=0,
                      help="0 = run to completion (default)")
    pcal.add_argument("--validation-frac", dest="validation_frac", type=float, default=0.3)
    pcal.add_argument("--ci-alpha", dest="ci_alpha", type=float,
                      default=discovery_defaults.ci_alpha)
    pcal.add_argument("--tolerance", type=float, default=None,
                      help="initial proxy-grid tolerance")
    pcal.add_argument("--hold-rate", dest="hold_rate", type=float, default=None,
                      help="initial proxy-grid hold-rate threshold")
    pcal.add_argument("--band-mode", dest="band_mode", choices=["adaptive", "global"],
                      default="global",
                      help="one fixed global tolerance (default) or a per-candidate self-calibrated band")
    pcal.add_argument("--max-capability-tiers", dest="max_capability_tiers", type=int, default=5,
                      help="grammar re-induction tiers (widen aggregations/degree) when recall stalls")
    pcal.add_argument("--name", default="", help="dataset name for the saved rules file (defaults to the input filename)")
    pcal.add_argument("--rules-dir", dest="rules_dir", default="rules",
                      help="directory for the saved learned-invariants .dl file (default: rules)")
    pcal.add_argument("--no-save-rules", dest="no_save_rules", action="store_true",
                      help="do not write the learned invariants to a .dl file (they still appear in --out)")
    pcal.add_argument("--out", default="", help="optional path to write the JSON report")
    pcal.set_defaults(func=cmd_calibrate)
    return p


def _enforce_capability_rule_budget(args: argparse.Namespace) -> None:
    advanced_possible = bool(getattr(args, "advanced", False))
    if getattr(args, "cmd", "") == "calibrate":
        from .calibrate import _capability_tiers

        advanced_possible |= int(
            getattr(args, "max_capability_tiers", 0) or 0
        ) >= len(_capability_tiers())
    if not advanced_possible:
        return
    limit = int(getattr(args, "max_rules", 0) or 0)
    if limit <= 0:
        raise ValueError(
            "advanced discovery requires a finite --max-rules budget "
            "(set it directly or through --config)"
        )
    if limit > 500_000:
        raise ValueError("advanced --max-rules must be <= 500000")


def main(argv: Optional[list] = None) -> int:
    # Explicit rules use math glyphs (Σ, ≠); force UTF-8 stdout so printing them never crashes on
    # a legacy Windows code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    args = _apply_config_file(args, raw_argv)
    _enforce_capability_rule_budget(args)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
