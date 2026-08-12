"""CLI smoke tests for v2."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from autogram.config import SearchConfig
from autogram.calibrate import CalibrationConfig
from autogram.cli import (
    _apply_config_file,
    _configure_dataframe_profile,
    _enforce_capability_rule_budget,
    _load_dataframe,
    _portfolio_payload,
    build_parser,
    cmd_validate,
    main,
)
from autogram.discovery import validate as V
from autogram.loader.gtib import AUTOGRAM_PROFILE_ATTR, profile_dataframe


def test_explicit_raw_input_is_authoritative_for_renamed_derived_csv(tmp_path):
    # An explicit --raw-input must drive the GTIB derived+raw join even when the derived CSV has
    # been renamed away from the conventional `timeseries_derived.csv` sibling-detection filename.
    derived = tmp_path / "my_derived.csv"
    raw = tmp_path / "my_raw.csv"
    times = pd.date_range("2026-01-01", periods=2, freq="1min")
    pd.DataFrame({
        "timestamp": times,
        "consumer_id": ["consumer_000", "consumer_000"],
        "minute_index": [0, 1],
        "input_rate_bytes_per_min": [float("nan"), float("nan")],
        "output_rate_bytes_per_min": [float("nan"), float("nan")],
        "backlog_bytes": [float("nan"), float("nan")],
        "cum_lost_bytes": [float("nan"), float("nan")],
    }).to_csv(derived, index=False)
    raw_rows = []
    counter = 0.0
    for step in range(12):
        counter += 10.0
        raw_rows.append({
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=10 * step),
            "consumer_id": "consumer_000",
            "shard_id": "s0",
            "collector_input_counted": counter,
            "presenter_output_counted": counter,
            "missing_flag": False,
            "reset_flag": False,
            "backlog_bytes": float(step),
            "cum_lost_bytes": float(step),
        })
    pd.DataFrame(raw_rows).to_csv(raw, index=False)

    frame = _load_dataframe(str(derived), str(raw))

    profile = frame.attrs.get(AUTOGRAM_PROFILE_ATTR, {})
    assert profile.get("related_aggregates"), "explicit --raw-input was ignored for a renamed CSV"


def test_discover_payload_includes_canonical_rule_payloads():
    from autogram.dsl import ast as A
    from autogram.dsl.parser import rule_from_dict

    rule = A.Rule("record", A.Compare(A.Ref("x"), ">=", A.Const(0)))
    evaluation = SimpleNamespace(
        rule=rule,
        strictness="one-sided",
        hold_rate=1.0,
        hold_rate_lo=1.0,
        hold_rate_hi=1.0,
        eps=0.0,
        mdl_gain=0.0,
        support=1.0,
        n_bindings=1,
        parameters={},
    )
    res = SimpleNamespace(
        dataset=SimpleNamespace(name="d", name_model=None),
        rounds_run=1,
        progress_history=[1.0],
        diagnostics=[],
        portfolio=[evaluation],
    )

    payload = _portfolio_payload(res)
    entry = payload["portfolio"][0]
    assert "rule_payload" in entry
    assert rule_from_dict(entry["rule_payload"]).unparse() == entry["rule"]

    # With provenance supplied, the discover payload carries the normalized runtime spec and
    # source/config fingerprints for auditability, consistent with calibration reports.
    enriched = _portfolio_payload(
        res,
        provenance={"engine_source_sha256": "abc", "input_sha256": "def"},
        normalized_spec={"name": "d", "patterns": []},
        effective={"discovery": {}, "search": {}},
    )
    assert enriched["provenance"]["engine_source_sha256"] == "abc"
    assert "normalized_spec" in enriched and "normalized_spec_sha256" in enriched
    assert "effective_settings" in enriched


def test_parser_wires_subcommands():
    p = build_parser()
    args = p.parse_args(["discover", "--entities", "4", "--seed", "1"])
    assert args.cmd == "discover" and args.entities == 4 and args.seed == 1


def test_validate_exit_code_requires_every_null_class_to_be_clean(
    monkeypatch,
):
    base = {
        "proxy_ok": True,
        "synthetic_recovery": {"ok": True},
        "null_equalities_accepted": 0,
        "null_temporal_accepted": 0,
        "null_definitions_accepted": 0,
    }
    monkeypatch.setattr(
        "autogram.cli.run_all",
        lambda seed=0: dict(base),
    )

    assert cmd_validate(SimpleNamespace(seed=0)) == 0

    for field in (
        "null_equalities_accepted",
        "null_temporal_accepted",
        "null_definitions_accepted",
    ):
        report = dict(base)
        report[field] = 1
        monkeypatch.setattr(
            "autogram.cli.run_all",
            lambda seed=0, report=report: report,
        )
        assert cmd_validate(SimpleNamespace(seed=0)) == 1


def test_parser_exposes_generic_temporal_and_condition_profile_flags():
    args = build_parser().parse_args([
        "discover",
        "--input",
        "series.csv",
        "--time-index",
        "time",
        "--group-key",
        "tenant",
        "--condition-column",
        "label",
        "--window",
        "45",
        "--window",
        "60",
        "--max-lag",
        "60",
        "--run-length",
        "10",
        "--advanced",
    ])
    assert args.time_index == "time"
    assert args.group_keys == ["tenant"]
    assert args.condition_columns == ["label"]
    assert args.windows == [45, 60]
    assert args.max_lag == 60
    assert args.run_lengths == [10]
    assert args.advanced is True


def test_parser_exposes_evaluation_settings_for_discover_and_calibrate():
    discover = build_parser().parse_args([
        "discover",
        "--band-mode",
        "global",
        "--ci-alpha",
        "0.1",
    ])
    calibrate = build_parser().parse_args([
        "calibrate",
        "--ci-alpha",
        "0.1",
    ])

    assert discover.band_mode == "global"
    assert discover.ci_alpha == 0.1
    assert calibrate.ci_alpha == 0.1


def test_checked_in_gtib_config_populates_calibration_arguments():
    args = build_parser().parse_args([
        "calibrate",
        "--config",
        "configs/gtib.yaml",
    ])

    configured = _apply_config_file(
        args,
        ["calibrate", "--config", "configs/gtib.yaml"],
    )

    assert configured.input == "data/gtib-emulation/timeseries_derived.csv"
    assert configured.raw_input == "data/gtib-emulation/timeseries_raw.csv"
    assert configured.known == "configs/gtib_known.yaml"
    assert configured.time_index == "timestamp"
    assert configured.group_keys == ["consumer_id"]
    assert configured.windows == [10, 45, 60]
    assert configured.max_lag == 45
    assert configured.run_lengths == [10]
    assert configured.max_capability_tiers == 5
    assert configured.max_rules == 400_000
    assert configured.max_nonlinear_leaves == 64
    assert configured.max_linear_leaves == 32
    assert configured.max_conditioned_rules == 0
    assert configured.advanced is True
    assert configured.max_degree == 2
    assert configured.proportional is True
    assert configured.agg_kinds == ["SUM"]
    assert configured.ci_alpha == 0.05
    assert configured.tolerance == 0.05
    assert configured.hold_rate == 0.62


def test_cli_aggregation_override_reaches_dataframe_profile():
    args = build_parser().parse_args([
        "discover",
        "--input",
        "series.csv",
        "--aggregation",
        "MAX",
    ])
    frame = _configure_dataframe_profile(
        pd.DataFrame({"value": [1.0, 2.0]}),
        args,
    )

    assert frame.attrs["autogram_profile"]["agg_kinds"] == ["MAX"]


def test_unrelated_profile_override_preserves_conjunction_arity():
    base = profile_dataframe(
        pd.DataFrame({"value": [1.0, 2.0]}),
        advanced=True,
        max_conjunction_terms=4,
    )
    unrelated = build_parser().parse_args([
        "discover",
        "--input",
        "series.csv",
        "--aggregation",
        "SUM",
    ])
    explicit = build_parser().parse_args([
        "discover",
        "--input",
        "series.csv",
        "--max-conjunction-terms",
        "2",
    ])

    preserved = _configure_dataframe_profile(base, unrelated)
    overridden = _configure_dataframe_profile(base, explicit)

    assert preserved.attrs["autogram_profile"]["max_conjunction_terms"] == 4
    assert overridden.attrs["autogram_profile"]["max_conjunction_terms"] == 2


def test_advanced_discovery_requires_finite_rule_budget():
    args = build_parser().parse_args(["discover", "--advanced"])
    with pytest.raises(ValueError, match="finite --max-rules"):
        _enforce_capability_rule_budget(args)


def test_automatic_advanced_calibration_rejects_explicit_unbounded_budget():
    args = build_parser().parse_args([
        "calibrate",
        "--input",
        "series.csv",
        "--known",
        "known.yaml",
        "--max-rules",
        "0",
    ])

    with pytest.raises(ValueError, match="finite --max-rules"):
        _enforce_capability_rule_budget(args)


def test_default_calibration_has_a_finite_advanced_rule_ceiling():
    args = build_parser().parse_args([
        "calibrate",
        "--input",
        "series.csv",
        "--known",
        "known.yaml",
    ])

    assert args.max_rules == 500_000
    _enforce_capability_rule_budget(args)


def test_cli_harness_defaults_follow_the_environment(monkeypatch):
    monkeypatch.setenv("AUTOGRAM_SUBAGENT_HARNESS", "claude")

    parser = build_parser()

    assert parser.parse_args(["discover"]).harness == "claude"
    assert parser.parse_args(["precheck"]).harness == "claude"
    assert parser.parse_args([
        "calibrate",
        "--input",
        "series.csv",
        "--known",
        "known.yaml",
    ]).harness == "claude"
    assert CalibrationConfig().harness == "claude"


def test_cli_default_search_bound_matches_proxy_runtime_bound():
    args = build_parser().parse_args(["discover"])
    tuned = V.proxy_tune(seed=0)
    assert args.max_complexity == SearchConfig().max_complexity
    assert args.max_add_arity == SearchConfig().max_add_arity
    assert tuned["runtime_search"].max_complexity == args.max_complexity
    assert tuned["runtime_search"].max_add_arity == args.max_add_arity


def test_discover_command_runs_and_writes_json(tmp_path):
    out = tmp_path / "portfolio.json"
    rules_dir = tmp_path / "rules"
    rc = main(["discover", "--entities", "5", "--snapshots", "160", "--seed", "0",
               "--hold-rate", "0.9", "--json", str(out), "--rules-dir", str(rules_dir)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["portfolio"]
    assert payload["portfolio"][0]["hold_rate"] >= 0.9
    pls = list(rules_dir.glob("*.dl"))
    assert pls, "discover should write a .dl rules file"
    text = pls[0].read_text(encoding="utf-8")
    assert "[forall" in text
    assert "Autogram discovered invariants" in text


def test_discover_no_save_rules(tmp_path):
    rules_dir = tmp_path / "rules"
    rc = main(["discover", "--entities", "5", "--snapshots", "120", "--seed", "0",
               "--hold-rate", "0.9", "--rules-dir", str(rules_dir), "--no-save-rules"])
    assert rc == 0
    assert not rules_dir.exists() or not list(rules_dir.glob("*.dl"))


def test_clean_command(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    (d / "x.json").write_text("{}")
    rc = main(["clean", "--out", str(d)])
    assert rc == 0
    assert not d.exists()
