"""CLI smoke tests for v2."""

from __future__ import annotations

import json

from autogram.config import SearchConfig
from autogram.cli import build_parser, main
from autogram.discovery import validate as V


def test_parser_wires_subcommands():
    p = build_parser()
    args = p.parse_args(["discover", "--entities", "4", "--seed", "1"])
    assert args.cmd == "discover" and args.entities == 4 and args.seed == 1


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
    text = pls[0].read_text()
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
