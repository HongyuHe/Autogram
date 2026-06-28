"""CLI smoke tests: the console entrypoint stays importable and the subcommands run offline."""

from __future__ import annotations

import json

from autogram.cli import build_parser, main


def test_parser_wires_subcommands():
    p = build_parser()
    args = p.parse_args(["discover", "--entities", "4", "--seed", "1"])
    assert args.cmd == "discover" and args.entities == 4 and args.seed == 1


def test_discover_command_runs_and_writes_json(tmp_path):
    out = tmp_path / "portfolio.json"
    rules_dir = tmp_path / "rules"
    rc = main(["discover", "--entities", "5", "--snapshots", "200", "--rounds", "5",
               "--proposals", "110", "--permutations", "10", "--seed", "0",
               "--json", str(out), "--rules-dir", str(rules_dir)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["portfolio"]
    assert payload["portfolio"][0]["lift"] > 1.0
    # discovered invariants are also persisted to ./rules as a timestamped .pl file
    pls = list(rules_dir.glob("*.pl"))
    assert pls, "discover should write a .pl rules file"
    text = pls[0].read_text()
    assert "[forall" in text
    assert "# Autogram discovered invariants" in text


def test_discover_no_save_rules(tmp_path):
    rules_dir = tmp_path / "rules"
    rc = main(["discover", "--entities", "5", "--snapshots", "150", "--rounds", "4",
               "--proposals", "90", "--permutations", "8", "--seed", "0",
               "--rules-dir", str(rules_dir), "--no-save-rules"])
    assert rc == 0
    assert not rules_dir.exists() or not list(rules_dir.glob("*.pl"))


def test_clean_command(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir()
    (d / "x.json").write_text("{}")
    rc = main(["clean", "--out", str(d)])
    assert rc == 0
    assert not d.exists()
