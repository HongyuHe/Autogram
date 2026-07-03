"""Publish layer (item 7 + calibration): fast tests that need no live model."""

from __future__ import annotations

import json

from autogram.discovery import regime as R
from autogram.discovery.known import KnownInvariant, _signature, load_known
from autogram.calibrate import CalibrationConfig, _split_known, precheck


def test_regime_default_and_edits():
    rs = R.default_regime()
    assert len(rs.active_entries()) == len(R.KNOWN_SHAPES)
    rs.deactivate("two_end")
    assert all(e.shape != "two_end" for e in rs.active_entries())
    rs.add("row_sum", noise=0.05)
    assert any(e.shape == "row_sum" and e.noise == 0.05 for e in rs.entries)


def test_regime_generate_plants_shape():
    data = R.generate(R.ProxyEntry("self_zero", n_entities=3, n_snapshots=40))
    assert "self_zero" in data.planted


def test_abstract_from_shapes_covers_known():
    rs = R.abstract_from_shapes(["row_sum", "presence_pair", "not_a_shape"])
    shapes = {e.shape for e in rs.entries}
    assert shapes == {"row_sum", "presence_pair"}


def test_known_signatures():
    assert _signature(KnownInvariant("a", "~=", "x", {"sum": ["y", "z"]}))[0] == "ref_sum"
    assert _signature(KnownInvariant("a", "==", "x", "y"))[0] == "pair"
    assert _signature(KnownInvariant("a", "==", "x", 0))[0] == "zero"
    assert _signature(KnownInvariant("a", "<|>", "x", "y"))[0] == "presence_pair"
    assert _signature(KnownInvariant("a", ">=", "x", 0))[0] == "one_sided"


def test_load_known_json(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"invariants": [{"name": "n", "op": ">=", "lhs": "c", "rhs": 0}]}))
    k = load_known(str(p))
    assert len(k) == 1 and k[0].op == ">=" and k[0].lhs == "c"


def test_precheck_unknown_harness_fails():
    assert precheck(harness="nope", backend="subagent")["ok"] is False


def test_split_known_holds_out_disjoint():
    known = [KnownInvariant(str(i), ">=", f"c{i}", 0) for i in range(10)]
    calib, valid = _split_known(known, 0.3, 0)
    assert len(valid) >= 1
    assert len(calib) + len(valid) == 10
    assert not ({id(x) for x in calib} & {id(x) for x in valid})


def test_calibration_config_defaults_to_run_to_completion():
    assert CalibrationConfig().max_iterations == 0
