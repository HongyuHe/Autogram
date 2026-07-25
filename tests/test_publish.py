"""Publish layer (item 7 + calibration): fast tests that need no live model."""

from __future__ import annotations

import json

from autogram.discovery import regime as R
from autogram.discovery.known import (
    KnownInvariant, _signature, abstract_shapes, load_known, shapes_for_invariant,
)
from autogram.calibrate import CalibrationConfig, _split_known, precheck


def test_shapes_for_invariant_maps_each_relation_form():
    # Every KnownInvariant relation form maps to the expected generic proxy shape set, using
    # only the op/rhs structure (no domain-specific words in the variable names).
    assert shapes_for_invariant(KnownInvariant("a", "==", "x", "y")) == ["two_end"]
    assert shapes_for_invariant(KnownInvariant("a", "~=", "x", "y")) == ["offset_pair"]
    assert shapes_for_invariant(KnownInvariant("a", "~=", "x", {"sum": ["y", "z"]})) == ["row_sum", "col_sum"]
    assert shapes_for_invariant(KnownInvariant("a", "==", "x", {"sum": ["y"]})) == ["row_sum", "col_sum"]
    assert shapes_for_invariant(KnownInvariant("a", "==", "x", 0)) == ["self_zero"]
    assert shapes_for_invariant(KnownInvariant("a", "~=", "x", 0)) == ["self_zero"]
    assert shapes_for_invariant(KnownInvariant("a", "<|>", "x", "y")) == ["presence_pair"]
    assert shapes_for_invariant(KnownInvariant("a", ">=", "x", 0)) == ["nonneg"]
    assert shapes_for_invariant(KnownInvariant("a", "<=", "x", 0)) == ["nonpos"]
    # a reference-vs-sum never abstracts to agg_ref_balance (the file format cannot express it)
    assert "agg_ref_balance" not in shapes_for_invariant(
        KnownInvariant("a", "==", "x", {"sum": ["y", "z"]}))


def test_abstract_shapes_unions_first_seen_order_and_dedupes():
    known = [
        KnownInvariant("a", "~=", "x", "y"),
        KnownInvariant("b", "~=", "p", "q"),          # duplicate offset_pair
        KnownInvariant("c", "==", "m", "n"),
        KnownInvariant("d", ">=", "z", 0),
    ]
    assert abstract_shapes(known) == ["offset_pair", "two_end", "nonneg"]


def test_abstract_from_shapes_empty_when_no_shape_maps():
    # The automatic path must NOT silently fall back to the full suite when nothing maps.
    rs = R.abstract_from_shapes(["not_a_shape", "also_bad"])
    assert rs.entries == []
    assert rs.active_entries() == []


def test_known_shapes_include_explicit_one_sided_proxies():
    assert "nonneg" in R.KNOWN_SHAPES and "nonpos" in R.KNOWN_SHAPES


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
