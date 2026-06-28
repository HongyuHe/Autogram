"""Tests for the human-readable learned-rule file writer.

``render_rules_text`` / ``write_rules_file`` operate on a serialised result *bundle* (a plain
dict), so these tests construct minimal bundles directly -- no engine, no datasets, fully
offline.  They pin: the filename-stamp reformat + fallback, the metadata header + per-rule
metric comment, tolerance of plain-string portfolio entries, the unscored / empty cases, and
that ``write_rules_file`` materialises ``<rules_dir>/<dataset>_<stamp>.pl`` faithfully.
"""

import os

from autogram.search.result_bundle import (
    _filename_stamp,
    render_rules_text,
    write_rules_file,
)


def _bundle(**over):
    """A minimal scored result bundle with one rule; override any field via kwargs."""
    b = {
        "dataset": "abilene",
        "proposer": "subagent",
        "seed": 0,
        "knobs": {"eval": {"deployed": False, "rel_noise": 0.02}},
        "provenance": {
            "git_commit": "c49ecfc6deadbeef",
            "git_dirty": True,
            "timestamp_utc": "2026-06-23T17:44:03Z",
        },
        "recall": {
            "n_targets": 8, "n_recovered": 7, "n_full": 7,
            "recall": 0.875, "strict_recall": 0.875, "matches": [],
        },
        "portfolio": [
            {
                "rule": "[forall link] e[X->Y] >= 0",
                "verdict": "EXACT", "eps": 0.0, "kappa_hat": 1.0, "support": 1.0,
                "lift": 1e12, "delta": 0.0, "combined_score": 2.44,
            },
        ],
    }
    b.update(over)
    return b


def test_filename_stamp_reformats_iso():
    assert _filename_stamp("2026-06-23T17:44:03Z") == "20260623T174403Z"


def test_filename_stamp_fallback_on_bad_input():
    # Bad / missing input must not raise -- it falls back to a well-formed current-UTC stamp.
    for bad in (None, "", "not-a-timestamp", "2026/06/23 17:44"):
        out = _filename_stamp(bad)
        assert len(out) == len("20260623T174403Z") and out.endswith("Z")


def test_render_includes_header_and_rule_and_metrics():
    text = render_rules_text(_bundle())
    # Metadata header fields.
    assert "# dataset    : abilene" in text
    assert "2026-06-23T17:44:03Z" in text
    assert "c49ecfc6" in text and "(dirty)" in text
    assert "proposer   : subagent" in text and "deployed=False" in text
    assert "87.50% form" in text and "87.50% strict" in text and "7/8 full" in text
    # The rule line carries the invariant plus a `  #` metric comment.
    line = next(ln for ln in text.splitlines() if ln.startswith("[forall link]"))
    rule, _, comment = line.partition("  #")
    assert rule.strip() == "[forall link] e[X->Y] >= 0"
    assert "EXACT" in comment and "lift=1e+12" in comment and "score=2.440" in comment


def test_render_rule_lines_are_grep_extractable():
    text = render_rules_text(_bundle())
    rule_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(rule_lines) == 1
    # Splitting on the `  #` sentinel recovers the bare invariant expression.
    assert rule_lines[0].partition("  #")[0].strip() == "[forall link] e[X->Y] >= 0"


def test_render_tolerates_string_portfolio_entries():
    text = render_rules_text(_bundle(portfolio=["e[X->Y] ~= i[Y<-X]   EXACT eps=0.0010"]))
    assert "e[X->Y] ~= i[Y<-X]" in text


def test_render_unscored_and_empty_portfolio():
    text = render_rules_text(_bundle(recall=None, portfolio=[]))
    assert "not scored" in text
    assert "empty portfolio" in text


def test_write_rules_file_creates_named_file(tmp_path):
    rules_dir = os.path.join(str(tmp_path), "rules")
    path = write_rules_file(rules_dir, _bundle())
    assert os.path.basename(path) == "abilene_20260623T174403Z.pl"
    assert os.path.isfile(path)
    body = open(path, encoding="ascii").read()
    assert "[forall link] e[X->Y] >= 0" in body
    # Rules come first; the metadata block is a trailing footer.
    assert body.startswith("[forall link] e[X->Y] >= 0")
    assert "# Autogram learned invariants" in body
    assert body.index("[forall link] e[X->Y] >= 0") < body.index("# Autogram learned invariants")


def test_write_rules_file_explicit_stamp_override(tmp_path):
    path = write_rules_file(str(tmp_path), _bundle(dataset="geant"), stamp="20260101T000000Z")
    assert os.path.basename(path) == "geant_20260101T000000Z.pl"
