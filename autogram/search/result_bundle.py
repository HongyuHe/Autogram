"""Machine-readable run artifacts: per-candidate JSONL trace + a structured result bundle.

Motivated by ``docs/autogram_poc_eval.md`` (recommendations 1 and 3): an independent
evaluator could not audit search dynamics or recall without re-instrumenting the loop, and
the run emitted only human-readable console text.  This module turns a finished
:class:`~autogram.search.loop.RunResult` (plus the recall report) into two artifacts:

* ``trace_<dataset>.jsonl`` -- one JSON object per *evaluated candidate*, carrying the
  iteration, island, origin (proposer / grammar seed / anti seed / random bootstrap /
  mutation / seed-mutation / random), parent signature, archive-improved flag, verdict, and
  the full metric vector.  The loop builds the rows (:func:`autogram.search.loop.learn`);
  this module only serialises them.
* ``result_<dataset>.json`` -- a single bundle with the knob settings, the assembled
  portfolio, the recall scorecard against the known catalogue, run counts, and provenance
  (git commit + dirty flag + dataset shape + UTC timestamp).

Both are pure functions of already-computed objects (no learning, no oracle access beyond the
recall report the caller already built), so they are cheap and unit-testable without a live
backend.  All strings are ASCII so the files round-trip on a cp1252 Windows console.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from ..config import RunConfig
from ..dsl.parser import rule_to_dict
from ..loader.loader import Dataset
from .loop import RunResult
from .recall import RecallReport


# --------------------------------------------------------------------------- provenance

def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_provenance() -> dict:
    """Best-effort git commit hash + dirty flag (``None`` outside a repo / no git)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    dirty = None if status is None else bool(status.strip())
    return {"git_commit": commit, "git_dirty": dirty}


# --------------------------------------------------------------------------- serialisers

def _rule_record(r) -> dict:
    return {
        "rule": r.rule.unparse(),
        "rule_dict": rule_to_dict(r.rule),
        "sig": r.rule.signature(),
        "binder": r.rule.binder,
        "op": r.rule.atom.op,
        "complexity": r.rule.complexity(),
        "verdict": r.verdict.value,
        "reason": r.reason,
        "eps": round(r.eps, 6),
        "kappa_hat": round(r.kappa_hat, 6),
        "kappa_lo": round(r.kappa_lo, 6),
        "kappa_hi": round(r.kappa_hi, 6),
        "support": round(r.support, 6),
        "tightness": round(r.tightness, 6),
        "lift": round(r.lift, 6),
        "delta": round(r.delta, 6),
        "sigma_prop": round(r.sigma_prop, 6),
        "combined_score": round(r.combined_score, 6),
        "n_points": r.n_points,
        "n_bindings": r.n_bindings,
    }


def _recall_record(rep: RecallReport) -> dict:
    return {
        "n_targets": rep.n_targets,
        "n_recovered": rep.n_recovered,
        "n_full": rep.n_full,
        "recall": round(rep.recall, 6),
        "strict_recall": round(rep.strict_recall, 6),
        "matches": [
            {
                "tid": m.tid,
                "status": m.status,
                "label": m.label,
                "rule": (m.rule_summary.strip() if m.rule_summary else None),
                "note": m.note,
            }
            for m in rep.matches
        ],
    }


def recall_record(rep: RecallReport) -> dict:
    """Public wrapper so ``autogram score`` can graft a recall scorecard onto an
    already-written (unscored) result bundle without rebuilding the whole bundle."""
    return _recall_record(rep)


def build_result_bundle(
    rc: RunConfig,
    data_path: str,
    ds: Dataset,
    res: RunResult,
    rep: Optional[RecallReport] = None,
    *,
    used_real_subagent: Optional[bool] = None,
    proposer_notes: str = "",
) -> dict:
    """Assemble a JSON-serialisable run bundle (pure; no learning, no IO).

    ``rep`` is ``None`` for an *unscored* run (``autogram run --no-score``): the learning
    path never imports the oracle grader, so the bundle records ``recall: null`` and can be
    graded later by ``autogram score``.  This keeps leakage-freedom auditable -- an unscored
    bundle is provably produced without the ground-truth catalogue.
    """
    return {
        "schema": "autogram.result/v1",
        "dataset": ds.name,
        "data_path": data_path,
        "n_snapshots": ds.n_snapshots,
        "n_columns": len(ds.observed.names),
        "seed": rc.seed,
        "proposer": rc.grammar.proposer,
        "used_real_subagent": used_real_subagent,
        "proposer_notes": proposer_notes,
        "knobs": {
            "grammar": asdict(rc.grammar),
            "search": asdict(rc.search),
            "eval": asdict(rc.eval),
        },
        "counts": {
            "n_evaluated": res.n_evaluated,
            "n_unique": res.n_unique,
            "n_accepted": len(res.accepted),
            "portfolio_size": len(res.portfolio),
            "trace_rows": len(res.trace),
        },
        "recall": (None if rep is None else _recall_record(rep)),
        "grammar_extension": {
            "start": getattr(rc.grammar, "start", "full"),
            "rounds_applied": len(res.extensions_applied),
            "applied": res.extensions_applied,
            "final_binders": list(res.grammar.binders),
            "final_ref_roles": list(res.grammar.ref_roles),
            "final_fam_roles": list(res.grammar.fam_roles),
            "final_ops": list(res.grammar.ops),
            "final_agg_kinds": list(res.grammar.agg_kinds),
            "final_max_complexity": res.grammar.max_complexity,
            "final_max_add_arity": res.grammar.max_add_arity,
        },
        "portfolio": [_rule_record(r) for r in res.portfolio],
        "island_rates": [round(x, 6) for x in res.island_rates],
        "provenance": {
            **git_provenance(),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


def build_bench_bundle(
    dataset: str,
    data_path: str,
    seeds,
    per_seed,
    *,
    proposer: str,
    iters: int,
    targets,
) -> dict:
    """Aggregate a multi-seed benchmark (design: poc-eval recommendation 7).

    ``per_seed`` is a list of dicts ``{"seed", "strict_recall", "recall", "n_recovered",
    "hits"}`` where ``hits`` is the set/list of recovered target ids for that seed.  We report
    mean / variance / worst-case strict recall over the seeds and a per-target hit-rate (the
    fraction of seeds that recovered each known invariant), so the stochastic search behaviour
    is summarised rather than reported for a single lucky seed.
    """
    n = max(1, len(per_seed))
    strict = [float(s["strict_recall"]) for s in per_seed]
    soft = [float(s["recall"]) for s in per_seed]
    mean_strict = sum(strict) / n
    var_strict = sum((x - mean_strict) ** 2 for x in strict) / n
    mean_soft = sum(soft) / n
    hit_counts = {t: 0 for t in targets}
    for s in per_seed:
        for t in s.get("hits", []):
            if t in hit_counts:
                hit_counts[t] += 1
    per_target = {t: round(hit_counts[t] / n, 6) for t in targets}
    return {
        "schema": "autogram.bench/v1",
        "dataset": dataset,
        "data_path": data_path,
        "proposer": proposer,
        "iters": iters,
        "seeds": list(seeds),
        "n_runs": len(per_seed),
        "strict_recall": {
            "mean": round(mean_strict, 6),
            "var": round(var_strict, 6),
            "worst": round(min(strict), 6) if strict else 0.0,
            "best": round(max(strict), 6) if strict else 0.0,
        },
        "soft_recall": {"mean": round(mean_soft, 6)},
        "per_target_hit_rate": per_target,
        "per_seed": per_seed,
        "provenance": {
            **git_provenance(),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }


# --------------------------------------------------------------------------- writers

def write_trace_jsonl(path: str, trace) -> int:
    """Write one JSON object per trace row; returns the number of rows written."""
    n = 0
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        for row in trace:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def write_result_json(path: str, bundle: dict) -> None:
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        json.dump(bundle, fh, ensure_ascii=True, indent=2, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------------- rule files

def _filename_stamp(ts_iso: Optional[str] = None) -> str:
    """Return a filesystem-safe UTC stamp ``YYYYMMDDTHHMMSSZ`` for a rule filename.

    Derives the stamp from a bundle's ISO ``timestamp_utc`` when present so a back-filled
    file keeps the original run time; falls back to the current UTC time otherwise.
    """
    if ts_iso:
        try:
            return datetime.strptime(ts_iso, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%dT%H%M%SZ")
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _metric_comment(e: dict) -> str:
    """One-line verdict + metrics summary for a serialised portfolio rule record."""
    return (f"{e.get('verdict', '?'):<16s} "
            f"eps={float(e.get('eps', 0.0)):.4f} "
            f"kappa={float(e.get('kappa_hat', 0.0)):.3f} "
            f"supp={float(e.get('support', 0.0)):.2f} "
            f"lift={float(e.get('lift', 0.0)):.4g} "
            f"delta={float(e.get('delta', 0.0)):+.4f} "
            f"score={float(e.get('combined_score', 0.0)):.3f}").rstrip()


def render_rules_text(bundle: dict) -> str:
    """Render a learned-portfolio bundle as a human-readable ASCII rule listing.

    The learned invariants come first: each line is one invariant followed by a ``  #`` comment
    carrying its verdict and metrics, so ``grep -v '^#'`` yields the rule lines and splitting a
    line on ``  #`` yields the bare invariant expression.  A trailing ``#`` block then records
    metadata (dataset, run time, git provenance, proposer knobs, recall summary).
    """
    pf = bundle.get("portfolio", []) or []
    prov = bundle.get("provenance", {}) or {}
    rec = bundle.get("recall")
    ev = (bundle.get("knobs", {}) or {}).get("eval", {}) or {}

    rows = []  # (rule_text, metric_comment_or_None) -- tolerate plain-string entries too
    for e in pf:
        if isinstance(e, str):
            rows.append((e, None))
        else:
            rows.append((str(e.get("rule", "")), _metric_comment(e)))

    # Rule lines first (the substance of the file).
    rule_lines = []
    if rows:
        width = min(60, max(len(r) for r, _ in rows))
        for rule_text, comment in rows:
            rule_lines.append(f"{rule_text:<{width}s}  # {comment}" if comment else rule_text)
    else:
        rule_lines.append("# (empty portfolio -- no invariants were assembled)")

    commit = prov.get("git_commit")
    commit_short = commit[:8] if isinstance(commit, str) and commit else "unknown"
    dirty = prov.get("git_dirty")
    dirty_txt = "" if dirty is None else (" (dirty)" if dirty else " (clean)")

    # Metadata block trails the rules.
    meta = [
        "# Autogram learned invariants",
        f"# dataset    : {bundle.get('dataset', 'unknown')}",
        f"# run time   : {prov.get('timestamp_utc', 'unknown')} (UTC)",
        f"# git        : {commit_short}{dirty_txt}",
        (f"# proposer   : {bundle.get('proposer', 'unknown')}  "
         f"seed={bundle.get('seed', '?')}  "
         f"deployed={ev.get('deployed', '?')}  "
         f"rel_noise={ev.get('rel_noise', '?')}"),
    ]
    if isinstance(rec, dict):
        meta.append(
            f"# recall     : {100.0 * float(rec.get('recall', 0.0)):.2f}% form | "
            f"{100.0 * float(rec.get('strict_recall', 0.0)):.2f}% strict | "
            f"{rec.get('n_full', '?')}/{rec.get('n_targets', '?')} full")
    else:
        meta.append("# recall     : not scored (oracle-free run; grade with `autogram score`)")
    meta.append(f"# portfolio  : {len(rows)} invariant(s); "
                "each rule line is `<invariant>  # <verdict> <metrics>`")

    return "\n".join(rule_lines + [""] + meta) + "\n"


def write_rules_file(rules_dir: str, bundle: dict, *, stamp: Optional[str] = None) -> str:
    """Write ``render_rules_text(bundle)`` to ``rules_dir/<dataset>_<time>.pl``; return the path.

    ``time`` defaults to a filesystem-safe stamp derived from the bundle's own provenance
    timestamp (so a back-filled bundle keeps its original run time); pass ``stamp`` to override.
    """
    os.makedirs(rules_dir, exist_ok=True)
    dataset = str(bundle.get("dataset") or "dataset")
    when = stamp or _filename_stamp((bundle.get("provenance", {}) or {}).get("timestamp_utc"))
    path = os.path.join(rules_dir, f"{dataset}_{when}.pl")
    with open(path, "w", encoding="ascii", errors="replace") as fh:
        fh.write(render_rules_text(bundle))
    return path
