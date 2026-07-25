"""Orchestrator: run the whole generation pipeline and write the dataset.

Wires the components in order for every consumer::

    build_consumers
      -> arrivals (workload)
      -> schedule events (anomalies)  --benign burst multiplier--> arrivals
      -> simulate fluid queue (pipeline)         [physical byte balance]
      -> measure (observed counters + artifacts) [what alerting sees]
      -> derive (rates, ratio, 1h smoothing, static alert)
      -> attach per-minute ground-truth labels + oracle

then assembles the raw and derived tables, the event catalogue, an evaluation
of the static rule vs the oracle (the "reproduce-the-bug" check), and a
manifest, and writes them to disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .anomalies import ConsumerSchedule, build_schedule
from .config import EmulatorConfig
from .deriver import derive_consumer, trajectory_alert
from .labeling import events_frame, per_minute_labels
from .measurement import measure
from .pipeline import simulate
from .workload import build_consumers


@dataclass
class GenerateResult:
    config: EmulatorConfig
    raw: pd.DataFrame
    derived: pd.DataFrame
    events: pd.DataFrame
    records: list[dict[str, Any]]
    manifest: dict[str, Any]


def run(cfg: EmulatorConfig) -> GenerateResult:
    """Generate the full synthetic dataset in memory."""

    consumers = build_consumers(cfg, np.random.default_rng(cfg.seed))
    n_min = cfg.n_raw_steps // cfg.raw_steps_per_minute
    dt = cfg.time.raw_scrape_seconds
    start = np.datetime64(cfg.time.start_timestamp.replace("Z", ""))
    raw_ts = start + (np.arange(cfg.n_raw_steps) * dt).astype("timedelta64[s]")

    schedules: list[ConsumerSchedule] = []
    records: list[dict[str, Any]] = []
    raw_frames: list[pd.DataFrame] = []
    derived_frames: list[pd.DataFrame] = []

    for i, consumer in enumerate(consumers):
        rng = np.random.default_rng(cfg.seed + 1 + i)
        # Import here to avoid a circular import at module load.
        from .workload import arrivals_for_consumer, baseline_for_consumer
        arrivals = arrivals_for_consumer(cfg, consumer, rng)
        baseline = baseline_for_consumer(cfg, consumer)
        schedule = build_schedule(cfg, consumer, rng)
        arrivals = arrivals * schedule.arrival_multiplier[None, :]        # benign bursts

        phys = simulate(cfg, arrivals, schedule.loss_fraction, baseline)
        obs = measure(cfg, consumer, phys, schedule.artifact_mask, rng)
        frame = derive_consumer(cfg, consumer, obs, phys)
        labels = per_minute_labels(cfg, schedule, n_min)
        frame = pd.concat([frame.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
        frame["traj_alert"] = trajectory_alert(cfg, frame)

        schedules.append(schedule)
        records.append({"consumer": consumer, "phys": phys, "obs": obs, "frame": frame})
        derived_frames.append(frame)
        if cfg.output.write_raw:
            raw_frames.append(_raw_frame(cfg, consumer, obs, raw_ts))

    derived = pd.concat(derived_frames, ignore_index=True)
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    events = events_frame(cfg, schedules)

    evaluation = _evaluate(derived)
    manifest = _manifest(cfg, consumers, events, derived, evaluation)
    return GenerateResult(cfg, raw, derived, events, records, manifest)


def _raw_frame(cfg: EmulatorConfig, consumer, obs, raw_ts: np.ndarray) -> pd.DataFrame:
    """Long per-shard raw counter table for one consumer."""

    n_shards, n = obs.input_counted.shape
    frames = []
    for sh in range(n_shards):
        df = pd.DataFrame({
            "timestamp": raw_ts,
            "consumer_id": consumer.consumer_id,
            "shard_id": consumer.shard_ids[sh],
            "collector_input_counted": obs.input_counted[sh],
            "presenter_output_counted": obs.output_counted[sh],
            "missing_flag": obs.missing_flag[sh],
            "reset_flag": obs.reset_flag[sh],
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _evaluate(derived: pd.DataFrame) -> dict[str, Any]:
    """Compare static rule and trajectory rule against the oracle (per minute)."""

    valid = derived["completeness_ratio_1h"].notna()
    d = derived[valid]
    oracle = d["oracle_alert"].to_numpy()
    benign_min = (d["is_benign_burst"] | d["is_artifact"]).to_numpy()

    def scores(alert_col: str) -> dict[str, Any]:
        alert = d[alert_col].to_numpy()
        tp = int(np.sum(alert & oracle))
        fp = int(np.sum(alert & ~oracle))
        fn = int(np.sum(~alert & oracle))
        tn = int(np.sum(~alert & ~oracle))
        fp_on_benign = int(np.sum(alert & ~oracle & benign_min))
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "false_positives_on_benign_minutes": fp_on_benign,
                "precision": round(precision, 4) if precision == precision else None,
                "recall": round(recall, 4) if recall == recall else None}

    return {
        "evaluated_minutes": int(len(d)),
        "static_rule": scores("static_alert"),
        "trajectory_rule": scores("traj_alert"),
        "note": ("static_rule.false_positives_on_benign_minutes > 0 with "
                 "trajectory_rule fewer is the 'reproduce-the-bug' signal."),
    }


def _manifest(cfg: EmulatorConfig, consumers, events: pd.DataFrame,
              derived: pd.DataFrame, evaluation: dict[str, Any]) -> dict[str, Any]:
    normal = derived.loc[derived["label"] == "normal", "completeness_ratio_1h"]
    type_counts = (events["type"].value_counts().to_dict() if not events.empty else {})
    return {
        "generator": "gtib_emulator",
        "version": __version__,
        "seed": cfg.seed,
        "scale": {
            "n_consumers": len(consumers),
            "archetypes": {a: sum(c.archetype == a for c in consumers)
                           for a in ("steady", "bursty_ml")},
            "total_shards": int(sum(c.n_shards for c in consumers)),
            "n_raw_steps": cfg.n_raw_steps,
            "n_minutes": int(cfg.n_raw_steps // cfg.raw_steps_per_minute),
        },
        "event_counts": type_counts,
        "summary_stats": {
            "normal_ratio_1h_median": _round(np.nanmedian(normal)) if len(normal) else None,
            "normal_ratio_1h_p05": _round(np.nanpercentile(normal, 5)) if len(normal) else None,
            "normal_ratio_1h_p95": _round(np.nanpercentile(normal, 95)) if len(normal) else None,
            "max_completeness_ratio_1m": _round(np.nanmax(derived["completeness_ratio"].to_numpy())),
        },
        "evaluation": evaluation,
        "config": cfg.to_dict(),
    }


def _round(x: Any) -> Any:
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #
def write_outputs(result: GenerateResult, out_dir: str | None = None) -> dict[str, str]:
    """Write raw/derived/events/manifest to disk. Returns the paths written."""

    cfg = result.config
    out_dir = out_dir or cfg.output.directory
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, str] = {}

    derived = result.derived
    if not cfg.output.include_hidden_state:
        derived = derived.drop(columns=[c for c in ("backlog_bytes", "cum_lost_bytes") if c in derived])

    if cfg.output.write_derived:
        written["derived"] = _write_table(derived, os.path.join(out_dir, "timeseries_derived"), cfg.output.fmt)
    if cfg.output.write_raw and not result.raw.empty:
        written["raw"] = _write_table(result.raw, os.path.join(out_dir, "timeseries_raw"), cfg.output.fmt)
    written["events"] = _write_table(result.events, os.path.join(out_dir, "events"), "csv")

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(result.manifest, fh, indent=2, default=str)
    written["manifest"] = manifest_path
    return written


def _write_table(df: pd.DataFrame, base: str, fmt: str) -> str:
    if fmt == "parquet":
        try:
            path = base + ".parquet"
            df.to_parquet(path, index=False)
            return path
        except Exception:  # pragma: no cover - pyarrow missing / write error
            pass  # fall back to CSV below
    path = base + ".csv"
    df.to_csv(path, index=False)
    return path
