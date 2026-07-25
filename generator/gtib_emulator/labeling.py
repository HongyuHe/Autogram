"""Ground-truth labelling: events -> per-minute masks + the alert oracle.

Two products:

* :func:`per_minute_labels` -- for one consumer, boolean masks per minute
  (``is_true_loss``, ``is_benign_burst``, ``is_artifact``), a single ``label``
  string (priority: true_loss > benign_burst > artifact > normal), and the
  ``oracle_alert`` (True iff real byte loss is present that minute -- i.e. a
  true-loss event covers it). The oracle is what a *correct* detector should do;
  it is deliberately independent of the static rule so evaluation is honest.

* :func:`events_frame` -- the event catalogue (``events.csv``), one row per
  injected event with absolute timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anomalies import ConsumerSchedule
from .config import EmulatorConfig


def _step_to_minute(step: int, spm: int) -> int:
    return step // spm


def per_minute_labels(cfg: EmulatorConfig, schedule: ConsumerSchedule, n_min: int) -> pd.DataFrame:
    """Per-minute label masks and oracle for one consumer."""

    spm = cfg.raw_steps_per_minute
    is_true_loss = np.zeros(n_min, dtype=bool)
    is_benign = np.zeros(n_min, dtype=bool)
    is_artifact = np.zeros(n_min, dtype=bool)

    for ev in schedule.events:
        m0 = _step_to_minute(ev.span_start_step, spm)
        m1 = min(n_min, _step_to_minute(ev.span_end_step - 1, spm) + 1)
        if m1 <= m0:
            continue
        if ev.type.startswith("true_loss"):
            is_true_loss[m0:m1] = True
        elif ev.type == "benign_burst":
            is_benign[m0:m1] = True
        elif ev.type == "artifact":
            is_artifact[m0:m1] = True

    label = np.full(n_min, "normal", dtype=object)
    label[is_artifact] = "artifact"
    label[is_benign] = "benign_burst"
    label[is_true_loss] = "true_loss"

    return pd.DataFrame({
        "is_true_loss": is_true_loss,
        "is_benign_burst": is_benign,
        "is_artifact": is_artifact,
        "label": label,
        "oracle_alert": is_true_loss,   # real byte loss present -> a correct detector fires
    })


def events_frame(cfg: EmulatorConfig, schedules: list[ConsumerSchedule]) -> pd.DataFrame:
    """Assemble the event catalogue with absolute timestamps."""

    dt = cfg.time.raw_scrape_seconds
    start = np.datetime64(cfg.time.start_timestamp.replace("Z", ""))
    rows = []
    for sched in schedules:
        for ev in sched.events:
            t0 = start + np.timedelta64(int(ev.span_start_step * dt), "s")
            t1 = start + np.timedelta64(int(ev.span_end_step * dt), "s")
            dur_min = (ev.span_end_step - ev.span_start_step) * dt / 60.0
            rows.append({
                "event_id": ev.event_id,
                "consumer_id": ev.consumer_id,
                "type": ev.type,
                "shard_scope": ev.shard_scope,
                "span_start": t0,
                "span_end": t1,
                "duration_minutes": round(dur_min, 2),
                "severity_pct": ev.severity_pct,
                "recovering": ev.recovering,
                "mechanism": ev.mechanism,
                "detection_sla_minutes": ev.detection_sla_minutes,
                "expected_alert": ev.expected_alert,
            })
    cols = ["event_id", "consumer_id", "type", "shard_scope", "span_start", "span_end",
            "duration_minutes", "severity_pct", "recovering", "mechanism",
            "detection_sla_minutes", "expected_alert"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["consumer_id", "span_start"]).reset_index(drop=True)
