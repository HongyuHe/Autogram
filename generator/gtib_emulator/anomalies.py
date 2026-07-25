"""Anomaly injector -- the source of the labelled ground truth.

For each consumer this module schedules a set of non-overlapping events and
turns them into three per-consumer control signals consumed downstream:

* ``arrival_multiplier[t]``   -- benign traffic bursts (>= 1), applied to arrivals
  *before* the queue, so they create real dip+overshoot dynamics with no byte loss.
* ``loss_fraction[shard, t]`` -- fraction of a shard's available bytes that are
  *truly lost* this step (removed from the physical byte balance).
* ``artifact_mask[t]``        -- spans where a benign measurement/alignment
  artifact is active (handled in the measurement layer).

Every event is also emitted as a row of the event catalogue (``events.csv``)
with an ``expected_alert`` oracle. Because the same event object drives both the
signal and the label, the labels are correct by construction -- there is no
post-hoc labelling to get wrong.

Event families (patterns to catch are [STATED]; ``creeping`` is an [ASSUMED]
soft/threshold-grazing extension)::

    true-loss  : persistent | cliff | partial | creeping   -> expected_alert = True
    benign     : benign_burst | artifact                   -> expected_alert = False
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnomalyTypeConfig, EmulatorConfig
from .workload import Consumer

TRUE_LOSS_TYPES = ("persistent", "cliff", "partial", "creeping")
BENIGN_TYPES = ("benign_burst", "artifact")


@dataclass
class Event:
    """One labelled span in the ground-truth catalogue."""

    event_id: str
    consumer_id: str
    type: str
    shard_scope: str                 # "all" or a comma-joined subset of shard ids
    span_start_step: int
    span_end_step: int
    severity_pct: float              # peak fractional byte deficit (0 for benign)
    recovering: bool
    mechanism: str
    detection_sla_minutes: int
    expected_alert: bool
    start_timestamp: str = ""        # filled in during output assembly
    end_timestamp: str = ""


@dataclass
class ConsumerSchedule:
    """Per-consumer control signals plus the events that produced them."""

    consumer_id: str
    events: list[Event]
    arrival_multiplier: np.ndarray   # shape (n_raw_steps,), >= 1
    loss_fraction: np.ndarray        # shape (n_shards, n_raw_steps), in [0, 1]
    artifact_mask: np.ndarray        # shape (n_raw_steps,), bool


def _place_span(occupied: list[tuple[int, int]], length: int, n: int,
                rng: np.random.Generator, margin: int) -> tuple[int, int] | None:
    """Try to place a span of ``length`` steps not overlapping ``occupied``.

    A ``margin`` of clear steps is kept on each side so the 1 h smoothing window
    does not blend adjacent events. Returns ``(start, end)`` or ``None``.
    """

    if length >= n:
        return None
    for _ in range(40):
        start = int(rng.integers(0, n - length))
        end = start + length
        clash = any(not (end + margin <= s or start - margin >= e) for s, e in occupied)
        if not clash:
            occupied.append((start, end))
            return start, end
    return None


def _n_events(rate: float, rng: np.random.Generator) -> int:
    if rate <= 0:
        return 0
    return int(rng.poisson(rate))


def build_schedule(cfg: EmulatorConfig, consumer: Consumer,
                   rng: np.random.Generator) -> ConsumerSchedule:
    """Schedule all events for one consumer and render the control signals."""

    n = cfg.n_raw_steps
    steps_per_min = 60 // cfg.time.raw_scrape_seconds
    margin = int(cfg.time.smoothing_window_seconds / cfg.time.raw_scrape_seconds)  # ~1 window
    sla = cfg.anomaly.detection_sla_minutes

    arrival_multiplier = np.ones(n)
    loss_fraction = np.zeros((consumer.n_shards, n))
    artifact_mask = np.zeros(n, dtype=bool)
    occupied: list[tuple[int, int]] = []
    events: list[Event] = []
    counter = 0

    def minutes_to_steps(minutes: int) -> int:
        return max(1, int(minutes) * steps_per_min)

    def make_span(tc: AnomalyTypeConfig) -> tuple[int, int] | None:
        dur_min = int(rng.integers(tc.min_duration_minutes, tc.max_duration_minutes + 1))
        return _place_span(occupied, minutes_to_steps(dur_min), n, rng, margin)

    # ---- true-loss families -------------------------------------------------
    families = {
        "persistent": cfg.anomaly.persistent,
        "cliff": cfg.anomaly.cliff,
        "partial": cfg.anomaly.partial,
        "creeping": cfg.anomaly.creeping,
    }
    for etype, tc in families.items():
        for _ in range(_n_events(tc.count_per_consumer, rng)):
            span = make_span(tc)
            if span is None:
                continue
            start, end = span
            severity = float(rng.uniform(tc.min_severity, tc.max_severity))

            if etype == "partial":
                # Loss scoped to a random subset of shards -> diluted in the SUM.
                k = max(1, consumer.n_shards // 2)
                idx = rng.choice(consumer.n_shards, size=k, replace=False)
                loss_fraction[idx, start:end] = severity
                scope = ",".join(consumer.shard_ids[i] for i in sorted(idx))
                mechanism = "shard_subset_loss"
                recovering = False
            elif etype == "cliff":
                # One shard "dies": abrupt, large, sustained loss on that shard.
                i = int(rng.integers(0, consumer.n_shards))
                loss_fraction[i, start:end] = severity
                scope = consumer.shard_ids[i]
                mechanism = "task_death"
                recovering = False
            elif etype == "creeping":
                # Soft ramp 0 -> severity across the span (threshold-grazing).
                ramp = np.linspace(0.0, severity, end - start)
                loss_fraction[:, start:end] = ramp[None, :]
                scope = "all"
                mechanism = "creeping_skim"
                recovering = False
            else:  # persistent
                loss_fraction[:, start:end] = severity
                scope = "all"
                mechanism = "persistent_skim"
                recovering = False

            counter += 1
            events.append(Event(
                event_id=f"{consumer.consumer_id}_ev{counter:03d}",
                consumer_id=consumer.consumer_id, type=f"true_loss_{etype}",
                shard_scope=scope, span_start_step=start, span_end_step=end,
                severity_pct=round(severity * 100, 3), recovering=recovering,
                mechanism=mechanism, detection_sla_minutes=sla, expected_alert=True))

    # ---- benign burst -------------------------------------------------------
    # Bursty ML tenants get proportionally more benign bursts than steady ones.
    tc = cfg.anomaly.benign_burst
    burst_rate = tc.count_per_consumer * (2.0 if consumer.archetype == "bursty_ml" else 1.0)
    for _ in range(_n_events(burst_rate, rng)):
        span = make_span(tc)
        if span is None:
            continue
        start, end = span
        # For benign_burst, the severity fields carry the burst amplitude range.
        amp_lo = tc.min_severity if tc.min_severity > 1.0 else 6.0
        amp_hi = tc.max_severity if tc.max_severity > amp_lo else amp_lo + 8.0
        amp = float(rng.uniform(amp_lo, amp_hi))
        arrival_multiplier[start:end] *= amp
        counter += 1
        events.append(Event(
            event_id=f"{consumer.consumer_id}_ev{counter:03d}",
            consumer_id=consumer.consumer_id, type="benign_burst",
            shard_scope="all", span_start_step=start, span_end_step=end,
            severity_pct=0.0, recovering=True, mechanism="traffic_burst",
            detection_sla_minutes=sla, expected_alert=False))

    # ---- measurement/alignment artifact ------------------------------------
    # A handful of short artifact spans (benign wobble, no real loss).
    for _ in range(_n_events(0.5, rng)):
        length = int(rng.integers(2, 6))  # a few raw steps
        span = _place_span(occupied, length, n, rng, margin=steps_per_min)
        if span is None:
            continue
        start, end = span
        artifact_mask[start:end] = True
        counter += 1
        events.append(Event(
            event_id=f"{consumer.consumer_id}_ev{counter:03d}",
            consumer_id=consumer.consumer_id, type="artifact",
            shard_scope="all", span_start_step=start, span_end_step=end,
            severity_pct=0.0, recovering=True, mechanism="scrape_misalignment",
            detection_sla_minutes=sla, expected_alert=False))

    events.sort(key=lambda e: e.span_start_step)
    return ConsumerSchedule(consumer.consumer_id, events, arrival_multiplier,
                            loss_fraction, artifact_mask)
