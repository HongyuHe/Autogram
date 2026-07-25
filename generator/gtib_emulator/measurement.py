"""Measurement layer: physical fluid state -> observed, imperfect counters.

The alerting system never sees the physical byte balance; it sees two scraped,
monotonic counters per Presenter task (shard). This layer injects the benign
imperfections the email calls out as causes of *false* ratio dips "not actual
persistent byte loss" [STATED], plus operational realism:

* **Healthy accounting offset** -- the observed output counter is scaled by a
  per-consumer factor ``eta_meas`` (~0.98) so presenter/collector "won't 100%
  match" [STATED]. This is a *metering* factor, not a physical loss, so the
  physical conservation invariant is untouched.
* **Scrape noise** -- tiny multiplicative jitter on each counter increment.
* **Alignment artifact** -- during rapid traffic change (and in explicit artifact
  spans) a fraction of an output increment is deferred to the next scrape,
  modelling input/output scrape phase misalignment. It conserves bytes exactly
  over two steps, so it produces a transient dip that recovers -- never net loss.
* **Missing samples** -- occasional scrape gaps (emitted as NaN).
* **Counter resets** -- task restarts reset a shard's monotonic counters to 0.
* **Shard churn** -- optional; shards may stop emitting mid-run (default off).

Internal gRPC/Channelz queue metrics are intentionally *not* emitted by default
(Q-J, "make hidden for now"); enable via ``measurement.emit_channel_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EmulatorConfig
from .pipeline import PhysicalResult
from .workload import Consumer


@dataclass
class ObservedResult:
    """Per-shard observed counters and quality flags (shape ``(n_shards, n)``)."""

    input_counted: np.ndarray        # observed collector_input_counted (may hold NaN)
    output_counted: np.ndarray       # observed presenter_output_counted (may hold NaN)
    missing_flag: np.ndarray         # bool: scrape gap at this step
    reset_flag: np.ndarray           # bool: counter reset at this step
    active_flag: np.ndarray          # bool: shard alive/emitting at this step


def _increments(cumulative: np.ndarray) -> np.ndarray:
    """Per-step non-negative increments of a cumulative series (axis=1)."""

    inc = np.diff(cumulative, axis=1, prepend=0.0)
    return np.clip(inc, 0.0, None)


def measure(cfg: EmulatorConfig, consumer: Consumer, phys: PhysicalResult,
            artifact_mask: np.ndarray, rng: np.random.Generator) -> ObservedResult:
    """Render observed counters for one consumer from its physical result."""

    m = cfg.measurement
    n_shards, n = phys.cum_input.shape
    dt = cfg.time.raw_scrape_seconds

    inc_in = _increments(phys.cum_input)
    inc_out = _increments(phys.cum_output_physical) * consumer.eta_meas  # accounting offset

    # --- multiplicative scrape noise --- #
    if m.counter_noise_rel > 0:
        inc_in *= 1.0 + rng.normal(0.0, m.counter_noise_rel, size=inc_in.shape)
        inc_out *= 1.0 + rng.normal(0.0, m.counter_noise_rel, size=inc_out.shape)
        inc_in = np.clip(inc_in, 0.0, None)
        inc_out = np.clip(inc_out, 0.0, None)

    # --- alignment artifact: defer a fraction of each output increment --- #
    agg_in = inc_in.sum(axis=0)
    baseline = np.median(agg_in[agg_in > 0]) if np.any(agg_in > 0) else 1.0
    rel_change = np.abs(np.diff(agg_in, prepend=agg_in[:1])) / max(baseline, 1.0)
    shift = np.clip(m.alignment_artifact_strength * rel_change, 0.0, 0.9)
    shift[artifact_mask] = np.maximum(shift[artifact_mask], 0.35)
    carry = np.zeros(n_shards)
    for t in range(n):
        moved = inc_out[:, t] * shift[t]
        inc_out[:, t] = inc_out[:, t] - moved + carry
        carry = moved
    inc_out[:, -1] += carry            # flush residual so bytes are conserved
    inc_out = np.clip(inc_out, 0.0, None)

    s_in = np.cumsum(inc_in, axis=1)
    s_out = np.cumsum(inc_out, axis=1)

    # --- counter resets (per shard, resets BOTH counters of that task) --- #
    reset_flag = np.zeros((n_shards, n), dtype=bool)
    p_reset = m.counter_reset_prob_per_hour * dt / 3600.0
    out_in = s_in.copy()
    out_out = s_out.copy()
    for sh in range(n_shards):
        base_in = base_out = 0.0
        for t in range(n):
            if t > 0 and p_reset > 0 and rng.random() < p_reset:
                base_in, base_out = s_in[sh, t - 1], s_out[sh, t - 1]
                reset_flag[sh, t] = True
            out_in[sh, t] = s_in[sh, t] - base_in
            out_out[sh, t] = s_out[sh, t] - base_out

    # --- shard churn: optional early death (shard stops emitting) --- #
    active = np.ones((n_shards, n), dtype=bool)
    p_churn = m.shard_churn_prob_per_hour * dt / 3600.0
    if p_churn > 0 and n_shards > 1:
        for sh in range(n_shards):
            if rng.random() < p_churn * n:  # rough per-run death probability
                death = int(rng.integers(n // 2, n))
                active[sh, death:] = False

    # --- missing samples (scrape gaps) --- #
    missing = rng.random((n_shards, n)) < m.missing_sample_prob

    out_in = np.round(out_in)
    out_out = np.round(out_out)
    out_in[missing | ~active] = np.nan
    out_out[missing | ~active] = np.nan

    return ObservedResult(out_in, out_out, missing, reset_flag, active)
