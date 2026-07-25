"""Fluid queue / server model -- the byte-conserving physical core.

Given per-shard arrivals, a burst-free baseline demand, and a per-shard
true-loss schedule, this simulates the "black box" between Collector and
Presenter as a fluid queue and produces the *physical* cumulative counters. One
invariant is enforced exactly at every step and every shard::

    cumulative_input == cumulative_output_physical + backlog + cumulative_true_loss

Service capacity is provisioned against the *baseline* demand (a slowly varying
quantity that tracks diurnal load but NOT short bursts), because a real Presenter
deployment is sized for typical load, not for instantaneous spikes::

    loss        = loss_fraction[t] * (backlog + arrivals[t])     # truly lost bytes (>= 0)
    deliverable = (backlog + arrivals[t]) - loss
    cap_ratio   = min(catch_up_max_ratio, service_margin + drain_gain * backlog / baseline[t])
    capacity    = cap_ratio * baseline[t]
    output      = min(deliverable, capacity)
    backlog'    = deliverable - output               # >= 0

The drain is *graduated*: capacity rises with backlog, so a burst builds real
backlog (a multi-minute dip) that drains over minutes with an overshoot that
grows with how much backlog accumulated -- approaching ``catch_up_max_ratio`` for
the largest bursts.

Consequences that reproduce the stated behaviour:

* **Healthy** (arrivals ~ baseline, no loss): capacity > arrivals, so backlog
  mean-reverts to ~0 and output ~ arrivals -> ratio ~ 1 (the healthy accounting
  offset that yields ~0.99+ is applied later, in the measurement layer).
* **Burst**: arrivals >> baseline for a while; capacity stays near
  ``service_margin * baseline`` so backlog builds and the 1 min ratio *dips*.
* **Post-burst catch-up**: once the spike passes, arrivals fall back to baseline
  while a large backlog remains; ``catch_up_max_ratio * baseline`` lets the queue
  drain far faster than the (now low) input, so the 1 min ratio *overshoots*
  toward ``catch_up_max_ratio`` -- the reported "ratio shoot up to 20" [STATED].
* **True loss**: ``loss`` removes bytes from the balance permanently, so the
  deficit accumulates and never recovers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EmulatorConfig


@dataclass
class PhysicalResult:
    """Per-shard physical time series (shape ``(n_shards, n_raw_steps)`` each)."""

    cum_input: np.ndarray            # cumulative arrivals (bytes)
    cum_output_physical: np.ndarray  # cumulative physically-processed bytes
    cum_true_loss: np.ndarray        # cumulative truly-lost bytes (ground truth)
    backlog: np.ndarray              # end-of-step backlog Q (bytes, >= 0)

    def conservation_residual(self) -> np.ndarray:
        """|input - (output + backlog + loss)| -- should be ~0 (float eps)."""

        resid = self.cum_input - (self.cum_output_physical + self.backlog + self.cum_true_loss)
        return np.abs(resid)


def _slow_baseline(cfg: EmulatorConfig, arrivals: np.ndarray) -> np.ndarray:
    """Fallback baseline: a slow EMA of arrivals (halflife ~30 min) that tracks
    diurnal load but smooths out short bursts. Used only when the caller does not
    supply an explicit burst-free baseline."""

    dt = cfg.time.raw_scrape_seconds
    half = cfg.pipeline.baseline_ema_halflife_seconds
    alpha = 1.0 - 0.5 ** (dt / half)
    n_shards, n = arrivals.shape
    ema = np.median(arrivals, axis=1)
    ema = np.where(ema > 0, ema, arrivals.mean(axis=1) + 1.0)
    out = np.empty_like(arrivals)
    for t in range(n):
        ema = (1.0 - alpha) * ema + alpha * arrivals[:, t]
        out[:, t] = ema
    return out


def simulate(cfg: EmulatorConfig, arrivals: np.ndarray, loss_fraction: np.ndarray,
             baseline: np.ndarray | None = None) -> PhysicalResult:
    """Run the fluid queue for one consumer.

    Parameters
    ----------
    arrivals
        ``(n_shards, n_raw_steps)`` bytes arriving per shard per step (benign
        burst multipliers already folded in by the caller).
    loss_fraction
        ``(n_shards, n_raw_steps)`` fraction of available bytes truly lost.
    baseline
        ``(n_shards, n_raw_steps)`` burst-free provisioned demand used to size the
        service capacity. If ``None``, a slow EMA of ``arrivals`` is used.
    """

    n_shards, n = arrivals.shape
    if baseline is None:
        baseline = _slow_baseline(cfg, arrivals)
    baseline = np.clip(baseline, 1.0, None)

    cum_in = np.zeros((n_shards, n))
    cum_out = np.zeros((n_shards, n))
    cum_loss = np.zeros((n_shards, n))
    backlog_out = np.zeros((n_shards, n))

    Q = np.zeros(n_shards)
    ci = np.zeros(n_shards)
    co = np.zeros(n_shards)
    cl = np.zeros(n_shards)

    svc = cfg.pipeline.service_margin
    catch = cfg.pipeline.catch_up_max_ratio
    gain = cfg.pipeline.drain_gain

    for t in range(n):
        a = arrivals[:, t]
        base = baseline[:, t]
        avail = Q + a
        loss = np.clip(loss_fraction[:, t], 0.0, 1.0) * avail
        deliverable = avail - loss
        # Graduated drain: capacity ramps up with backlog (up to the catch-up
        # cap), so a burst builds real backlog that drains over minutes.
        cap_ratio = np.minimum(catch, svc + gain * (Q / base))
        capacity = cap_ratio * base
        output = np.minimum(deliverable, capacity)
        output = np.clip(output, 0.0, None)
        Q = deliverable - output            # >= 0 by construction

        ci = ci + a
        co = co + output
        cl = cl + loss
        cum_in[:, t] = ci
        cum_out[:, t] = co
        cum_loss[:, t] = cl
        backlog_out[:, t] = Q

    return PhysicalResult(cum_in, cum_out, cum_loss, backlog_out)
