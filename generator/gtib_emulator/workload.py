"""Workload generator: the byte arrival process feeding the Collector.

Produces, per consumer and per shard, an array of *bytes arriving per raw step*
(the 10 s scrape grid). Two archetypes are modelled:

* ``steady``    -- diurnal baseline with mild log-normal jitter.
* ``bursty_ml`` -- diurnal baseline modulated by an ON/OFF burst process plus
  periodic "all-reduce" synchronization spikes, mimicking ML training traffic.

Grounding
---------
* Per-consumer aggregate rates are heavy-tailed (log-normal), matching data
  center flow-size fits [PAPER: Benson et al., IMC 2010; VL2, SIGCOMM 2009].
* ON/OFF burstiness at short timescales is a well-documented data-center
  property [PAPER: Benson IMC 2010]. ML workloads add coordinated, periodic
  bursts from gradient synchronization, which we model as recurring spikes with
  straggler jitter.

Everything here is deterministic given the seeded RNG and is pure with respect
to the physical pipeline (no loss, no measurement effects yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EmulatorConfig


@dataclass
class Consumer:
    """Static description of one tenant and its Presenter shards."""

    consumer_id: str
    archetype: str                        # "steady" | "bursty_ml"
    base_rate_bytes_per_s: float          # mean input rate (heavy-tailed across consumers)
    shard_ids: list[str]
    shard_weights: np.ndarray             # fraction of traffic per shard, sums to 1
    eta_meas: float                       # healthy output-counter accounting factor (~0.98)

    @property
    def n_shards(self) -> int:
        return len(self.shard_ids)


def build_consumers(cfg: EmulatorConfig, rng: np.random.Generator) -> list[Consumer]:
    """Create the tenant population with heavy-tailed base rates and shard splits."""

    consumers: list[Consumer] = []
    n_bursty = int(round(cfg.scale.n_consumers * cfg.scale.fraction_bursty))
    archetypes = ["bursty_ml"] * n_bursty + ["steady"] * (cfg.scale.n_consumers - n_bursty)
    rng.shuffle(archetypes)

    for i in range(cfg.scale.n_consumers):
        base_rate = float(rng.lognormal(cfg.scale.base_rate_log_mu, cfg.scale.base_rate_log_sigma))
        n_shards = int(rng.integers(cfg.scale.shards_min, cfg.scale.shards_max + 1))
        # Dirichlet split gives an uneven but normalised share per shard.
        weights = rng.dirichlet(np.full(n_shards, 3.0))
        # Healthy accounting factor per consumer (Q-C/Q-D). Typical healthy ratio
        # sits above the alert threshold; the good-data threshold is the floor.
        eta = float(np.clip(
            rng.normal(cfg.pipeline.healthy_ratio_mean, cfg.pipeline.healthy_ratio_std),
            cfg.alerting.good_data_threshold, 1.0))
        consumers.append(Consumer(
            consumer_id=f"consumer_{i:03d}",
            archetype=archetypes[i],
            base_rate_bytes_per_s=base_rate,
            shard_ids=[f"shard_{i:03d}_{s}" for s in range(n_shards)],
            shard_weights=weights,
            eta_meas=eta,
        ))
    return consumers


def _diurnal(cfg: EmulatorConfig, t_seconds: np.ndarray) -> np.ndarray:
    """Multiplicative diurnal seasonality in [1 - amp, 1 + amp]."""

    period = cfg.workload.diurnal_period_hours * 3600.0
    phase = 2.0 * np.pi * t_seconds / period
    return 1.0 + cfg.workload.diurnal_amplitude * np.sin(phase)


def _on_off_multiplier(cfg: EmulatorConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Two-state ON/OFF Markov chain -> burst multiplier per raw step.

    OFF state multiplier == 1; ON state multiplier == burst_amplitude. Short,
    clustered ON periods reproduce data-center burstiness.
    """

    on = np.zeros(n, dtype=bool)
    state = False
    for k in range(n):
        if state:
            if rng.random() < cfg.workload.off_prob:
                state = False
        else:
            if rng.random() < cfg.workload.on_prob:
                state = True
        on[k] = state
    mult = np.ones(n)
    mult[on] = cfg.workload.burst_amplitude
    return mult


def _allreduce_multiplier(cfg: EmulatorConfig, t_seconds: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Periodic synchronized spikes (ML all-reduce), with straggler jitter.

    Each sync tick contributes a narrow Gaussian bump; ticks are jittered so the
    spikes are not perfectly aligned (stragglers).
    """

    period = cfg.workload.allreduce_period_seconds
    if period <= 0 or cfg.workload.allreduce_amplitude <= 0:
        return np.ones_like(t_seconds)
    total = float(t_seconds[-1]) if len(t_seconds) else 0.0
    mult = np.ones_like(t_seconds)
    width = max(cfg.time.raw_scrape_seconds, period * 0.03)  # narrow bump
    n_ticks = int(total // period) + 1
    for j in range(n_ticks):
        centre = j * period + rng.normal(0.0, cfg.workload.allreduce_jitter_seconds)
        bump = np.exp(-0.5 * ((t_seconds - centre) / width) ** 2)
        mult += cfg.workload.allreduce_amplitude * bump
    return mult


def arrivals_for_consumer(cfg: EmulatorConfig, consumer: Consumer,
                          rng: np.random.Generator) -> np.ndarray:
    """Bytes arriving per shard per raw step.

    Returns an array of shape ``(n_shards, n_raw_steps)`` of non-negative bytes.
    """

    n = cfg.n_raw_steps
    dt = cfg.time.raw_scrape_seconds
    t_seconds = np.arange(n, dtype=float) * dt

    season = _diurnal(cfg, t_seconds)
    noise = rng.lognormal(0.0, cfg.workload.noise_log_sigma, size=n)

    if consumer.archetype == "bursty_ml":
        burst = _on_off_multiplier(cfg, n, rng) * _allreduce_multiplier(cfg, t_seconds, rng)
    else:
        burst = np.ones(n)

    # Bytes per raw step for the whole consumer, then split across shards.
    rate = consumer.base_rate_bytes_per_s * season * burst * noise  # bytes/sec
    per_step_total = np.clip(rate * dt, 0.0, None)                   # bytes/step

    # Per-shard split. Shard weights are stable here; shard churn (membership
    # changes over time) is applied later in the measurement layer.
    arrivals = np.outer(consumer.shard_weights, per_step_total)     # (n_shards, n)
    return arrivals


def baseline_for_consumer(cfg: EmulatorConfig, consumer: Consumer) -> np.ndarray:
    """Burst-free provisioned demand per shard per raw step.

    This is the deterministic diurnal baseline (no burst modulation, no noise)
    used to size the Presenter's service capacity in the queue model, so that
    short bursts build real backlog rather than being instantly absorbed.
    """

    n = cfg.n_raw_steps
    dt = cfg.time.raw_scrape_seconds
    t_seconds = np.arange(n, dtype=float) * dt
    season = _diurnal(cfg, t_seconds)
    per_step_total = np.clip(consumer.base_rate_bytes_per_s * season * dt, 0.0, None)
    return np.outer(consumer.shard_weights, per_step_total)
