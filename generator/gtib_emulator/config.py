"""Configuration schema for the gTIB byte-completeness emulator.

Every tunable knob lives here as a typed dataclass field with a documented
default. Defaults are chosen to be either:

* **Anchored** -- fixed by a fact stated in the email thread (e.g. 10 s scrape,
  1 min rate, 1 h smoothing, healthy ratio ~0.98, catch-up ratio ~20x), or
* **Assumed** -- an educated guess grounded in published measurements where
  possible (data-center / ML traffic: Benson et al., IMC 2010; VL2, SIGCOMM
  2009; DCTCP, SIGCOMM 2010), and clearly marked as such.

A user config file (YAML) overlays these defaults; only the keys the user sets
are overridden (see :func:`load_config`). The *effective* config is snapshotted
into the run manifest so any generated dataset is fully reproducible.

Provenance tags used in the comments below:
    [STATED]  -- taken directly from the email thread
    [ASSUMED] -- our modelling choice / default, tunable
    [PAPER]   -- default informed by published literature
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import yaml


# --------------------------------------------------------------------------- #
# Time grid                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class TimeConfig:
    """Simulation clock and the windows used by the alerting math."""

    duration_hours: float = 6.0          # [ASSUMED] start small; long enough for many 1 h windows
    raw_scrape_seconds: int = 10         # [STATED] "raw metric is scraped at 10s interval"
    rate_window_seconds: int = 60        # [STATED] "pre-computed results ... at 1m interval"
    smoothing_window_seconds: int = 3600  # [STATED] "one-hour smoothing window"
    start_timestamp: str = "2026-01-01T00:00:00Z"  # [ASSUMED] arbitrary epoch for readable output


# --------------------------------------------------------------------------- #
# Scale (Q-H: "make configurable and start small")                            #
# --------------------------------------------------------------------------- #
@dataclass
class ScaleConfig:
    """How many consumers / shards and how their base rates are spread.

    The gTIB pipeline is multi-tenant [STATED]. Per-consumer aggregate byte
    rates are heavy-tailed: a few "whale" tenants dominate volume while most are
    small. We model the spread with a log-normal, consistent with the log-normal
    flow-size fits reported for data-center traffic [PAPER: Benson IMC 2010; VL2].
    """

    n_consumers: int = 6                 # [ASSUMED] start small; scale up freely
    shards_min: int = 1                  # [ASSUMED] Presenter tasks per consumer (SUM is over these)
    shards_max: int = 4                  # [ASSUMED]
    # Per-consumer mean input rate ~ LogNormal(mu, sigma) in bytes/second.
    # mu=13.8 -> median ~1 MB/s; sigma=1.0 -> heavy tail (whales an order of
    # magnitude above the median). Shape follows published log-normal fits
    # [PAPER: Benson IMC 2010, VL2 SIGCOMM 2009]; absolute scale is [ASSUMED].
    base_rate_log_mu: float = 13.8
    base_rate_log_sigma: float = 1.0
    # Fraction of consumers that are bursty ML-style tenants vs steady tenants.
    fraction_bursty: float = 0.4         # [ASSUMED] "ML related workloads ... different traffic" [STATED]


# --------------------------------------------------------------------------- #
# Workload / arrivals (Q-F: ML-burst stats, paper-backed defaults)            #
# --------------------------------------------------------------------------- #
@dataclass
class WorkloadConfig:
    """Arrival process feeding the Collector.

    Two archetypes:

    * ``steady``   -- gentle log-normal jitter around a diurnal baseline.
    * ``bursty_ml`` -- ON/OFF Markov modulation (data-center ON/OFF burstiness
      [PAPER: Benson IMC 2010]) combined with *periodic* synchronized spikes that
      mimic ML all-reduce / gradient-sync phases [PAPER: ML training traffic is
      periodic + coordinated with heavy-tailed burst sizes].

    All shape parameters are knobs so they can be recalibrated once real burst
    statistics (amplitude, inter-arrival, duration) are available.

    Note on separation of concerns: the *intrinsic* burstiness here is kept mild
    -- it is normal traffic variation that must NOT by itself push the 1 h ratio
    below the alert threshold. The large, threshold-crossing ML/all-reduce spikes
    that trigger the reported false positives are injected as *labelled*
    ``benign_burst`` events (see ``anomalies.py``), so the ground truth stays clean.
    """

    # Seasonality (diurnal). Amplitude is fraction of base rate.
    diurnal_amplitude: float = 0.30      # [ASSUMED]
    diurnal_period_hours: float = 24.0   # [ASSUMED]
    # Multiplicative per-step noise (log-normal), applies to every archetype.
    noise_log_sigma: float = 0.08        # [ASSUMED] within-flow variability

    # --- bursty_ml archetype: ON/OFF modulation (mild, unlabelled variation) --- #
    on_prob: float = 0.05                # [ASSUMED] P(enter ON burst) per raw step
    off_prob: float = 0.30               # [ASSUMED] P(leave ON burst) per raw step -> short bursts
    burst_amplitude: float = 2.5         # [ASSUMED] mild ON multiplier; big spikes are labelled events
    # --- bursty_ml archetype: periodic all-reduce spikes (mild) --- #
    allreduce_period_seconds: float = 300.0  # [ASSUMED] synchronized sync cadence
    allreduce_amplitude: float = 2.0     # [ASSUMED] extra multiplier at each sync tick
    allreduce_jitter_seconds: float = 20.0   # [ASSUMED] stragglers de-synchronize the spike


# --------------------------------------------------------------------------- #
# Pipeline / fluid-queue model (Q-C healthy ratio, Q-E catch-up)              #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """The black box between Collector and Presenter, as a byte-conserving queue.

    Conservation held per shard at every step::

        input == output_physical + backlog + true_loss

    The Presenter is normally *faster* than the Collector [STATED], so backlog
    drains quickly; after a burst it can drain far faster than the input,
    producing the reported catch-up overshoot ("ratio shoot up to 20") [STATED].
    """

    # Service margin: mean service capacity as a multiple of mean arrival rate.
    # >1 so the Presenter keeps up and backlog mean-reverts to ~0. [STATED dir.]
    service_margin: float = 2.0          # [ASSUMED] value; direction is [STATED]
    # Max drain rate while clearing backlog, as a multiple of the baseline input
    # rate. Governs the post-burst overshoot peak of the 1 min ratio. [STATED ~20]
    catch_up_max_ratio: float = 20.0     # [STATED] "ratio shoot up to 20"
    # How aggressively backlog is drained: capacity grows by drain_gain * (backlog
    # / baseline) above the service margin, capped at catch_up_max_ratio. Small
    # values make a burst drain gradually (a multi-minute dip + graduated
    # overshoot); large values drain almost instantly. [ASSUMED]
    drain_gain: float = 0.5              # [ASSUMED]
    baseline_ema_halflife_seconds: float = 1800.0  # [ASSUMED] fallback baseline smoothing

    # Healthy accounting offset (Q-C / Q-D). presenter/collector "won't 100%
    # match" due to pipeline delay [STATED]. Two stated numbers must be
    # reconciled: the rule alerts at ratio < 0.99, and Tao cites a 0.98
    # "good-data threshold". Since a rule that alerts below 0.99 does NOT fire
    # constantly in production, healthy smoothed ratio must sit *above* 0.99
    # (near 1.0, a hair below due to delay), with 0.98 as the tolerance floor.
    # So we model the *typical healthy* ratio as ~0.995 [ASSUMED reconciliation,
    # Q-C/Q-D] and keep 0.98/0.99 as separate configurable thresholds
    # (see AlertingConfig). This offset is a per-consumer *measurement*
    # calibration factor on the OUTPUT counter -- it does NOT remove physical
    # bytes, so physical conservation is untouched. The distribution is a knob so
    # it can be fit to real healthy-ratio statistics later.
    healthy_ratio_mean: float = 0.998    # [ASSUMED] typical healthy ratio (comfortably > 0.99 alert)
    healthy_ratio_std: float = 0.0015    # [ASSUMED] tight; real spread is open question Q-C


# --------------------------------------------------------------------------- #
# Measurement layer                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class MeasurementConfig:
    """Turns the physical fluid state into observed, imperfect counters.

    Produces the *benign* measurement/alignment artifacts that the email says
    cause transient ratio dips "not actual persistent byte loss" [STATED].
    """

    counter_noise_rel: float = 0.0005    # [ASSUMED] tiny multiplicative scrape noise
    # Alignment artifact: during rapid traffic change, input vs output scrapes are
    # slightly out of phase, injecting mean-zero *anti-correlated* wobble into the
    # observed increments (nets to zero -> no real byte loss). [STATED cause]
    alignment_artifact_strength: float = 0.10  # [ASSUMED]
    missing_sample_prob: float = 0.002   # [ASSUMED] scrape gaps -> NaN raw sample
    counter_reset_prob_per_hour: float = 0.05  # [ASSUMED] task restart resets a monotonic counter
    shard_churn_prob_per_hour: float = 0.0     # [ASSUMED] 0 by default; >0 lets shards join/leave

    # Q-J: internal gRPC/Channelz queue metrics -- "make hidden for now".
    emit_channel_metrics: bool = False   # [STATED preference] keep hidden by default


# --------------------------------------------------------------------------- #
# Alerting rule under study (Q-D thresholds)                                   #
# --------------------------------------------------------------------------- #
@dataclass
class AlertingConfig:
    """The current static rule, kept configurable so 0.98 vs 0.99 is explicit.

    The rule doc alerts at ``ratio < 0.99 FOR 1h``; Tao separately cites a 0.98
    good-data threshold and a "one-hour smoothing window to avoid noise". How to
    map "FOR 1 hour" onto (1 h smoothing + short confirmation) vs (short smoothing
    + 1 h sustain) is genuinely ambiguous (open question Q-D), so both the
    threshold and the post-smoothing confirmation duration are knobs. Here the
    alert signal is the 1 h-smoothed ratio (the noise filter Tao describes) and
    ``alert_duration_minutes`` is a short confirmation on top of it. Both the
    good-data floor (0.98) and the alert threshold (0.99) are exposed; the floor
    is used only to decide the ground-truth oracle, never to tune the rule.
    """

    alert_threshold: float = 0.99        # [STATED] rule doc: ALERT IF ratio < 0.99
    alert_duration_minutes: int = 10     # [ASSUMED interp, Q-D] confirmation on the 1h-smoothed ratio
    good_data_threshold: float = 0.98    # [STATED] Tao: 0.98 threshold for good data


# --------------------------------------------------------------------------- #
# Anomaly injection (the labelled ground truth)                               #
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyTypeConfig:
    """Rate and severity bounds for one injected event family.

    ``count_per_consumer`` events are placed at random non-overlapping spans.
    Severity is the fractional byte deficit delta (e.g. 0.03 == 3% under-count).
    """

    count_per_consumer: float = 0.0
    min_severity: float = 0.01
    max_severity: float = 0.05
    min_duration_minutes: int = 30
    max_duration_minutes: int = 120


@dataclass
class AnomalyConfig:
    """Catalogue of injectable behaviours. Patterns to catch are [STATED]:
    persistent 1-5% under-accounting, sudden cliffs, and shard-scoped partial
    loss; the *creeping* (soft ramp) family is an [ASSUMED] extension for
    threshold-grazing robustness.
    """

    # --- true-loss families (MUST alert) --- #
    persistent: AnomalyTypeConfig = field(default_factory=lambda: AnomalyTypeConfig(
        count_per_consumer=0.5, min_severity=0.01, max_severity=0.05,   # [STATED] 1-5%
        min_duration_minutes=70, max_duration_minutes=140))
    cliff: AnomalyTypeConfig = field(default_factory=lambda: AnomalyTypeConfig(
        count_per_consumer=0.25, min_severity=0.30, max_severity=0.90,  # a task dies -> big drop
        min_duration_minutes=60, max_duration_minutes=120))
    partial: AnomalyTypeConfig = field(default_factory=lambda: AnomalyTypeConfig(
        count_per_consumer=0.25, min_severity=0.20, max_severity=0.60,  # shard-scoped, diluted in SUM
        min_duration_minutes=70, max_duration_minutes=140))
    creeping: AnomalyTypeConfig = field(default_factory=lambda: AnomalyTypeConfig(
        count_per_consumer=0.25, min_severity=0.02, max_severity=0.08,  # ramps 0 -> severity
        min_duration_minutes=90, max_duration_minutes=160))

    # --- benign families (must NOT alert) --- #
    # For benign_burst, min/max_severity are reused as the burst *amplitude*
    # range (multiplicative), not a loss fraction (benign bursts lose no bytes).
    benign_burst: AnomalyTypeConfig = field(default_factory=lambda: AnomalyTypeConfig(
        count_per_consumer=1.0, min_severity=6.0, max_severity=14.0,     # amplitude, not loss
        min_duration_minutes=10, max_duration_minutes=40))

    # Detection SLA recorded on true-loss events (minutes). [STATED] "minutes vs ~1 hour".
    detection_sla_minutes: int = 60


# --------------------------------------------------------------------------- #
# Output                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class OutputConfig:
    """Where and how the labelled dataset is written."""

    directory: str = "output"
    fmt: str = "csv"                     # "csv" or "parquet" (parquet needs pyarrow)
    include_hidden_state: bool = True    # backlog_bytes / cum_lost_bytes ground-truth columns
    write_raw: bool = True               # per-shard 10 s counters
    write_derived: bool = True           # per-consumer 1 min derived signals + labels
    write_plot: bool = False             # optional quick-look PNG per consumer (needs matplotlib)


# --------------------------------------------------------------------------- #
# Top-level config                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class EmulatorConfig:
    """Root configuration object threaded through the whole generator."""

    seed: int = 20260101                 # deterministic runs
    time: TimeConfig = field(default_factory=TimeConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- convenience derived quantities -- #
    @property
    def n_raw_steps(self) -> int:
        return int(round(self.time.duration_hours * 3600 / self.time.raw_scrape_seconds))

    @property
    def raw_steps_per_minute(self) -> int:
        return int(round(self.time.rate_window_seconds / self.time.raw_scrape_seconds))

    @property
    def minutes_per_smoothing_window(self) -> int:
        return int(round(self.time.smoothing_window_seconds / self.time.rate_window_seconds))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# YAML loading with recursive overlay                                          #
# --------------------------------------------------------------------------- #
def _overlay(obj: Any, patch: Mapping[str, Any], path: str = "") -> Any:
    """Recursively overlay ``patch`` (from YAML) onto a dataclass instance.

    Unknown keys raise, so typos in a config file fail loudly rather than being
    silently ignored.
    """

    if not is_dataclass(obj):
        return copy.deepcopy(patch)
    valid = {f.name: f for f in fields(obj)}
    for key, value in patch.items():
        where = f"{path}.{key}" if path else key
        if key not in valid:
            raise KeyError(f"Unknown config key: '{where}'. Valid keys here: {sorted(valid)}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _overlay(current, value, where)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | None = None, overrides: Mapping[str, Any] | None = None) -> EmulatorConfig:
    """Build an :class:`EmulatorConfig` from defaults, an optional YAML file, and
    an optional in-memory ``overrides`` mapping (applied last)."""

    cfg = EmulatorConfig()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"Config file {path} must contain a top-level mapping.")
        _overlay(cfg, data)
    if overrides:
        _overlay(cfg, overrides)
    _validate(cfg)
    return cfg


def _validate(cfg: EmulatorConfig) -> None:
    """Cheap sanity checks that catch obviously broken configs early."""

    if cfg.time.duration_hours <= 0:
        raise ValueError("time.duration_hours must be > 0")
    if cfg.time.rate_window_seconds % cfg.time.raw_scrape_seconds != 0:
        raise ValueError("rate_window_seconds must be a multiple of raw_scrape_seconds")
    if cfg.time.smoothing_window_seconds % cfg.time.rate_window_seconds != 0:
        raise ValueError("smoothing_window_seconds must be a multiple of rate_window_seconds")
    if cfg.scale.shards_min < 1 or cfg.scale.shards_max < cfg.scale.shards_min:
        raise ValueError("require 1 <= shards_min <= shards_max")
    if not (0.0 < cfg.pipeline.healthy_ratio_mean <= 1.0):
        raise ValueError("pipeline.healthy_ratio_mean must be in (0, 1]")
    if cfg.pipeline.service_margin <= 1.0:
        raise ValueError("pipeline.service_margin must be > 1 (Presenter faster than Collector)")
    if cfg.output.fmt not in ("csv", "parquet"):
        raise ValueError("output.fmt must be 'csv' or 'parquet'")
