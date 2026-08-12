"""Deriver: reproduce the exact signals the alerting system computes.

Pipeline (mirrors the email's description of the production rule)::

    raw counters @10s  --ffill missing-->  per-minute counter samples
    per-minute increase (bytes/min)  ->  Input_Rate, Output_Rate  per shard
    SUM over shards                  ->  consumer Input_Rate, Output_Rate
    Completeness_Ratio = SUM(Output_Rate) / SUM(Input_Rate)        (1 min)
    1-hour rolling ratio             ->  Completeness_Ratio_1h    (smoothing)
    ALERT if ratio_1h < threshold for the alert duration          (static rule)

Counter resets and missing scrapes are handled the way a real rate() function
must handle them: a decrease is treated as a reset (that minute's rate is
dropped), and a missing scrape is forward-filled.

A second, *trajectory-aware* rule is also provided (``trajectory_alert``) to
demonstrate the target behaviour the collaboration wants: fire only when the
ratio is low **and** the byte deficit is accumulating **and** not recovering --
so it stays silent during benign burst dips and while the ratio is climbing back.
It is not the object under study; it exists to show the labelled data supports a
better rule than the static threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EmulatorConfig
from .measurement import ObservedResult
from .pipeline import PhysicalResult
from .workload import Consumer


def _minute_counters(counter: np.ndarray, reset_flag: np.ndarray, spm: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce raw per-step counters to per-minute increments (bytes/minute).

    Returns ``(rate_per_min, valid_mask)`` of shape ``(n_shards, n_minutes)``.
    A minute is invalid where its increment cannot be trusted (reset inside the
    minute, negative jump, or no valid scrape).
    """

    n_shards, n = counter.shape
    n_min = n // spm
    counter = counter[:, : n_min * spm]
    reset_flag = reset_flag[:, : n_min * spm]

    # Forward-fill missing (NaN) scrapes along time, per shard.
    filled = counter.copy()
    for sh in range(n_shards):
        row = filled[sh]
        last = np.nan
        for t in range(n):
            if np.isnan(row[t]):
                row[t] = last
            else:
                last = row[t]

    boundary = filled.reshape(n_shards, n_min, spm)[:, :, -1]        # value at end of minute
    inc = np.diff(boundary, axis=1, prepend=boundary[:, :1])
    reset_in_min = reset_flag.reshape(n_shards, n_min, spm).any(axis=2)

    valid = np.isfinite(inc) & (inc >= 0) & ~reset_in_min
    valid[:, 0] = False                                             # no prior minute to diff against
    return inc, valid


def derive_consumer(cfg: EmulatorConfig, consumer: Consumer, obs: ObservedResult,
                    phys: PhysicalResult) -> pd.DataFrame:
    """Per-consumer, per-minute derived signals + the static alert."""

    spm = cfg.raw_steps_per_minute
    n_min = cfg.n_raw_steps // spm
    win = cfg.minutes_per_smoothing_window

    in_inc, in_valid = _minute_counters(obs.input_counted, obs.reset_flag, spm)
    out_inc, out_valid = _minute_counters(obs.output_counted, obs.reset_flag, spm)
    valid = in_valid & out_valid

    in_inc = np.where(valid, in_inc, np.nan)
    out_inc = np.where(valid, out_inc, np.nan)

    # SUM over shards (nan-aware): drop shards missing that minute from both sums.
    input_rate = np.nansum(in_inc, axis=0)
    output_rate = np.nansum(out_inc, axis=0)
    any_valid = valid.any(axis=0)
    input_rate = np.where(any_valid, input_rate, np.nan)
    output_rate = np.where(any_valid, output_rate, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(input_rate > 0, output_rate / input_rate, np.nan)

    num_1h = pd.Series(output_rate).rolling(win, min_periods=win).sum()
    den_1h = pd.Series(input_rate).rolling(win, min_periods=win).sum()
    ratio_1h = (num_1h / den_1h).to_numpy()

    static_alert = _sustained_below(ratio_1h, cfg.alerting.alert_threshold,
                                    cfg.alerting.alert_duration_minutes)

    # Hidden ground-truth state aggregated to the minute (sum over shards at the
    # minute boundary): backlog and cumulative true loss.
    backlog_min = phys.backlog.reshape(phys.backlog.shape[0], n_min, spm)[:, :, -1].sum(axis=0)
    cumloss_min = phys.cum_true_loss.reshape(phys.cum_true_loss.shape[0], n_min, spm)[:, :, -1].sum(axis=0)

    dt = cfg.time.raw_scrape_seconds
    start = np.datetime64(cfg.time.start_timestamp.replace("Z", ""))
    minute_ts = start + (np.arange(n_min) * (spm * dt)).astype("timedelta64[s]")

    return pd.DataFrame({
        "timestamp": minute_ts,
        "consumer_id": consumer.consumer_id,
        "archetype": consumer.archetype,
        "minute_index": np.arange(n_min),
        "input_rate_bytes_per_min": input_rate,
        "output_rate_bytes_per_min": output_rate,
        "completeness_ratio": ratio,
        "completeness_ratio_1h": ratio_1h,
        "static_alert": static_alert,
        "backlog_bytes": backlog_min,
        "cum_lost_bytes": cumloss_min,
    })


def _sustained_below(series: np.ndarray, threshold: float, duration_minutes: int) -> np.ndarray:
    """True where ``series`` has been < threshold for >= ``duration_minutes`` in a row."""

    below = np.where(np.isfinite(series), series < threshold, False)
    out = np.zeros_like(below, dtype=bool)
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        out[i] = run >= duration_minutes
    return out


def trajectory_alert(cfg: EmulatorConfig, frame: pd.DataFrame,
                     accumulate_minutes: int = 45, recover_slope: float = 0.0) -> np.ndarray:
    """Demonstration of the *target* rule the collaboration wants.

    Fires when, over a trailing window, (a) the smoothed ratio is below the
    good-data threshold, (b) the byte deficit ``Input_Rate - Output_Rate`` is
    net positive (bytes going missing, not being recovered), and (c) the ratio is
    not already climbing back. This distinguishes persistent true loss from a
    benign burst dip (which recovers, so the deficit reverses and the slope is
    positive). Purely illustrative; the emulator does not learn this -- Autogram
    would.
    """

    ratio_1h = frame["completeness_ratio_1h"].to_numpy()
    deficit = (frame["input_rate_bytes_per_min"] - frame["output_rate_bytes_per_min"]).to_numpy()
    thr = cfg.alerting.good_data_threshold

    low = np.where(np.isfinite(ratio_1h), ratio_1h < thr, False)
    deficit_sum = pd.Series(deficit).rolling(accumulate_minutes, min_periods=accumulate_minutes).sum().to_numpy()
    slope = pd.Series(ratio_1h).diff(accumulate_minutes).to_numpy()  # >0 means recovering
    accumulating = np.where(np.isfinite(deficit_sum), deficit_sum > 0, False)
    not_recovering = np.where(np.isfinite(slope), slope <= recover_slope, True)

    return low & accumulating & not_recovering
