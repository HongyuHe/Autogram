"""Invariants that follow from the generation process.

This module is the executable companion to the "Invariants" section of the
README. It defines the relations that *must* hold given how the data is
generated, and checks them on a produced dataset. Two tiers:

* **hard** -- structural guarantees of the model (byte conservation, monotone
  counters, non-negative backlog/rates). A failure here is a generator bug.
* **soft** -- statistical expectations of *normal operation* (smoothed ratio sits
  in the healthy band; benign bursts do not lose bytes; true-loss spans do lose
  bytes). These should hold but are checked with tolerances because the data is
  deliberately noisy.

``check_all`` returns a list of :class:`InvariantResult`; ``--validate`` in the
CLI prints them and fails the run if any *hard* invariant is violated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import EmulatorConfig


@dataclass
class InvariantResult:
    name: str
    tier: str            # "hard" | "soft"
    passed: bool
    detail: str


def _hard(name: str, ok: bool, detail: str) -> InvariantResult:
    return InvariantResult(name, "hard", bool(ok), detail)


def _soft(name: str, ok: bool, detail: str) -> InvariantResult:
    return InvariantResult(name, "soft", bool(ok), detail)


def check_all(cfg: EmulatorConfig, records: list[dict[str, Any]],
              events: pd.DataFrame) -> list[InvariantResult]:
    """Run every invariant over all per-consumer records."""

    results: list[InvariantResult] = []

    # -- Hard: physical byte conservation  I == O + Q + L  --------------------
    max_resid = 0.0
    max_scale = 1.0
    for rec in records:
        phys = rec["phys"]
        max_resid = max(max_resid, float(phys.conservation_residual().max()))
        max_scale = max(max_scale, float(phys.cum_input.max()))
    rel = max_resid / max_scale
    results.append(_hard(
        "physical_byte_conservation",
        rel < 1e-6,
        f"max |I-(O+Q+L)| = {max_resid:.3g} bytes (relative {rel:.2e}); "
        "input == output + backlog + true_loss must hold exactly."))

    # -- Hard: backlog non-negative ------------------------------------------
    min_backlog = min(float(rec["phys"].backlog.min()) for rec in records)
    results.append(_hard(
        "backlog_non_negative", min_backlog >= -1e-9,
        f"min backlog = {min_backlog:.3g} bytes; the queue can never hold negative bytes."))

    # -- Hard: cumulative true loss non-decreasing ---------------------------
    worst = 0.0
    for rec in records:
        d = np.diff(rec["phys"].cum_true_loss, axis=1)
        worst = min(worst, float(d.min()) if d.size else 0.0)
    results.append(_hard(
        "true_loss_monotone_non_decreasing", worst >= -1e-6,
        f"min step in cumulative true loss = {worst:.3g}; loss can only accumulate."))

    # -- Hard: observed counters monotone within reset-free runs -------------
    bad = _counter_monotone_violations(records)
    results.append(_hard(
        "observed_counters_monotone_between_resets", bad == 0,
        f"{bad} negative counter steps outside a flagged reset (should be 0)."))

    # -- Hard: derived rates non-negative ------------------------------------
    neg_rates = 0
    for rec in records:
        f = rec["frame"]
        for col in ("input_rate_bytes_per_min", "output_rate_bytes_per_min"):
            v = f[col].to_numpy()
            neg_rates += int(np.nansum(v < -1e-6))
    results.append(_hard(
        "derived_rates_non_negative", neg_rates == 0,
        f"{neg_rates} negative derived rate values (should be 0)."))

    # -- Soft: normal-operation ratio sits in the healthy band ---------------
    # Scoped to STEADY consumers and the *instantaneous* 1 min ratio: calm
    # healthy operation must look healthy. (The 1 h smoothed ratio is NOT used
    # here because its trailing window has memory -- a normal minute within an
    # hour after a true-loss event legitimately shows a depressed smoothed ratio;
    # that is real smoothing behaviour, not a generator fault.) Bursty tenants are
    # allowed to dip -- that dip IS the reported false-positive bug.
    normal_ratio = _collect_steady(records, label="normal", col="completeness_ratio")
    if normal_ratio.size:
        med = float(np.nanmedian(normal_ratio))
        lo = cfg.pipeline.healthy_ratio_mean - 0.03
        hi = 1.05
        results.append(_soft(
            "steady_normal_ratio_in_healthy_band", lo <= med <= hi,
            f"median steady-normal 1min ratio = {med:.4f}; expected ~[{lo:.3f}, {hi:.3f}] "
            f"around healthy_ratio_mean={cfg.pipeline.healthy_ratio_mean}."))
        frac_below = float(np.nanmean(normal_ratio < cfg.alerting.alert_threshold))
        results.append(_soft(
            "steady_normal_minutes_rarely_below_threshold", frac_below < 0.10,
            f"{frac_below:.1%} of steady-consumer normal minutes have 1min ratio < "
            f"{cfg.alerting.alert_threshold} (should be small; these would be spurious)."))
    else:
        results.append(_soft("steady_normal_ratio_in_healthy_band", True, "no steady-normal minutes."))

    # -- Soft: benign bursts lose no bytes; true loss accumulates ------------
    benign_ok, benign_detail = _event_loss_behaviour(records, events, benign=True)
    results.append(_soft("benign_events_do_not_lose_bytes", benign_ok, benign_detail))
    loss_ok, loss_detail = _event_loss_behaviour(records, events, benign=False)
    results.append(_soft("true_loss_events_accumulate_deficit", loss_ok, loss_detail))

    return results


def _counter_monotone_violations(records: list[dict[str, Any]]) -> int:
    bad = 0
    for rec in records:
        obs = rec["obs"]
        for counter in (obs.input_counted, obs.output_counted):
            n_shards, n = counter.shape
            for sh in range(n_shards):
                row = counter[sh]
                reset = obs.reset_flag[sh]
                prev = np.nan
                for t in range(n):
                    v = row[t]
                    if np.isnan(v):
                        continue
                    if not np.isnan(prev) and not reset[t] and v < prev - 1.0:
                        bad += 1
                    prev = v
    return bad


def _collect(records: list[dict[str, Any]], mask_col: str, mask_val: str, col: str) -> np.ndarray:
    chunks = []
    for rec in records:
        f = rec["frame"]
        if mask_col in f:
            chunks.append(f.loc[f[mask_col] == mask_val, col].to_numpy())
    return np.concatenate(chunks) if chunks else np.array([])


def _collect_steady(records: list[dict[str, Any]], label: str, col: str) -> np.ndarray:
    """Collect a column over ``label`` minutes, steady-archetype consumers only."""

    chunks = []
    for rec in records:
        if rec["consumer"].archetype != "steady":
            continue
        f = rec["frame"]
        chunks.append(f.loc[f["label"] == label, col].to_numpy())
    return np.concatenate(chunks) if chunks else np.array([])


def _event_loss_behaviour(records: list[dict[str, Any]], events: pd.DataFrame,
                          benign: bool) -> tuple[bool, str]:
    """Check that benign events add ~0 true-loss and true-loss events add >0."""

    if events.empty:
        return True, "no events to check."
    by_consumer = {rec["consumer"].consumer_id: rec for rec in records}
    checked = 0
    violations = 0
    for _, ev in events.iterrows():
        is_benign = ev["type"] in ("benign_burst", "artifact")
        if is_benign != benign:
            continue
        rec = by_consumer.get(ev["consumer_id"])
        if rec is None:
            continue
        frame = rec["frame"]
        seg = frame[(frame["timestamp"] >= ev["span_start"]) & (frame["timestamp"] < ev["span_end"])]
        if len(seg) < 2:
            continue
        delta_loss = float(seg["cum_lost_bytes"].iloc[-1] - seg["cum_lost_bytes"].iloc[0])
        input_bytes = float(np.nansum(seg["input_rate_bytes_per_min"].to_numpy()))
        checked += 1
        if benign:
            if delta_loss > max(1.0, 1e-6 * input_bytes):
                violations += 1
        else:
            if delta_loss <= 0.0:
                violations += 1
    if checked == 0:
        return True, "no matching events with enough coverage."
    ok = violations == 0
    kind = "benign" if benign else "true-loss"
    expect = "add ~0 lost bytes" if benign else "accumulate lost bytes"
    return ok, f"{violations}/{checked} {kind} events violated the expectation that they {expect}."
