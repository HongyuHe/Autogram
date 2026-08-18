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

    # -- Hard: every physical state value is finite --------------------------
    physical_fields = (
        "cum_input",
        "cum_output_physical",
        "cum_true_loss",
        "backlog",
    )
    nonfinite_by_field = {field: 0 for field in physical_fields}
    for rec in records:
        phys = rec["phys"]
        for field in physical_fields:
            values = np.asarray(getattr(phys, field))
            nonfinite_by_field[field] += int(np.count_nonzero(
                ~np.isfinite(values)
            ))
    physical_nonfinite = sum(nonfinite_by_field.values())
    nonfinite_detail = ", ".join(
        f"{field}={count}"
        for field, count in nonfinite_by_field.items()
        if count
    ) or "none"
    results.append(_hard(
        "physical_state_is_finite",
        physical_nonfinite == 0,
        f"{physical_nonfinite} non-finite physical state values "
        f"({nonfinite_detail}).",
    ))

    # -- Hard: every physical byte state is non-negative --------------------
    minimum_by_field = {}
    for field in physical_fields:
        minima = []
        for rec in records:
            values = np.asarray(getattr(rec["phys"], field))
            if values.size and np.all(np.isfinite(values)):
                minima.append(float(np.min(values)))
        minimum_by_field[field] = (
            min(minima)
            if minima
            else float("nan")
        )
    physical_non_negative = (
        physical_nonfinite == 0
        and all(
            minimum >= -1e-9
            for minimum in minimum_by_field.values()
        )
    )
    results.append(_hard(
        "physical_state_is_non_negative",
        physical_non_negative,
        "minimum physical values: " + ", ".join(
            f"{field}={minimum:.3g}"
            for field, minimum in minimum_by_field.items()
        ),
    ))

    # -- Hard: physical byte conservation  I == O + Q + L  --------------------
    max_resid = 0.0
    max_relative = 0.0
    conservation_values_finite = physical_nonfinite == 0
    for rec in records:
        phys = rec["phys"]
        residual = np.asarray(phys.conservation_residual())
        cumulative_input = np.asarray(phys.cum_input)
        if (
            not residual.size
            or residual.shape != cumulative_input.shape
            or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(cumulative_input))
        ):
            conservation_values_finite = False
            continue
        max_resid = max(max_resid, float(np.max(residual)))
        relative = residual / np.maximum(
            np.abs(cumulative_input),
            1.0,
        )
        max_relative = max(
            max_relative,
            float(np.max(relative)),
        )
    if not conservation_values_finite:
        max_relative = float("inf")
    results.append(_hard(
        "physical_byte_conservation",
        conservation_values_finite and max_relative < 1e-6,
        f"max |I-(O+Q+L)| = {max_resid:.3g} bytes "
        f"(pointwise relative to input {max_relative:.2e}); "
        "input == output + backlog + true_loss must hold exactly."))

    # -- Hard: backlog non-negative ------------------------------------------
    backlogs = [
        np.asarray(rec["phys"].backlog)
        for rec in records
    ]
    backlog_values_finite = all(np.all(np.isfinite(v)) for v in backlogs)
    min_backlog = (
        min(float(np.min(v)) for v in backlogs)
        if backlog_values_finite
        else float("nan")
    )
    results.append(_hard(
        "backlog_non_negative",
        backlog_values_finite and min_backlog >= -1e-9,
        f"min backlog = {min_backlog:.3g} bytes; the queue can never hold negative bytes."))

    # -- Hard: cumulative true loss non-decreasing ---------------------------
    worst = 0.0
    true_loss_values_finite = True
    for rec in records:
        cumulative_loss = np.asarray(rec["phys"].cum_true_loss)
        if not np.all(np.isfinite(cumulative_loss)):
            true_loss_values_finite = False
            continue
        d = np.diff(cumulative_loss, axis=1)
        worst = min(worst, float(d.min()) if d.size else 0.0)
    results.append(_hard(
        "true_loss_monotone_non_decreasing",
        true_loss_values_finite and worst >= -1e-6,
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

    nonfinite_reported = 0
    derived_infinite = 0
    for rec in records:
        obs = rec["obs"]
        allowed_missing = obs.missing_flag | ~obs.active_flag
        for counter in (obs.input_counted, obs.output_counted):
            nonfinite_reported += int(np.count_nonzero(
                ~np.isfinite(counter) & ~allowed_missing
            ))
        numeric = rec["frame"].select_dtypes(include=[np.number])
        derived_infinite += int(np.count_nonzero(
            np.isinf(numeric.to_numpy(dtype=float))
        ))
    results.append(_hard(
        "reported_telemetry_is_finite",
        nonfinite_reported == 0 and derived_infinite == 0,
        f"{nonfinite_reported} unflagged nonfinite counter values and "
        f"{derived_infinite} infinite derived values.",
    ))

    # -- Hard: every emitted derived identity is reproducible ----------------
    derived_bad = 0
    static_bad = 0
    trajectory_bad = 0
    label_bad = 0

    def strict_boolean(series: pd.Series) -> tuple[np.ndarray, int]:
        raw = series.to_numpy(dtype=object)
        valid = np.fromiter(
            (
                isinstance(value, (bool, np.bool_))
                for value in raw
            ),
            dtype=bool,
            count=raw.size,
        )
        output = np.zeros(raw.size, dtype=bool)
        output[valid] = np.asarray(raw[valid], dtype=bool)
        return output, int(np.count_nonzero(~valid))

    for rec in records:
        frame = rec["frame"]
        obs = rec["obs"]
        phys = rec["phys"]
        spm = cfg.raw_steps_per_minute
        win = cfg.minutes_per_smoothing_window
        minute_values = []
        for counter in (obs.input_counted, obs.output_counted):
            filled = pd.DataFrame(counter.T).ffill().to_numpy().T
            boundary = filled.reshape(
                filled.shape[0], -1, spm,
            )[:, :, -1]
            delta = np.diff(
                boundary,
                axis=1,
                prepend=boundary[:, :1],
            )
            reset = obs.reset_flag.reshape(
                obs.reset_flag.shape[0], -1, spm,
            ).any(axis=2)
            valid = np.isfinite(delta) & (delta >= 0.0) & ~reset
            valid[:, 0] = False
            minute_values.append((delta, valid))
        common_valid = minute_values[0][1] & minute_values[1][1]
        rates = []
        for delta, _valid in minute_values:
            usable = np.where(common_valid, delta, np.nan)
            summed = np.nansum(usable, axis=0)
            rates.append(np.where(
                common_valid.any(axis=0),
                summed,
                np.nan,
            ))
        input_rate, output_rate = rates
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                input_rate > 0.0,
                output_rate / input_rate,
                np.nan,
            )
        ratio_1h = (
            pd.Series(output_rate)
            .rolling(win, min_periods=win)
            .sum()
            / pd.Series(input_rate)
            .rolling(win, min_periods=win)
            .sum()
        ).to_numpy()
        backlog = phys.backlog.reshape(
            phys.backlog.shape[0], -1, spm,
        )[:, :, -1].sum(axis=0)
        loss = phys.cum_true_loss.reshape(
            phys.cum_true_loss.shape[0], -1, spm,
        )[:, :, -1].sum(axis=0)
        expected_columns = {
            "input_rate_bytes_per_min": input_rate,
            "output_rate_bytes_per_min": output_rate,
            "completeness_ratio": ratio,
            "completeness_ratio_1h": ratio_1h,
            "backlog_bytes": backlog,
            "cum_lost_bytes": loss,
        }
        for column, expected_values in expected_columns.items():
            if not np.allclose(
                frame[column].to_numpy(dtype=float),
                expected_values,
                rtol=1e-12,
                atol=1e-9,
                equal_nan=True,
            ):
                derived_bad += 1
        below = np.where(
            np.isfinite(ratio_1h),
            ratio_1h < cfg.alerting.alert_threshold,
            False,
        )
        expected_static = np.zeros(len(frame), dtype=bool)
        run = 0
        for index, active in enumerate(below):
            run = run + 1 if active else 0
            expected_static[index] = (
                run >= cfg.alerting.alert_duration_minutes
            )
        static_actual, invalid = strict_boolean(
            frame["static_alert"]
        )
        static_bad += invalid + int(np.count_nonzero(
            static_actual != expected_static
        ))
        deficit = input_rate - output_rate
        low = np.where(
            np.isfinite(ratio_1h),
            ratio_1h < cfg.alerting.good_data_threshold,
            False,
        )
        deficit_sum = pd.Series(deficit).rolling(
            45,
            min_periods=45,
        ).sum().to_numpy()
        slope = pd.Series(ratio_1h).diff(45).to_numpy()
        expected_trajectory = (
            low
            & np.where(
                np.isfinite(deficit_sum),
                deficit_sum > 0.0,
                False,
            )
            & np.where(
                np.isfinite(slope),
                slope <= 0.0,
                True,
            )
        )
        trajectory_actual, invalid = strict_boolean(
            frame["traj_alert"]
        )
        trajectory_bad += invalid + int(np.count_nonzero(
            trajectory_actual != expected_trajectory
        ))
        timestamps = pd.to_datetime(frame["timestamp"])
        true_loss = np.zeros(len(frame), dtype=bool)
        benign = np.zeros(len(frame), dtype=bool)
        artifact = np.zeros(len(frame), dtype=bool)
        consumer_events = events[
            events["consumer_id"] == rec["consumer"].consumer_id
        ] if not events.empty else events
        for _, event in consumer_events.iterrows():
            interval_end = timestamps + pd.to_timedelta(
                cfg.time.rate_window_seconds,
                unit="s",
            )
            active = (
                (timestamps < pd.Timestamp(event["span_end"]))
                & (
                    interval_end
                    > pd.Timestamp(event["span_start"])
                )
            ).to_numpy(dtype=bool)
            event_type = str(event["type"])
            if event_type.startswith("true_loss"):
                true_loss |= active
            elif event_type == "benign_burst":
                benign |= active
            elif event_type == "artifact":
                artifact |= active
        for column, expected_mask in (
            ("is_true_loss", true_loss),
            ("is_benign_burst", benign),
            ("is_artifact", artifact),
        ):
            actual, invalid = strict_boolean(frame[column])
            label_bad += invalid + int(np.count_nonzero(
                actual != expected_mask
            ))
        expected_label = np.full(len(frame), "normal", dtype=object)
        expected_label[artifact] = "artifact"
        expected_label[benign] = "benign_burst"
        expected_label[true_loss] = "true_loss"
        label_bad += int(np.count_nonzero(
            frame["label"].to_numpy(dtype=object) != expected_label
        ))
        oracle, invalid = strict_boolean(frame["oracle_alert"])
        label_bad += invalid + int(np.count_nonzero(
            oracle != true_loss
        ))
        label_bad += int(np.count_nonzero(
            frame["consumer_id"].to_numpy(dtype=object)
            != rec["consumer"].consumer_id
        ))
    results.extend((
        _hard(
            "derived_signal_identities",
            derived_bad == 0,
            f"{derived_bad} emitted derived columns disagree with re-derivation.",
        ),
        _hard(
            "static_alert_definition",
            static_bad == 0,
            f"{static_bad} static-alert rows disagree with the sustained rule.",
        ),
        _hard(
            "trajectory_alert_definition",
            trajectory_bad == 0,
            f"{trajectory_bad} trajectory-alert rows disagree with the target rule.",
        ),
        _hard(
            "label_priority_definition",
            label_bad == 0,
            f"{label_bad} label/oracle rows disagree with priority identities.",
        ),
    ))

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
    benign_ok, benign_detail = _event_loss_behaviour(
        cfg,
        records,
        events,
        benign=True,
    )
    results.append(_soft("benign_events_do_not_lose_bytes", benign_ok, benign_detail))
    loss_ok, loss_detail = _event_loss_behaviour(
        cfg,
        records,
        events,
        benign=False,
    )
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


def _event_loss_behaviour(cfg: EmulatorConfig, records: list[dict[str, Any]], events: pd.DataFrame,
                          benign: bool) -> tuple[bool, str]:
    """Check that benign events add ~0 true-loss and true-loss events add >0."""

    if events.empty:
        return True, "no events to check."
    by_consumer = {rec["consumer"].consumer_id: rec for rec in records}
    checked = 0
    violations = 0
    overlap_skipped = 0
    checked_by_type: dict[str, int] = {}
    matching_types = {
        str(value)
        for value in events["type"].unique()
        if (value in ("benign_burst", "artifact")) == benign
    }
    for _, ev in events.iterrows():
        is_benign = ev["type"] in ("benign_burst", "artifact")
        if is_benign != benign:
            continue
        rec = by_consumer.get(ev["consumer_id"])
        if rec is None:
            continue
        if benign:
            overlapping_loss = events[
                (events["consumer_id"] == ev["consumer_id"])
                & ~events["type"].isin(("benign_burst", "artifact"))
                & (events["span_start"] < ev["span_end"])
                & (events["span_end"] > ev["span_start"])
            ]
            if not overlapping_loss.empty:
                overlap_skipped += 1
                continue
        frame = rec["frame"]
        seg = frame[(frame["timestamp"] >= ev["span_start"]) & (frame["timestamp"] < ev["span_end"])]
        if len(seg) >= 2:
            delta_loss = float(
                seg["cum_lost_bytes"].iloc[-1]
                - seg["cum_lost_bytes"].iloc[0]
            )
            input_bytes = float(np.nansum(
                seg["input_rate_bytes_per_min"].to_numpy()
            ))
        else:
            start = np.datetime64(
                cfg.time.start_timestamp.replace("Z", "")
            )
            dt = int(cfg.time.raw_scrape_seconds)
            first = int(
                (np.datetime64(ev["span_start"]) - start)
                / np.timedelta64(dt, "s")
            )
            stop = int(
                (np.datetime64(ev["span_end"]) - start)
                / np.timedelta64(dt, "s")
            )
            first = max(0, first)
            stop = min(cfg.n_raw_steps, stop)
            if stop <= first:
                continue
            prior = max(0, first - 1)
            current = max(prior, stop - 1)
            phys = rec["phys"]
            delta_loss = float(
                np.sum(phys.cum_true_loss[:, current])
                - np.sum(phys.cum_true_loss[:, prior])
            )
            input_bytes = float(
                np.sum(phys.cum_input[:, current])
                - np.sum(phys.cum_input[:, prior])
            )
        checked += 1
        checked_by_type[str(ev["type"])] = (
            checked_by_type.get(str(ev["type"]), 0) + 1
        )
        if benign:
            if delta_loss > max(1.0, 1e-6 * input_bytes):
                violations += 1
        else:
            if delta_loss <= 0.0:
                violations += 1
    if checked == 0:
        return True, "no matching events with enough coverage."
    uncovered = sorted(
        event_type
        for event_type in matching_types
        if checked_by_type.get(event_type, 0) == 0
    )
    ok = violations == 0 and not uncovered
    kind = "benign" if benign else "true-loss"
    expect = "add ~0 lost bytes" if benign else "accumulate lost bytes"
    suffix = (
        f"; skipped {overlap_skipped} precedence-probe overlaps"
        if overlap_skipped else ""
    )
    coverage = ", ".join(
        f"{event_type}:{checked_by_type.get(event_type, 0)}"
        for event_type in sorted(matching_types)
    )
    if uncovered:
        suffix += f"; uncovered subtypes={uncovered}"
    return (
        ok,
        f"{violations}/{checked} {kind} events violated the expectation "
        f"that they {expect}; coverage={coverage}{suffix}.",
    )
