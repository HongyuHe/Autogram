# gTIB Consumer Byte Completeness — synthetic-data emulator

A deterministic, configurable generator of **realistic synthetic telemetry** for the *gTIB Consumer Byte Completeness Internal* SLO described in the project email thread (`learning_rules_for_production.pdf`). We do **not** have access to the production data, so this emulator produces labelled, ground-truth synthetic data that resembles production as closely as we can honestly justify from what the emails state. Every quantitative choice is either **anchored** to a stated fact or an **explicit, documented assumption exposed as a tunable knob**.

> **Honesty first.** Realism here is *assumed and unvalidated*. The anchored facts (scrape/rate/smoothing intervals, thresholds, catch-up-to-20, multi-tenancy, target loss patterns) are reproduced faithfully; everything else is an educated guess (grounded in published traffic measurements where possible) that should be recalibrated once real data is available. See [Open questions](#open-questions).

---

## What problem is being emulated?

The gTIB pipeline is a multi-tenant telemetry system. For each consumer, bytes are counted at two stages and compared:

```
consumer traffic --> [Collector] --> (gRPC channel + receive buffer ~ a queue) --> [Presenter] --> metrics
                         |                                                              |
                collector_input_counted                                     presenter_output_counted
```

The monitored signal is the **completeness ratio**:

```
Input_Rate  = Rate(collector_input_counted, 1 min)
Output_Rate = Rate(presenter_output_counted, 1 min)
Completeness_Ratio = SUM(Output_Rate) / SUM(Input_Rate)      # SUM over the Presenter's shards/tasks
```

The current production rule is `ALERT IF Completeness_Ratio < 0.99 FOR 1 hour`. It **false-positives on bursty (ML-style) workloads**: rapid traffic changes cause transient ratio dips (from queueing and measurement/alignment artifacts) that are *not* real byte loss. The goal of the wider collaboration is a rule that **reliably detects true persistent byte loss while staying silent during short-lived dips and while the ratio is recovering**. This emulator produces the labelled data needed to build and evaluate such a rule.

A fuller written analysis (with quotes from the thread and a stated-vs-inferred split) lives in the Lavish report at `../.lavish/gtib-byte-completeness.html`.

---

## Quick start

The repo is `uv`-managed (Python ≥ 3.12; `numpy`, `pandas`, `pyyaml` are already in the environment).

```bash
# from the repository root
cd generator

# generate a small default dataset into ./output and validate invariants
uv run python -m gtib_emulator generate --config config.yaml

# scale up, change the seed, write parquet
uv run python -m gtib_emulator generate --config config.yaml --n-consumers 40 --duration-hours 24 --format parquet

# only run the invariant checks (no files written)
uv run python -m gtib_emulator validate --config config.yaml

# print the effective configuration (defaults + file + CLI overrides)
uv run python -m gtib_emulator describe --config config.yaml
```

> If `uv` reports a `uv.lock` parse error, prefix commands with `UV_SKIP_WHEEL_FILENAME_CHECK=1` (a pre-existing lockfile quirk unrelated to this package). On PowerShell: `$env:UV_SKIP_WHEEL_FILENAME_CHECK=1` first.

Running without `uv` also works if `numpy`, `pandas` and `pyyaml` are importable: `python -m gtib_emulator generate` from inside `generator/`.

---

## Outputs

All files are written to `output/` (configurable). They are exactly the two data shapes proposed in the thread (time-series + event catalogue), plus a reproducibility manifest.

| File | Grain | Contents |
| --- | --- | --- |
| `timeseries_raw.csv` | per shard, per 10 s | observed cumulative counters, quality flags, and per-shard `backlog_bytes`/`cum_lost_bytes` hidden state when `include_hidden_state` is enabled |
| `timeseries_derived.csv` | per consumer, per 1 min | workload `archetype`, rates, ratios, static/trajectory alerts, label masks, oracle, and consumer-level hidden ground truth |
| `events.csv` | per injected event | `event_id`, `consumer_id`, `type`, `shard_scope`, `span_start/end`, `duration_minutes`, `severity_pct`, `recovering`, `mechanism`, `detection_sla_minutes`, `expected_alert` |
| `manifest.json` | per run | seed, versions, full effective config, scale, event counts, summary stats, and the static-vs-oracle evaluation |

`backlog_bytes` and `cum_lost_bytes` are **hidden ground truth** (never observable in production); when `output.include_hidden_state` is enabled they appear both per shard in the raw table and summed per consumer in the derived table, which makes the cross-grain boundary identities directly testable. Set `output.include_hidden_state: false` to emit metrics-only tables. Internal gRPC/Channelz queue metrics remain hidden (`measurement.emit_channel_metrics: false`, open question Q-J).

---

## How the data is generated

Six components run per consumer (see the module of the same name):

1. **`workload`** — byte arrivals. Diurnal seasonality × heavy-tailed per-consumer scale (log-normal, per data-center flow-size fits) × archetype. `steady` tenants have mild log-normal jitter; `bursty_ml` tenants add ON/OFF bursts and periodic all-reduce spikes. Intrinsic burstiness is kept mild; the large threshold-crossing spikes are injected as *labelled* `benign_burst` events so the ground truth stays clean.
2. **`pipeline`** — a byte-conserving fluid queue. Service capacity is provisioned against the slow **baseline** demand, so a burst builds real backlog (a dip) that drains gradually with an overshoot up to `catch_up_max_ratio` (the reported "shoot up to 20"). This is where the core invariant `input = output + backlog + true_loss` is enforced exactly.
3. **`measurement`** — physical state → observed counters: the healthy accounting offset, scrape noise, an alignment artifact (deferring a fraction of an output increment during rapid change; byte-conserving, so it dips-then-recovers), missing scrapes, counter resets, and optional shard churn.
4. **`anomalies`** — schedules labelled events and renders the control signals (benign burst multipliers, true-loss fractions, artifact spans). The first benign burst overlaps a true-loss span when both exist so categorical label precedence is identifiable. Writes the event catalogue.
5. **`deriver`** — reproduces the alerting math exactly (1 min rate → SUM over shards → ratio → 1 h smoothing → static alert), plus an illustrative trajectory-aware rule.
6. **`labeling`** — expands events into per-minute masks and the oracle.

```
             benign burst multiplier
                     |
 workload --> [x] --> pipeline (I=O+Q+L) --> measurement --> deriver --> static/traj alerts
   |          ^            |                     |              |
 baseline   anomalies   true loss            artifacts       labels + oracle  --> evaluation
```

### Event families (the labelled ground truth)

| Family | Alerts? | What it looks like | How it is injected |
| --- | --- | --- | --- |
| `true_loss_persistent` | yes | ratio sits 1–5% low, sustained | constant loss fraction on all shards |
| `true_loss_cliff` | yes | ratio steps down and stays | one shard "dies" (large sustained loss) |
| `true_loss_partial` | yes | diluted drop | loss on a subset of shards only |
| `true_loss_creeping` | yes | soft ramp that grazes the threshold | loss fraction ramps 0 → severity |
| `benign_burst` | no | sharp dip then overshoot, recovers | large arrival multiplier, no byte loss |
| `artifact` | no | brief wobble | measurement/alignment span, no byte loss |
| `normal` | no | ratio ~ healthy band | absence of any event |

---

## Invariants expected to hold during normal operation

These follow directly from the generation process and are the executable contract in `invariants.py` (checked by `--validate` and after every `generate`). They are the relations a learned rule may rely on, and the reference against which realism should later be validated.

**Hard invariants (structural — a violation is a generator bug):**

1. **Byte conservation.** For every shard and every step, `cumulative_input == cumulative_output_physical + backlog + cumulative_true_loss` (to floating-point precision). Nothing appears or vanishes except through the explicit true-loss channel.
2. **Non-negative backlog.** The queue never holds negative bytes: `backlog(t) ≥ 0`.
3. **Monotone true loss.** Cumulative true loss only ever increases: `cum_lost_bytes` is non-decreasing.
4. **Monotone counters between resets.** Observed counters are non-decreasing except at flagged `reset_flag` steps (task restarts).
5. **Non-negative rates.** Derived `input_rate` and `output_rate` are ≥ 0.
6. **Exact generated derived grid.** The generator emits exactly `n_consumers` distinct, non-missing typed consumer identities and `n_consumers × n_minutes` derived rows. Within every consumer record, `minute_index` is exactly `0..n_minutes-1` in row order, `timestamp == start_timestamp + minute_index × rate_window_seconds`, and every row carries that record's consumer identity. Across all records, each typed `(consumer_id, minute_index)` identity occurs exactly once; nested tuple identities are valid when every component is non-missing. This is a generator-output contract; the Autogram loader intentionally still accepts valid slices and gaps.
7. **Exact generated raw grid.** When raw output is emitted, `check_all(..., raw)` requires exactly the consumers and per-consumer shards declared by the generated records. Typed shard IDs must be non-missing and unique within each consumer (so `True`, `1`, `"1"`, and `1.0` remain distinct), while valid nested tuple identities are preserved. Rows must appear in consumer/shard/step order with `timestamp == start_timestamp + step × raw_scrape_seconds`, every row must carry its owning consumer and shard identity, and each global `(timestamp, consumer_id, shard_id)` identity must occur exactly once. Missing, extra, reordered, or malformed raw rows produce a hard invariant failure rather than an exception.
8. **Exact counter missingness and finiteness.** A raw counter for a missing scrape or inactive shard is exactly `NaN`; an active, unflagged counter is finite. Infinity is never valid in raw or derived telemetry.

**Soft invariants (statistical expectations of normal operation — checked with tolerances):**

9. **Healthy band.** For steady (non-bursty) consumers during `normal` minutes, the instantaneous 1-minute completeness ratio sits in a tight band around `healthy_ratio_mean` (~0.998) and is **rarely below the alert threshold** (< ~0.1% of the time). Calm, healthy operation looks healthy.
10. **Benign mechanisms lose no bytes.** Over a `benign_burst` or `artifact` span that does not overlap a true-loss event, cumulative true loss does not increase — the dip is a queueing/measurement effect that fully recovers. Deliberate precedence-probe overlaps are excluded from this attribution check because simultaneous true loss legitimately increases the shared counter.
11. **True-loss events accumulate a deficit.** Over any true-loss span, cumulative true loss strictly increases — the deficit is real and (because loss is monotone) never recovers.

**Behavioural expectations that make the dataset useful (demonstrated, not asserted as invariants):**

- **Reproduce-the-bug.** The static `ratio < threshold` rule produces false positives on benign-burst minutes (where `oracle_alert` is false), while a trajectory-aware rule that keys on an *accumulating, non-recovering* deficit is far more precise. The run summary and `manifest.json → evaluation` report both, including `false_positives_on_benign_minutes`.
- **Catch-up overshoot.** After large bursts the 1-minute ratio overshoots (up to `catch_up_max_ratio`), reproducing the reported "ratio shoot up to 20".

---

## Configuration knobs

Full defaults and provenance tags live in `config.py` and are mirrored in `config.yaml`. Highlights, including the six knobs flagged in review:

| Group | Knob | Default | Notes |
| --- | --- | --- | --- |
| time | `raw_scrape_seconds` / `rate_window_seconds` / `smoothing_window_seconds` | 10 / 60 / 3600 | **[STATED]** |
| scale (Q-H) | `n_consumers`, `shards_min/max`, `base_rate_log_mu/sigma` | 6, 1–4, 13.8/1.0 | start small; heavy-tailed rates **[PAPER]** |
| workload (Q-F) | `on_prob`, `off_prob`, `burst_amplitude`, `allreduce_*` | mild ON/OFF + periodic spikes | ML-burst shape; recalibrate to real stats |
| pipeline (Q-E) | `service_margin`, `catch_up_max_ratio`, `drain_gain` | 2.0, 20.0, 0.5 | catch-up to ~20 **[STATED]** |
| pipeline (Q-C) | `healthy_ratio_mean`, `healthy_ratio_std` | 0.998, 0.0015 | healthy ratio distribution (configurable) |
| alerting (Q-D) | `alert_threshold`, `good_data_threshold`, `alert_duration_minutes` | 0.99, 0.98, 10 | both thresholds configurable; 0.99≠0.98 |
| measurement | `alignment_artifact_strength`, `missing_sample_prob`, `counter_reset_prob_per_hour`, `shard_churn_prob_per_hour` | small | benign artifacts + operational realism |
| measurement (Q-J) | `emit_channel_metrics` | `false` | gRPC/Channelz metrics kept hidden for now |
| anomaly | per-family `count_per_consumer`, severity, duration | see `config.yaml` | patterns to catch **[STATED]** |

### Paper-backed defaults (Q-F)

The ML/data-center traffic defaults are grounded in published measurements: heavy-tailed (log-normal) flow/rate sizes and ON/OFF burstiness (Benson et al., *Network Traffic Characteristics of Data Centers in the Wild*, IMC 2010; Greenberg et al., *VL2*, SIGCOMM 2009), with mice/elephant flow mixes reaffirmed by DCTCP (SIGCOMM 2010). ML training traffic adds coordinated, periodic bursts from gradient synchronization (all-reduce), modelled as recurring spikes with straggler jitter. Absolute scales are assumptions; the *shapes* follow the literature. All are knobs to recalibrate against real burst statistics when available.

---

## Maximising realism (and its limits)

- **Anchored where stated, knobbed elsewhere.** Intervals, thresholds, catch-up magnitude, tenancy and the target loss patterns are anchored. Distributions, scale, burst shape, jitter and anomaly base-rates are assumptions, each a labelled knob.
- **Calibration hooks.** `manifest.json` records summary statistics (healthy ratio distribution, catch-up peak, event mix). When real data arrives, fit the knobs so these match — a small loop that turns "assumed" into "validated".
- **Honest caveat.** The synthetic set is a *hypothesis* about production, useful for building and unit-testing a detector. Rules learned on it must be re-validated on real telemetry before deployment, and known-invariant recall is a lower bound, not a guarantee.

---

## Open questions

Answers (or production data) would let us replace assumptions with measurements:

- **Q-C** Real healthy-ratio distribution (mean/variance), not just "~0.98".
- **Q-D** Authoritative thresholds and rule semantics: rule-doc 0.99 vs Tao's 0.98, and how "1 h smoothing" combines with "FOR 1 hour". Both are configurable here; the default is one defensible interpretation.
- **Q-E** Real catch-up timescale and the window over which "~20×" is measured.
- **Q-F** Real ML-burst statistics (amplitude, inter-arrival, duration).
- **Q-G** Base rate and typical severity/duration of real true-loss events (the default injection rate is for a balanced learning set, not a realistic base rate).
- **Q-H** Real scale: consumer count, shards per consumer, size spread.
- **Q-A/Q-B** Exact meaning of `SUM` (over shards, over time) and whether "initial filtering" is ratio-neutral.
- **Q-J** Whether queue/Channelz metrics can be provided (kept hidden for now).

---

## Layout

```
generator/
  README.md              # this file
  config.yaml            # example config mirroring the defaults
  gtib_emulator/
    __init__.py
    __main__.py          # enables `python -m gtib_emulator`
    cli.py               # generate | validate | describe
    config.py            # typed config schema + YAML overlay + validation
    workload.py          # arrivals + burst-free baseline
    pipeline.py          # byte-conserving fluid queue (I = O + Q + L)
    measurement.py       # observed counters + benign artifacts
    anomalies.py         # labelled event injector + event catalogue
    deriver.py           # alerting math + trajectory-aware contrast rule
    labeling.py          # per-minute masks + oracle
    invariants.py        # executable normal-operation invariants
    generate.py          # orchestrator + writers
  tests/
    test_emulator.py     # invariants, determinism, schema
```
