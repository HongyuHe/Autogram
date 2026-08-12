# GTIB implementation — status and handoff

**Last updated:** end of review round 37.
**Branch:** `dev`, pushed to `origin/dev`. The base of this work is commit `3221b4b`; everything since is committed, so `git --no-pager diff 3221b4b..HEAD` shows the whole change and `git --no-pager log 3221b4b..HEAD` explains each step.

This document exists so that another engineer can pick this work up cold. It covers what was built, how it was verified, how the implementation–review loop works, and exactly what is left to do.

---

## 1. What this work is

The task was to implement the **GTIB plan**, whose authoritative specification is the HTML file [`./gtib_autogram_plan.html`](./gtib_autogram_plan.html) in this repository. **Read that file first.** Everything below assumes it.

Structure of the plan, with the line numbers you will want:

| Plan section | Lines | Content |
| --- | --- | --- |
| Phases 0–3 | up to ~530 | Ingestion, ratios and proportional laws, temporal operators, row conditions |
| Phase 4 | ~533–558 | Cross-grain joins, run-length predicates, multi-term conjunctions |
| Acceptance criteria | ~668–690 | Per-phase unit tests and pass conditions |
| Line 674 | | "run-length and conjunction match their planted spans exactly" |
| Line 675 | | CrossCheck non-regression: Abilene + GÉANT must be byte-for-byte unchanged |
| Line 676 | | "end-to-end on data/gtib-emulation with gtib_known.yaml; held-out recall climbs monotonically per phase; null stays 0" |

Autogram itself is a deterministic invariant-discovery engine. The pipeline is: name-driven grammar induction → exhaustive bounded candidate enumeration → data-only evaluation → Pareto archive → known-invariant matching → calibration with null controls. The hard design rule that constrains almost every decision here is that **calibration must accept zero false discoveries on null (shuffled/independent) controls**. Any change that raises recall by also accepting noise is a regression, not progress.

---

## 2. How the implementation–review loop works

This work is being driven by an iterative adversarial review process. It matters that you keep running it, because every round so far has found at least one genuine defect, and several rounds found defects introduced by the *previous* round's fix.

**The loop:**

1. Implement (or remediate the previous round's findings).
2. Verify: run the full test suite, regenerate both calibration reports, confirm the acceptance criteria.
3. Spawn a **fresh reviewer with a clean context** — model GPT-5.6 Sol, long-context tier, maximum reasoning effort — and hand it the plan plus the current state.
4. The reviewer must end its response with a final line of exactly `DONE` or `NOT DONE`. `DONE` is only permitted when it found no Critical and no Important defects.
5. If `NOT DONE`, remediate and go to step 1.
6. **Stop condition: two consecutive `DONE` verdicts.**

**Current streak: 0 of 2.** Rounds 28 through 37 all returned `NOT DONE`. Rounds 29-37 found seven, seven, five, two, three, six, four, five and five Important defects respectively -- every one of them in the *previous* round's remediation. Expect the next round to find something too.

**Rules that make the loop work, learned the hard way:**

- **Always spawn a fresh reviewer.** Reusing a reviewer lets it anchor on its own earlier conclusions.
- **Tell the reviewer how to see the diff.** `git --no-pager diff 3221b4b..HEAD`, plus `git --no-pager log 3221b4b..HEAD` for the reasoning behind each commit.
- **Tell the reviewer what is already running.** Give it the list of long jobs in flight, or it will start a three-hour suite of its own.
- **Tell the reviewer to clear `__pycache__`** before targeted runs. A stale `.pyc` masked a real fix during round 33 and cost an hour of misdiagnosis.
- **Give the reviewer the environment caveats** (venv paths, which jobs are already running, and which are too slow to re-run), or it will report environmental failures as defects, or start a three-hour job you are already running.
- **Give the reviewer the "settled decisions" list** so it does not re-litigate resolved questions, but explicitly invite it to overturn them with new concrete evidence.
- **Push back when a finding is wrong.** Not every finding has been accepted; one was rejected with measurements (see §6).
- **Red-green verify every regression test.** Write the test, confirm it passes, then revert the fix and confirm the test *fails*. A test that passes both ways guards nothing.

### 2.1 A trap in the red-green harness — read this before you write one

A scripted red-green harness that stores one "original" copy per file and then applies **two** edits to that same file will re-read the already-modified file on the second edit, clobber the pristine copy, and silently leave the first edit in place after "restoring".

This happened. It left `covered = np.ones(...)` in `autogram/loader/gtib.py` instead of restoring `covered = adjacent.copy()`, and the corruption was only caught hours later by the full suite (`417 passed, 2 failed`). Two separate targeted runs had passed in the meantime because the second-pass logic masked it.

**Mitigations, all of which are now standard practice here:**
- Snapshot each file **once**, before any edit.
- After restoring, assert the file is **byte-identical** to the snapshot.
- Prefer one edit per file per red-green run.
- Never trust a targeted run as proof the tree is clean; the full suite is the only real check.

---

## 3. Where the code lives

Everything is committed, so `git --no-pager diff --stat 3221b4b..HEAD` is the authoritative map. The
files you will touch most:

| File | Role |
| --- | --- |
| `autogram/dsl/evaluate.py` | Grounding. `Grounded`, `ground()`, overflow tracking (`eval_term_overflow`, `_blowup`, `_union_overflow`, `_shift_overflow`, `_window_overflow`), `robust_median`, cross-grain aggregation, `typed_group_key`. |
| `autogram/discovery/evaluate.py` | The evaluator. Acceptance predicate, definitions, conjunctions, `SUSTAINED`, threshold fitting, condition-support floors, the finite-arithmetic guards, `_typed_label` / `_display_keys`. |
| `autogram/discovery/known.py` | Known-invariant signatures, canonicalisation (`_drop_negligible`, `_stable_row_sum`, `_candidate_is_exact`), `recover_known`. |
| `autogram/calibrate.py` | Calibration driver, `_ColumnScaleView`, `_split_known`, capability tiers, null-control gating. |
| `autogram/discovery/propose.py` | Enumeration, conditioning, conjunction tiers, the conditioned pre-count. |
| `autogram/discovery/validate.py` | Proxy suites and null controls, including the null magnitude envelope. |
| `autogram/loader/gtib.py` | GTIB ingestion, profiling, shard materialisation, `_is_regime_column`. |
| `docs/autogram_guarantees.md` | The published guarantees. Must stay true to the code. |
| `tests/test_overflow.py` | The finite-arithmetic regression suite added by this work. |

---

## 4. Verification status

**Read this before trusting any number below.** The engine fingerprint recorded in a report is
computed from the source files *when the report is written*, not from the modules the run loaded, so
a report can match the fingerprint yet have been produced by different code. Freeze the tree before
launching a verification run, and do not edit `autogram/**` while one is in flight. (Editing `docs/`
or `tests/` is safe -- the fingerprint only covers `autogram/`.)

### 4.1 Current state — fully verified on commit `18623a4`

The tree was frozen at `18623a4` and all three verifications were run to completion against it:

- **Full test suite: 497 passed, 0 failed** (4:51:08).
- `artifacts/gtib_report.json`: **21/21** recovered, `recall_all` 1.0, `recall_validation` 1.0,
  false discovery `{equalities: 0, temporal: 0, definitions: 0}`, no missed knowns.
- `artifacts/gtib_raw_report.json`: **2/2** recovered, same figures.
- Both reports' `provenance.engine_source_sha256` matches the live
  `autogram.calibrate._engine_source_fingerprint()`, so neither is stale.

This is the first commit in the round-29..37 series on which the derived report and the full suite
both finished while it was still `HEAD`; earlier rounds' remediation always landed first.

### 4.2 Runtime, for planning

The suite is serial and two tests dominate it:

| Test | Time |
| --- | --- |
| `test_gtib_end_to_end.py::test_gtib_phase_ladder_climbs_held_out_recall_with_zero_null_acceptances` | 2:20:36 |
| `test_gtib_end_to_end.py::test_checked_in_gtib_known_catalog_reaches_full_recall` | 38:40 |
| `test_validate.py::test_run_all_reports_proxy_phase` | 23:46 |
| `test_cli.py::test_cli_default_search_bound_matches_proxy_runtime_bound` | 22:18 |

The derived calibration is ~100 minutes and the raw one ~10; both run happily alongside the suite.
Budget about five hours of wall clock for a complete verification, and start it only once the tree
is frozen.

### 4.3 Hygiene

`git --no-pager diff --check 3221b4b` exits 0.

---

## 5. How to run things

### 5.1 Environment

- Use the venv: `.venv/bin/python` and `.venv/bin/autogram`. **There is no `python -m autogram` entry point.** The system Python lacks pandas. Provision with `UV_SKIP_WHEEL_FILENAME_CHECK=1 uv sync` (the lock file carries one wheel whose filename version disagrees with its metadata).
- **Use the whole machine.** There is no core cap and no thread cap; do not set `OMP_NUM_THREADS` or its siblings, and do not pin affinity. Run the full suite and both calibration reports concurrently -- they are independent, and the wall clock is dominated by a single long test either way.
- Set `AUTOGRAM_SUBAGENT_CACHE=1` for anything that triggers subagent schema induction, so an identical prompt is answered once per process. The subagent timeout needs no override.
- Give each concurrent job a **distinct** `AUTOGRAM_SUBAGENT_LOG`, or their JSONL appends interleave.
- stdout is buffered until the process ends, so an output file of size 0 is normal for a long run. Redirect to a file and poll it rather than waiting on the terminal.

### 5.2 Full test suite

~2.5–3 hours. `tests/test_gtib_end_to_end.py::test_gtib_phase_ladder_climbs_held_out_recall_with_zero_null_acceptances` alone is ~1.5 hours by design, and the suite is serial, so that test sets the floor.

```bash
export AUTOGRAM_SUBAGENT_HARNESS=copilot AUTOGRAM_SUBAGENT_CACHE=1
export AUTOGRAM_SUBAGENT_LOG=artifacts/subagent_tests.jsonl
nohup .venv/bin/python -m pytest -q -p no:cacheprovider tests --durations=25 \
      -W ignore::RuntimeWarning > artifacts/full_suite.out 2>&1 &
```

### 5.3 Calibration reports

The derived report is ~100 minutes; the raw one is ~10. Launch both alongside the suite -- they are independent.

```bash
export AUTOGRAM_SUBAGENT_HARNESS=copilot AUTOGRAM_SUBAGENT_CACHE=1

AUTOGRAM_SUBAGENT_LOG=artifacts/subagent_gtib.jsonl \
  nohup .venv/bin/autogram calibrate --config configs/gtib.yaml \
        --out artifacts/gtib_report.json > artifacts/gtib_calib.out 2>&1 &

AUTOGRAM_SUBAGENT_LOG=artifacts/subagent_gtib_raw.jsonl \
  nohup .venv/bin/autogram calibrate --config configs/gtib_raw.yaml \
        --out artifacts/gtib_raw_report.json > artifacts/gtib_raw_calib.out 2>&1 &
```

`artifacts/gtib_report.json` is ~10 MB. Query it with Python; do not print it.

### 5.4 Acceptance check

Any source change alters the engine fingerprint, so the reports must be regenerated and re-checked:

```python
import json, autogram.calibrate as C
live = C._engine_source_fingerprint()
for path, expected in (("artifacts/gtib_report.json", 21), ("artifacts/gtib_raw_report.json", 2)):
    r = json.load(open(path, encoding="utf-8"))
    recovered = sum(1 for item in r["invariants"] if item["recovered"])
    assert recovered == len(r["invariants"]) == expected
    assert r["recall_all"] == 1.0 and r["recall_validation"] == 1.0
    assert all(v == 0 for v in r["false_discovery"].values())
    assert r["provenance"]["engine_source_sha256"] == live      # else the report is STALE
```

The fingerprint is computed from the source files **when the report is written**, not from the
modules the run loaded. Editing source while a calibration is in flight therefore produces a report
that *matches* the fingerprint yet was produced by different code. Freeze the tree before launching
a verification run.

---

## 6. Settled decisions — do not undo these without new evidence

Each was investigated in depth and is load-bearing. If you think one is wrong, bring a failing case, a code path, or a measurement.

1. **Lag retention and shadow suppression.** Retaining `LAG_k(x) OP 0` sign bounds unconditionally drives `null_temporal_accepted` from 0 to ~20, which violates the plan's zero-false-discovery rule. Lag bounds are retained by `_is_atomic_ref`, then `portfolio(non_redundant=True)` calls `_suppress_lag_shadows`, which drops a lag **only** when a same-role/same-direction atomic bound with tolerance-free `raw_exact_sign` is also retained. Lag and atomic occupy different archive cells, so cross-cell suppression at portfolio level is structurally necessary — the `add()` loop never compares them.
2. **`raw_exact_sign` is not `hold_rate == 1.0`.** A hold rate of 1.0 can mean the acceptance tolerance absorbed a real violation against a large scale. Exactness is computed tolerance-free on the **full population before subsampling**.
3. **Learned-threshold conjunctions are deliberately not exhaustively enumerated.** The grammar admits them generically (≤2 learned) and the evaluator fits ≤2 jointly and exactly, but *enumerating* them is O(|candidates|²) joint fits; even ≤1 learned threshold at arity 2 measured **18+ minutes** on a single GTIB discovery. The bounded compound tier that replaces it is load-bearing for known invariant B5.
4. **Missing-data conventions were chosen by measurement, not taste.**
   - *Conjunction:* drop-the-row grades 1501/2160 at agreement 1.000 but hides 45 rows where the target fires; the chosen three-valued rule grades 1730/2160 at agreement 1.000; "missing is vacuously true" grades 1861/2160 but agreement falls to 0.954.
   - *`SUSTAINED`:* a missing period falsifies the operator's own claim that a run of observations occurred, which grades all 2160 rows at agreement 1.000.
   - Only **parameter-free** conjuncts may decide gradability, so the graded population cannot move with a fitted threshold.
5. **Increments and levels differ deliberately in cross-grain sums.** Increments skip a shard that reset, because the emitted per-minute value excludes it too (`test_related_delta_sums_valid_shards_across_reset`). Levels are all-or-nothing, because every shard's level exists whether or not it was reported. Documented at both sites.
6. **The phase ladder pins capabilities on the grammar, not the spec.** `build_dataframe_grammar` derives capabilities from the *frame's adapter*, and `prepare_gtib_files` already declares all of them, so widening a bare spec is a no-op. The ladder asserts candidate-set nesting and per-phase absence/presence, and asserts null controls only at the widest rung on the strength of that nesting.
7. **`tune_joint`'s early return** on an eligible explicit initial cell is intentional and tested.

**One finding was rejected**, not accepted: a reviewer claimed the `tune_joint` early return was a defect. It was investigated and pushed back on, because null controls demonstrate 0 false discoveries at the tuned operating point (tolerance 0.05, hold 0.62) with held-out recall 1.0. Reviewers are not automatically right.

---

## 7. Remaining work

The four round-28 findings that this document used to list are **all fixed**, together with the
thirty-nine further Important defects rounds 29-36 found in that remediation. Read
`git --no-pager log 3221b4b..HEAD` for the full account; each commit message states the defect, why
it mattered, and what the fix is.

### 7.1 Open findings from review round 37

Round 37 raised five Important findings. Three are fixed (the two regressions round 36's typed
identity introduced -- missing labels fragmenting into singleton groups, and partition ordering by
rendered form changing a floating-point total -- plus saturated windows excluding a reading exactly
on the representable ceiling). **Two remain open**, both reproduced by the reviewer, neither
re-derived independently, so reproduce them first:

- **TODO-A — the materialised GTIB join still collapses typed identities.**
  `autogram/loader/gtib.py` around lines 479-516 and 563-589 uses `groupby`, `astype(str)` and
  string-keyed lookups, so `True`/`1` merge and `1`/`"1"` collide. The reviewer reports two
  typed-distinct shards producing materialised `[20, 20]` against streaming `[30, 30]`. That is a
  streaming-vs-materialised disagreement, which is exactly the class of defect round 27 was about,
  and the two paths are required to agree. Fix by threading the shared typed key through
  materialisation and making the generated column names injective.

- **TODO-B — categorical identity is still type-insensitive outside the grouping paths.**
  `autogram/discovery/loop.py` (~359-361), `autogram/dsl/evaluate.py` (~691-708) and
  `autogram/discovery/evaluate.py` (~1066-1102) use `pd.unique`, plain equality and membership when
  profiling condition domains, matching conditions, and scoring categorical definitions. The
  reviewer reports an enumerable categorical definition accepted at hold rate 1.0 whose
  type-correct agreement is 2/3. Fix by applying the same typed equality across profiling,
  validation, enumeration, conditions and definition scoring.

Three Minor findings are also open: `calibrate()` runs an external induction before validating the
known-count and validation fraction (validate deterministic inputs first); a canonicalisation
docstring still says approximate removal leaves a sum "unchanged" when it bounds the change by the
tolerance; and the tier commentary promises *strict* search-space growth where merging can be
identity-preserving (state non-shrinking growth, or report zero-delta tiers).

What is left:

1. **Fix TODO-A and TODO-B above**, then re-verify (§4 is green as of `18623a4`, so any change
   invalidates it and both reports plus the suite must be re-run).
2. **Continue the review loop to two consecutive `DONE` verdicts** (§2). The next round is round 38.
3. **Do not assume convergence.** The findings have become more exotic as the obvious ones were
   fixed -- the last few rounds turned up pre-epoch timestamp wrap-around, `True` colliding with `1`
   inside composite group keys, and pandas `groupby` merging those two before the engine ever saw
   them -- but they have not stopped, and each was a genuine soundness or honesty defect.

### What the review loop has hardened, thematically

Useful orientation for whoever picks this up, because it is where the next defect is most likely to
be found too:

- **Finite arithmetic became an acceptance conjunct (S2).** Overflow used to be treated as missing
  data; it is now tracked through term evaluation, cross-grain aggregation, post-fit proportional
  arithmetic, the band centre subtraction, and definitions, and a tolerated overflow is still
  excluded from the scored population and the reported support.
- **Known-invariant canonicalisation** became pointwise, collective, domain-preserving,
  deterministically summed, and exactness-aware -- each property added because its absence credited
  a law the data does not satisfy.
- **Group identity** is recursively type-qualified everywhere a label is a key, is compared, or is
  used to bucket. `True == 1` and they hash alike; every place that forgot it merged two groups, and
  a merged group cannot fail the per-group gate.
- **Reported support** counts only the rows a rule was actually graded on, everywhere.

---

## 8. History of the review rounds

Rounds 1–10 are covered in earlier session checkpoints (the full GTIB feature set, plus correctness, search, holdout, null-control, schema and temporal remediation). Rounds 11–28 all returned `NOT DONE`. The most instructive ones:

| Round | Finding worth remembering |
| --- | --- |
| 17 | Retaining lag bounds as prescribed drove `null_temporal` from 0 to 20 — a hard plan violation. Resolved with exactness-gated suppression instead. |
| 18 | Measured that exhaustive learned-threshold conjunction enumeration adds 18+ minutes to one discovery; kept the generic grammar, dropped the enumeration. |
| 21 | A self-caught regression: generalising the conditioning predicate narrowed `Diff`-vs-zero to one-sided operators and broke two knowns. |
| 23 | `SUSTAINED` was not total — warm-up and gap rows were excluded rather than predicted `False`, letting a wrong target score a perfect hold rate. |
| 24 | Conjunctions were not total either; the missing-data convention was then chosen by **measuring three candidate policies against the emitted data**. |
| 25 | The phase ladder's null-coverage argument rested on nesting that did not hold. |
| 26 | The round-25 ladder fix set `grammar.proportional_enabled`, **a field `Grammar` does not have** — an inert attribute that looked like a fix and did nothing. Phase 0 still carried 30 proportional and 1,969 cross-grain candidates. |
| 27 | Materialised and streaming cross-grain paths still disagreed on non-adjacent minutes and all-invalid minutes. |
| 28 | Four findings: overflow silently treated as missing data; median-decided negligibility; the split's view decoding differently from the runtime frame; a condition-inference guard too weak to reject an identifier. |
| 29 | Seven. The most consequential: the post-fit arithmetic of a proportional law had no finite-arithmetic guard at all, and 600 individually-negligible summed members contributed 6% of a total between them. |
| 30 | Seven. A blown-up counter difference was absorbed by a cross-grain validity mask and contributed zero to a total that still looked complete; the post-fit guard read the coreset rather than the population. |
| 31 | Five. Per-group coefficients keyed by `str(label)` collided; a SUSTAINED window's taint was counted as one row rather than the whole window. |
| 32 | Two. A *tolerated* overflow was still scored as evidence, and reporting re-introduced the group-key collision that fitting had just removed. |
| 33 | Three. Exact relations were canonicalised with a tolerance they do not have, crediting a law false on every row. |
| 34 | Six. Reductions can return **NaN** from finite members (pairwise summation cancels ±inf), which a guard looking only for infinities lets through as missing data; `True` and `1` merged in every dict keyed by group label. |
| 35 | Four. The exactness override was inert; int64 timestamp arithmetic wrapped near `Timestamp.max`; typed identity was shallow and missed subsampling and temporal grouping. |
| 36 | Five. Grouped holdout splits and the related-grain join still merged `True` with `1` -- pandas `groupby` does it before the engine sees the key -- and saturation itself overflowed for pre-epoch timestamps. |

The pattern worth internalising: **most rounds found a defect in the previous round's fix.** Verify fixes empirically against the data rather than reasoning about them, and red-green every regression test — thirteen tests in this series were caught passing whether or not their own fix was present, and were strengthened or removed.
