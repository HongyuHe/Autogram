# GTIB implementation — status and handoff

**Last updated:** end of review round 28.
**Branch:** `main`, working tree only. Nothing is committed; `HEAD` is still the base commit `d575db0`.

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

**Current streak: 0 of 2.** Round 28 returned `NOT DONE`.

**Rules that make the loop work, learned the hard way:**

- **Always spawn a fresh reviewer.** Reusing a reviewer lets it anchor on its own earlier conclusions.
- **Tell the reviewer how to see the diff.** All work is uncommitted and `HEAD == base`, so `git diff BASE..HEAD` shows nothing. The reviewer must use `git --no-pager diff d575db0`, and must be given the explicit list of untracked files (below), because those appear in no diff at all.
- **Give the reviewer the environment caveats** (venv paths, the 4-core cap, the 900 s subagent timeout), or it will report environmental failures as defects.
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

### 3.1 Untracked files (invisible to `git diff` — read them directly)

```
.gitattributes
autogram/loader/gtib.py
configs/gtib.yaml
configs/gtib_known.yaml
configs/gtib_raw.yaml
configs/gtib_raw_known.yaml
docs/autogram_guarantees.md
scripts/                       (whole directory)
tests/test_boolean_logic.py
tests/test_conditions.py
tests/test_crosscheck_golden.py
tests/test_distribution_band.py
tests/test_gtib_end_to_end.py
tests/test_gtib_ingest.py
tests/test_gtib_proxies.py
tests/test_ratio_proportional.py
tests/test_relational.py
tests/test_temporal.py
```

### 3.2 Modified tracked files

60 files, roughly +56k / −44k lines. See `git --no-pager diff --stat d575db0`. The files you will touch most:

| File | Role |
| --- | --- |
| `autogram/discovery/evaluate.py` | The evaluator. Acceptance predicate, definitions, conjunctions, `SUSTAINED`, threshold fitting, condition-support floors. |
| `autogram/dsl/evaluate.py` | Grounding. `Grounded`, `ground()`, cross-grain related aggregates, `_partition_values`. |
| `autogram/calibrate.py` | Calibration driver, `_split_known`, capability tiers, null-control gating. |
| `autogram/discovery/known.py` | Known-invariant signatures, canonicalisation, `recover_known`. |
| `autogram/discovery/archive.py` | Pareto archive, redundancy suppression, lag-shadow suppression. |
| `autogram/discovery/propose.py` | Enumeration, conditioning, conjunction tiers. |
| `autogram/loader/gtib.py` | GTIB ingestion, profiling, shard materialisation. |
| `docs/autogram_guarantees.md` | The published guarantees. Must stay true to the code. |

---

## 4. Verification status

### 4.1 Last clean measurement (end of round 26 work, before the round-27 changes)

- Full test suite: **415 passed, 0 failed**.
- `artifacts/gtib_report.json`: 21/21 known invariants recovered, `recall_all` 1.0, `recall_validation` 1.0, false discovery `{equalities: 0, temporal: 0, definitions: 0}`.
- `artifacts/gtib_raw_report.json`: 2/2 recovered, same recall and false-discovery figures.
- Both reports' `provenance.engine_source_sha256` matched the live `autogram.calibrate._engine_source_fingerprint()`.

### 4.2 Current state — confirmed

The round-27 fixes are in the tree and the full verification completed cleanly:

- Full test suite: **419 passed, 0 failed** (2:52:25).
- `artifacts/gtib_report.json`: **21/21** recovered, `recall_all` 1.0, `recall_validation` 1.0, false discovery `{equalities: 0, temporal: 0, definitions: 0}`, no missed knowns.
- `artifacts/gtib_raw_report.json`: **2/2** recovered, same figures.
- Both reports' `provenance.engine_source_sha256` is `204888b9695ff02a0a9e1391c6e5d04611a791d069d2c3c87bb368b53a3f93af`, matching the live `autogram.calibrate._engine_source_fingerprint()`. The reports are current, not stale.

Treat the earlier `417 passed / 2 failed` run as superseded — both failures were the restore-bug corruption described in §2.1, not design defects. The corrupted line was restored and the fix independently red-green verified with a byte-identical-restore assertion.

**The four items in §7 are open against this otherwise-green tree.** A green suite does not mean the work is done: every one of the round-28 findings describes a defect that the current tests do not catch, which is precisely why the review loop continues.

### 4.3 Hygiene

`git --no-pager diff --check d575db0` exits 0.

---

## 5. How to run things

### 5.1 Environment

- Use the venv: `.venv\Scripts\python.exe` and `.venv\Scripts\autogram.exe`. **There is no `python -m autogram` entry point.** The system Python lacks pandas.
- **Cap all jobs at 4 CPU cores.** The machine is shared. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and pin affinity:

```powershell
$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'; $env:NUMEXPR_NUM_THREADS='4'
$p = Start-Process -FilePath .venv\Scripts\python.exe `
     -ArgumentList @('-m','pytest','-q','-p','no:cacheprovider','tests') `
     -NoNewWindow -PassThru -RedirectStandardOutput fs.out -RedirectStandardError fs.err
$p.ProcessorAffinity = 15     # cores 0-3; children inherit this
$p.Id | Out-File fs.pid -Encoding ascii
```

- **Always** set `AUTOGRAM_SUBAGENT_TIMEOUT=900` and `AUTOGRAM_SUBAGENT_CACHE=1` for anything that triggers subagent schema induction. The default 300 s timeout causes spurious failures.
- `Start-Process` returns a launcher PID; the real worker is a grandchild. Use `Get-CimInstance Win32_Process -Filter "ParentProcessId=$pid"` to find it. A launcher exiting does **not** mean the job finished.
- stdout is buffered until the process ends, so an output file of size 0 is normal for a long run.
- PowerShell here does not support `&&`, `||`, `?.`. Use `;` and `if ($?) { }`.

### 5.2 Full test suite

~2.5–3 hours. `tests/test_gtib_end_to_end.py::test_gtib_phase_ladder_climbs_held_out_recall_with_zero_null_acceptances` alone is ~1.5 hours by design.

### 5.3 Calibration reports

~1.5–2.5 hours each. They can run concurrently with the suite on the same 4-core mask.

```powershell
$env:AUTOGRAM_SUBAGENT_HARNESS='copilot'; $env:AUTOGRAM_SUBAGENT_CACHE='1'
$env:AUTOGRAM_SUBAGENT_TIMEOUT='900'
$env:AUTOGRAM_SUBAGENT_LOG='artifacts\subagent_gtib.jsonl'
.venv\Scripts\autogram.exe calibrate --config configs\gtib.yaml     --out artifacts\gtib_report.json
.venv\Scripts\autogram.exe calibrate --config configs\gtib_raw.yaml --out artifacts\gtib_raw_report.json
```

Give each concurrent job a **distinct** `AUTOGRAM_SUBAGENT_LOG`, or their JSONL appends interleave.

### 5.4 Acceptance check

Any source change alters the engine fingerprint, so the reports must be regenerated and re-checked:

```python
import json, autogram.calibrate as C
live = C._engine_source_fingerprint()
for path in (r"artifacts\gtib_report.json", r"artifacts\gtib_raw_report.json"):
    r = json.load(open(path, encoding="utf-8"))
    assert r["recall_all"] == 1.0 and r["recall_validation"] == 1.0
    assert all(v == 0 for v in r["false_discovery"].values())
    assert r["provenance"]["engine_source_sha256"] == live      # else the report is STALE
```

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

## 7. Remaining TODO items

These are the four Important findings from **review round 28**. None has been fixed. They are ordered by my judgement of severity. Each was reproduced by the reviewer; I confirmed each against the code but did **not** independently re-derive the reviewer's numbers, so treat those figures as reported-not-verified and reproduce them first.

---

### TODO-1 — Arithmetic overflow is silently treated as missing data (soundness)

**Severity:** Important. This is the most serious of the four, because it can let a **false discovery** through, which is the one thing the engine is not allowed to do.

**Sites:** `autogram/dsl/evaluate.py` lines ~121–127 (the `A.Div` branch) and ~395–402 (the finite mask in `ground()`).

**What is wrong.** `A.Div` guards division by exact zero (`np.where(den == 0.0, np.nan, num / den)`), but a division by a *finite but tiny* denominator overflows to `±inf`. `ground()` then applies

```python
mask = np.isfinite(rho) & np.isfinite(scale)
```

which drops every overflowed row from the population. Those rows vanish silently: they are not counted as violations, and the reported `support` is derived from `n_bindings / n_candidates × condition_support`, not from the number of rows that survived the finite mask.

**Why it matters.** The reviewer reports a false ratio law accepted on 20 of 100 rows while the other 80 overflowed and were discarded, with a reported support of `1.0`. That is a false discovery presented with full confidence. It also makes the Wilson bound meaningless, because `n` no longer reflects the population the rule claims to describe.

**Suggested fix.**
- Distinguish "the term is undefined here" (genuinely missing → drop) from "the arithmetic overflowed" (the rule's own expression blew up → this is a **violation**, or a reason to reject the rule outright). Overflow is a property of the candidate, not of the data.
- Consider rejecting a candidate whose evaluation overflows on more than a negligible fraction of rows, with a fail-loud reason string, rather than quietly shrinking its population.
- Whatever the policy, make the *reported* support reflect the rows actually graded. Compare with `Grounded.graded_points` / `graded_condition_support`, which were added for the conditioned case and already measure the graded population before subsampling — the same idea needs to apply here.

**Tests to add.**
- A ratio with a denominator that underflows towards zero on most rows: assert the rule is not accepted, and that the reported support reflects the graded rows.
- Extreme-value tests around `float64` limits for `Mul`, `Div` and `Add` chains.
- Red-green each one.

---

### TODO-2 — "Negligible" sum members are decided by the median, so a bimodal term can be discarded (soundness)

**Severity:** Important. Also a false-discovery path, this time in the recall figure rather than in acceptance.

**Sites:** `autogram/discovery/known.py` lines ~472–502 (`_col_scale`, `_drop_negligible`) and ~605–650 (the `recover_known` matching path).

**What is wrong.** `_col_scale` returns the **median absolute value** of a column. `_drop_negligible` removes a summed member whose scale is negligible against the anchor. A column that is `0` on 51% of rows and `1000` on the other 49% has a median of `0`, so it is judged negligible and dropped from the signature.

**Why it matters.** The reviewer reports recall `1.0` for a law that is violated on 49% of rows. The known-invariant recall figure is the headline claim of this whole exercise; a canonicalisation that discards a materially non-zero term makes it dishonest. Note the docstring already claims the transform "strictly widens matching: anything that matched exactly still matches after canonicalizing" — that claim is only true if the dropped member really is negligible everywhere.

**Suggested fix.**
- Replace the median test with a **pointwise** one: a member may be dropped only if it is negligible on (essentially) every gradeable row — for example `max(|value|)` over gradeable rows, or a high quantile, measured against the anchor's scale.
- Keep the existing "never canonicalise an entire group away" guard.
- Re-check `_canonicalize`'s idempotence and the widening claim in the docstring after the change, and update the docstring if the property changes.

**Tests to add.**
- The bimodal counterexample: a member zero on ~half the rows and large on the rest must **not** be dropped, and the law must not be credited as recovered.
- A genuinely all-zero member must still be dropped (this is what makes `total == SUM(a)` and `total == SUM(a, z)` one relation — see TODO-3 and §7's note on `_split_known`).
- Red-green both.

---

### TODO-3 — `_ColumnScaleView` decodes cells differently from the runtime frame (held-out integrity)

**Severity:** Important.

**Sites:** `autogram/calibrate.py` lines ~280–298 (`_ColumnScaleView`) and ~724–729 (the `_split_known` call site).

**What is wrong.** `_ColumnScaleView` was introduced in round 27 so that `_split_known` could canonicalise signatures exactly as `recover_known` does. It reads columns with `pd.to_numeric(...)`. The runtime `Frame` does not: for CrossCheck-style data it decodes dict-valued cells. So the two disagree about a column's scale, and therefore about which sum members are negligible.

**Why it matters.** The whole point of the round-27 change was that the calibration and validation halves must not contain two spellings of one relation. If the view decodes differently from the runtime frame, aliases can still straddle the split — the reviewer reproduced this at seed 0. When that happens, tuning on the calibration half is tuning on a "held-out" invariant, and the calibration/validation gap stops being an overfitting alarm.

**Suggested fix.**
- Canonicalise using the **compiled runtime frame** rather than a bespoke view. The obstacle is ordering: `_split_known` currently runs before `build_dataframe_grammar`. Options, roughly in order of preference:
  1. Build the frame (or just the column-decoding part of it) earlier, and pass the real thing in.
  2. Reuse the exact decoding helper the `Frame` uses inside `_ColumnScaleView`, so the two cannot drift.
  3. If neither is practical, make `_ColumnScaleView` fail loudly on a cell shape it cannot decode, rather than silently coercing it to `NaN`.
- Whichever you pick, add an assertion or a test that the view and the runtime frame agree on column scales for a CrossCheck fixture.

**Tests to add.**
- A CrossCheck-style dict-cell fixture where the naive `pd.to_numeric` view and the runtime frame disagree; assert the split keeps the aliases together.
- Red-green it.

---

### TODO-4 — The identifier guard for condition inference is too weak (robustness / blow-up)

**Severity:** Important.

**Site:** `autogram/loader/gtib.py` lines ~169–179 (inside `infer_tabular_profile`).

**What is wrong.** Round 27 added two guards: reject a domain above `_MAX_CONDITION_DOMAIN` (64), and reject a near-unique column (`distinct > len(frame) // 2`). A column with 50 distinct values over 100 rows passes both: 50 ≤ 64, and 50 is not `> 50`. It is still an identifier repeated twice, not a regime label.

**Why it matters.** The reviewer reports 251,175 conditions generated before rule expansion. That is a combinatorial blow-up in the conditioned search space, which at best wastes hours and at worst trips the fail-loud ceiling and makes an ordinary CSV unusable — the same class of failure the round-27 fix was meant to eliminate.

**Suggested fix.**
- Require **meaningful per-value support**: each condition value should cover at least some minimum number or fraction of rows (a handful of rows per value is not a regime). This subsumes both existing guards and is the property actually wanted.
- Additionally, **pre-count the complete conditioned search space** before enumeration and fail loudly with a clear message if it exceeds the ceiling, rather than discovering it deep inside expansion.
- Keep the existing `_MAX_CONDITION_DOMAIN` mirror of `schema/compiler._MAX_CONDITION_DOMAIN`, so inference can never propose something the compiler will reject.

**Tests to add.**
- 50 distinct values over 100 rows must not be inferred as a condition.
- A genuine low-cardinality regime label (e.g. two or three values over many rows) must still be inferred.
- A pre-count test asserting the fail-loud path triggers before expansion.
- Extend `tests/test_gtib_ingest.py::test_high_cardinality_identifier_is_not_inferred_as_a_condition`, which already covers the near-unique and above-ceiling cases.
- Red-green each.

---

### TODO-5 — Re-verify and close the loop

After fixing TODO-1 … TODO-4:

1. Run the **full suite** (§5.2) and confirm 0 failures. Do not rely on targeted runs (§2.1).
2. **Regenerate both calibration reports** (§5.3) — any source change invalidates the fingerprint — and confirm the acceptance criteria in §5.4: 21/21 and 2/2 recovered, `recall_all` and `recall_validation` both 1.0, all three false-discovery counters 0, fingerprints matching the live engine.
3. Confirm `git --no-pager diff --check d575db0` exits 0 and delete scratch `*.out` / `*.err` / `*.pid` files.
4. Spawn **review round 29** (§2), fresh context, with the round-28 findings listed as "fixed, please audit specifically".
5. Continue until **two consecutive `DONE` verdicts**.

**Do not commit** unless explicitly asked. All work to date is deliberately uncommitted on `main`.

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
| 28 | The four items in §7. |

The pattern worth internalising: **most rounds found a defect in the previous round's fix.** Verify fixes empirically against the data rather than reasoning about them, and red-green every regression test.
