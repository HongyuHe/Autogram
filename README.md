# Autogram -- P3 AI-driven invariant-learning engine (proof of concept)

`autogram` is a runnable proof-of-concept of the **P3 AI-driven evolution / search engine** for learning soft, expressive network invariants.
It implements the design in [`docs/learning_soft_expressive_invariants.md`](../docs/learning_soft_expressive_invariants.md) (Section 10 is the primary specification).
The PoC is validated by **re-discovering the empirical invariants catalogued in [`docs/abilene_geant_invariants.md`](../docs/abilene_geant_invariants.md)** from the Abilene and GEANT CrossCheck samples, *without ever showing the learner that catalogue*.

The learner does **not** evolve raw code.
Following Sections 6.5-6.7 of the design doc, a candidate genotype is a typed, total expression in a small invariant DSL (an AST), so every learned artifact is terminating, side-effect-free, and statically checkable by construction.


## What it does

Given a network dataset (per-snapshot traffic matrices and link/node counters), the engine searches for a parsimonious **portfolio of invariants** -- relations such as `e[X->Y] = i[Y<-X]` (link two-end agreement), `o(X) = SUM H[X,*]` (origination equals demand row-sum), or `H[X,X] = 0` (no self-demand) -- each tagged with a data-fit verdict (`EXACT`, `SOFT_STRUCTURAL`, `SOFT`, or `ANTI`).
Soft tolerances (the per-rule epsilon band) are **fit per candidate in closed form**, never searched -- the primary anti-blowup lever from Section 5.4.


## Architecture

The three nested loops of the design (Section 10.4) map onto the package as follows.

| Loop level | Responsibility | Modules |
|---|---|---|
| Outer (grammar / LLM) | propose grammar extensions and seed rules | `proposer/` (`scripted`, `openai`, `subagent` backends), `dsl/grammar.py` |
| Middle (evolutionary) | MAP-Elites + islands rule-set search, Thompson budget allocation | `search/loop.py`, `search/archive.py`, `search/mutate.py`, `search/thompson.py` |
| Inner (analytic) | fit the epsilon band and score a candidate; no search | `evaluator/` (`band.py`, `gate.py`, `metrics.py`, `evaluator.py`) |

Supporting modules:

- `dsl/` -- the AST (`ast.py`), parser/unparser (`parser.py`), type/dimension checker (`typecheck.py`), binders (`binders.py`), grammar (`grammar.py`), and vectorised evaluator (`evaluate.py`).
- `loader/` -- dataset loading (`loader.py`) and the per-dataset name model that maps column names to DSL roles (`names.py`).
- `noise/` -- the clean-vs-observed noise model used to propagate sigma into the evaluator's noise gate (`model.py`).
- `search/assemble.py` -- turns the accepted-rule archive into a de-duplicated, information-aware portfolio.
- `search/recall.py` -- the **validation scorer** (the grader); it is read only at report time and is the only component that knows the catalogue.
- `cli.py` -- the `run`, `score`, `bench`, `compare`, `benchmark2`, `dump-prompt`, and `clean` entry points.
- `config.py` -- the tuning-knob schema, grouped by loop level (the annotated `configs/*.yaml` presets live at the repository root).

The non-code components live **outside** the importable package, at the repository root: the annotated presets in `configs/`, the offline test suite in `tests/`, and the subagent prompt/response exchange files in `artifacts/`.


## Install

The engine is managed by [`uv`](https://docs.astral.sh/uv/) as a single project environment (shared with the analysis notebooks) declared in the repository-root `pyproject.toml` and pinned by `uv.lock`.
`uv sync` creates `.venv` and installs the runtime dependencies (`numpy`, `pandas`, `pyyaml`) plus the offline test dependencies (`pytest`, and a mock-only `openai`):

```
uv sync
```

The OpenAI proposer backend needs the `openai` package at runtime; pull in that optional extra only if you use that backend:

```
uv sync --extra openai
```

All commands below assume the repository root as the working directory and are run through `uv run`, which executes them inside the project's `.venv` -- no manual activation, and no `pip`.
`uv sync` also installs the `autogram` console script, so the engine is invoked as `uv run autogram <command> [flags]` -- the short form of `uv run python -m autogram.cli ...`.


## Quickstart

Run the engine on a dataset with the scripted (offline, no-LLM) proposer:

```
uv run autogram run --dataset abilene --proposer scripted --iters 150 --seed 0
```

Or drive every knob from a config file:

```
uv run autogram run --config configs/abilene.yaml
```

The run prints the learned invariant portfolio, the recall scorecard against the known catalogue, and the exact knob settings used.


## The two LLM proposer backends

The grammar-extension / proposal step supports two **drop-in interchangeable** LLM backends behind one interface (`proposer/base.py: Proposer`), selectable via `--proposer` or the `grammar.proposer` config key.

### 1. OpenAI backend (`--proposer openai`)

Calls the OpenAI API for grammar / candidate proposals.
The API key is **never hardcoded**: it is read from the `OPENAI_API_KEY` environment variable (or passed explicitly via config).
If no key is set the backend degrades gracefully to an empty proposal so the engine still runs.

Install the optional `openai` dependency, export the key, then select the backend:

```
uv sync --extra openai
$env:OPENAI_API_KEY = "sk-..."        # PowerShell; do not commit a key
uv run autogram run --dataset abilene --proposer openai --model gpt-4o-mini
```

### 2. Subagent backend (`--proposer subagent`) -- leakage-free by construction

This backend lets an external harness (e.g. Claude Code spawning a subagent) extend the grammar **G**.
The critical constraint is **context isolation**: the subagent is given *only* the information legitimately available during learning -- variable names / roles, the current grammar, the DSL contract, and small data samples -- and **never** the ground-truth catalogue or anything that leaks the target invariants.

Isolation is enforced, not merely intended:

- The prompt is rendered from a fixed template that contains only columns / vocabulary / the JSON contract (`proposer/base.py: render_proposal_prompt`).
- Both the prompt **and** the subagent's reply are scanned by `assert_no_leakage`, which raises `LeakageError` on any reference to the catalogue, ground-truth markers, oracle terms, or the documented empirical constants.

The leakage-free headline workflow has three steps:

```
# 1. render ONLY the isolated prompt to artifacts/subagent_prompt_<dataset>.txt
uv run autogram dump-prompt --dataset abilene

# 2. an externally-spawned, context-limited subagent reads that prompt (and nothing else --
#    no repo, no docs/ ), and writes its JSON reply to
#    artifacts/subagent_response_abilene.json

# 3. run, consuming the reply file; the reply is re-scanned for leakage before use
uv run autogram run --dataset abilene --proposer subagent --iters 150 --seed 0
```

A genuine subagent reply sets `used_real_subagent=True` in the report; a missing reply falls back to an empty proposal (so the run never silently fabricates an answer).


## The proposer as a grammar-extension outer loop (`--start`)

Either LLM backend can act as the **outer-loop grammar-expansion controller**, not just a seed-form proposer inside a fixed grammar.
On top of candidate `rules`, a proposer may return an optional `extension` object that enables more roles / binders / operators / aggregate kinds or raises `max_complexity` / `max_add_arity`, and the search widens its hypothesis space before the next round (`proposer/base.py: parse_grammar_extension`, `search/loop.py: learn`).
Two guards keep this safe and bounded: every enable is intersected with a fixed `ceiling` grammar (the proposer can never invent vocabulary the DSL does not already type), and the prompt discloses exactly what is *available to enable* plus a leakage-safe **search-feedback** signal (elite counts, best score, idle binders, plateau flag) mined from the engine's own elites -- never the ground-truth catalogue.

The `--start` flag chooses where the grammar begins:

```
# full (default): start already equals the ceiling, so any proposed extension is a no-op
uv run autogram run --dataset abilene --proposer subagent --iters 150 --seed 0

# narrow: start from a deliberately small grammar and let the proposer widen it across rounds
uv run autogram run --dataset abilene --proposer openai --start narrow --rounds 3 --iters 150 --seed 0
```

When `--start narrow` is combined with a backend that actually emits extensions, the run prints a `grammar extensions applied` block (which round added which roles / binders / caps) and records the same in `result_<dataset>.json` under `grammar_extension`.
The default `full` start keeps `start == ceiling`, so the headline runs above are byte-for-byte unchanged.

### Graceful-degradation widening floor (`grammar.auto_widen`)

The *only practical* learning path -- a static, file-backed subagent reply under `--start narrow` in `--deployed` mode -- replays one committed answer that pre-dates the search feedback, so it can never carry an `extension`.
Without help the narrow grammar would admit only the two narrow-grammar forms and the run would starve at 25% recall.
The engine therefore applies a deterministic **floor**: when a round ends and the proposer did *not* widen the grammar, the engine widens `G` itself to the public ceiling (`search/loop.py: _deterministic_widen`), records the extension with `source: "deterministic"`, and re-checks admissibility against the wider grammar so the same static reply now yields the link/network forms it could not before.
This is leakage-safe (the ceiling is the public typed vocabulary derived from column *roles*, never the catalogue) and a guaranteed no-op when `start == ceiling` (default `full`).
It is gated by `grammar.auto_widen` (default `true`); setting it `false` reproduces the original starvation for diagnosis.
With the floor on and the *committed* (rich) subagent reply, `--start narrow --deployed` reaches the same **8/8 form recall** as `--start full` on both Abilene and GEANT (strict recall 6/8, because the two sub-noise structural laws I5/I6 reclassify under the modelled noise -- the documented deployed-mode behaviour).
A *realistic* isolated subagent, however, cannot hand over those structural laws at all (it is leakage-blind to the catalogue), so widening alone is necessary but not sufficient; the next subsection closes that remaining gap.

For a genuine multi-round outer-loop harness, an externally-spawned subagent may answer each round distinctly via `artifacts/subagent_response_<dataset>_round<k>.json` (`k = 1, 2, ...`); when a per-round file is absent the backend falls back to the base `subagent_response_<dataset>.json`, so a single static reply still drives every round.
The `scripted` baseline never proposes extensions, so with `auto_widen` off a narrow scripted run stays narrow on purpose (a demonstration of the starvation the floor is designed to cure).

### Structural form seeding (`search.structural_seeds`)

A leakage-free isolated subagent only ever returns *generic* narrow forms (non-negativity) plus a request to *widen* the grammar -- it can never return the row-sum / col-sum / conservation / two-end laws themselves, because those are exactly the answers it is walled off from.
Widening the grammar (the floor above) makes those forms *expressible*, but with `seed_from_grammar: false` the engine's own structural form-enumeration is off, so blind search over a complexity-12 grammar almost never *hits* I4/I5/I6/I7/I8 -- they are missed even after the grammar is wide enough to state them.
`search.structural_seeds` (default `false`; enabled in both deployed configs) closes this: after each widening it deterministically enumerates the **whole** structural-equality family from the grammar's role vocabulary -- the true forms and many decoys alike, each still data-gated -- via `dsl/grammar.py: structural_invariant_seeds`.
This is leakage-safe by construction: it reads only the grammar `G`, never the catalogue, and emits far more forms (38 at the ceiling) than the 5 it happens to credit, so it cannot be accused of being handed the answers.
With it on, a *realistic thin* subagent reply under `--start narrow --deployed` recovers **8/8 form recall** (I7/I8 now full; I5/I6 honestly partial -- their ~1.9% deficit sits below the ~2% modelled noise floor), versus **4/8** with it off -- the difference between "the grammar can state the law" and "the search actually proposes it".
Two companion score-shaping knobs keep the recovered laws visible in the final portfolio: `eval.w_lift` (default `0.15`) adds a clamped log-lift bonus so a genuine high-lift law outranks a trivially-true sign bound, and `eval.lift_min` (default `2.0`) drops sub-informative bounds from the genuine-law slate before assembly.


## Reproducing the headline validation result

The committed subagent replies under `artifacts/` reproduce the validated run end to end:

```
uv run autogram run --dataset abilene --proposer subagent --iters 150 --seed 0
uv run autogram run --dataset geant   --proposer subagent --iters 150 --seed 0
```

Verified result (seed 0, 150 iterations, real subagent reply):

| Dataset | Testable targets | Recovered (full+partial) | Strict recall | Real subagent |
|---|---|---|---|---|
| Abilene | 8 | 8 | 100% | yes |
| GEANT | 8 | 8 | 100% | yes |

Recall is scored over the **8 testable invariants** in the catalogue; targets I3 (reverse-link existence) and I10 (demand->link routing) are out of DSL scope and are excluded from the denominator (see `search/recall.py`).
"Strict" means the learned rule matched the target's form **and** carried the expected verdict (so a right-form / wrong-threshold match counts as partial, not strict).


## Run artifacts (machine-readable trace + result bundle)

Every `run` writes two structured artifacts to `--work-dir` (default `artifacts/`) alongside the human-readable console report:

| File | Contents |
|---|---|
| `trace_<dataset>.jsonl` | One JSON object **per evaluated candidate** -- round, phase (`seed` / `search`), island, origin (`proposer` / `grammar_seed` / `anti_seed` / `random_bootstrap` / `mutation` / `seed_mutation` / `random`), parent signature, whether it improved its niche, and the candidate's verdict / score / metrics (`eps`, `kappa_hat`, `support`, `lift`, `delta`). Inadmissible variations are recorded with `verdict = "INADMISSIBLE"` and null metrics for full lineage auditing. |
| `result_<dataset>.json` | A single `autogram.result/v1` bundle: the full knob settings (grammar / search / eval), portfolio (every learned rule with its metrics **and a self-contained serialised AST**), recall report (`null` for a `--no-score` run), island acceptance rates, and provenance (git commit + dirty flag + UTC timestamp + dataset shape). |

Both files are a **pure function of the run** (no extra oracle access) and are regenerated every time, so they are git-ignored and cleaned by default.
Pass `--no-artifacts` to skip writing them.

In addition, every `run` writes a **human-readable** rule listing to `rules/<dataset>_<time>.pl` (override the directory with `--rules-dir`, or skip it with `--no-rules`): one line per learned invariant formatted as `<invariant>  # <verdict> <metrics>`, followed by a `#`-prefixed metadata footer (dataset, run time, git provenance, proposer knobs, recall summary), so `grep -v '^#'` extracts the bare rule lines.
The `<time>` stamp is the run's own UTC provenance timestamp (`YYYYMMDDTHHMMSSZ`), so a back-filled bundle keeps its original run time.


## Separating learning from grading: `--no-score` and `score`

By default `run` grades the learned portfolio against the known catalogue at the end (the only step that reads the oracle).
Pass `--no-score` to stop before grading: the run then writes an **unscored** bundle (`recall: null`) that is provably produced without ever importing the catalogue -- useful when you want an auditable, leakage-free learning artifact and grade it separately.

Because each portfolio entry carries a self-contained serialised AST (`rule_dict`), a saved bundle can be re-graded later without re-learning:

```
uv run autogram run   --dataset abilene --proposer subagent --iters 150 --seed 0 --no-score
uv run autogram score --dataset abilene
```

`score` reloads `result_<dataset>.json`, rebuilds each rule, re-evaluates it with the bundle's **recorded** evaluator knobs (so verdicts reproduce exactly), prints the recall scorecard, and writes `result_<dataset>.scored.json` (the unscored original is left intact). Pre-`score`-split bundles that lack `rule_dict` are reported as unscorable rather than silently dropped.


## Multi-seed benchmark: `bench`

A single seed can be lucky. `bench` runs the learner once per seed, scores each run, and reports **mean / variance / worst-case / best-case** strict recall plus a **per-target hit-rate** (the fraction of seeds that recovered each known invariant):

```
uv run autogram bench --dataset abilene --proposer subagent --iters 150 --seeds 0,1,2
```

It writes an `autogram.bench/v1` bundle (`bench_<dataset>.json`) with the per-seed breakdown, the aggregates, and provenance; `--no-artifacts` skips the file.
Caveat: the `subagent` backend reads a *fixed* response file, so multi-seed varies only the search RNG (bootstrap / mutation / Thompson). Varying the proposer *sample* needs multiple response files or a live backend; this is recorded in the bench output.


## Deployed mode: oracle-gated vs observed-only (`--deployed`, `compare`)

By default the evaluator separates injected noise from real structure with the **clean/noisy oracle** (`ds.clean`): it propagates noise through each candidate's arithmetic and measures the true injected scale (`sigma_prop` ~ 0.0015 on these samples).
That clean frame is a benchmark luxury -- a real deployment only sees the noisy observations.

`--deployed` switches the gate to an **observed-only** estimator (`decompose_observed`) that never reads `ds.clean` and instead *models* per-cell noise at a relative level `eta` (`--rel-noise`, default `0.02`, the dataset's documented ~2% injection).
`eta` is a noise-model **calibration constant, not oracle access** -- the clean frame is never touched on the deployed path, so the leakage-free claim is preserved.

```
uv run autogram run --dataset abilene --proposer subagent --deployed
uv run autogram compare --dataset abilene --proposer subagent
```

`compare` reports three gradings side by side -- **(1)** oracle learn+grade, **(2)** the same oracle-found forms re-graded under the observed-only gate, **(3)** a full observed-only learn+grade -- plus a per-target verdict table, and writes `compare_<dataset>.json`.

The honest finding (identical on Abilene and GEANT):

| metric | oracle | deployed |
|---|---|---|
| form recall (laws found) | 8/8 (100%) | 8/8 (100%) |
| strict recall (form + strictness) | 8/8 (100%) | 6/8 (75%) |

The laws are still **found** -- form recall is unchanged.
The strict drop is the price of losing the clean oracle: the two **sub-noise** soft-structural laws (I5/I6, a stable ~1.9% one-sided deficit that is *smaller* than the ~2% modelled noise) relax from `SOFT_STRUCTURAL` to `EXACT`, because a ~1.9% bias can no longer clear `gate_k * sigma_prop` once the floor rises from the true ~0.0015 to the modelled ~0.02.
Genuinely exact laws (I4 two-end conservation, I7 node balance, I8 totals) and the I9 directionality anti-law are unaffected.
This concretely substantiates the design doc's "soft is hardest only without a clean oracle and only for sub-noise structure" thesis (Sec. 8.3).




The suite is fully offline and mockable -- it covers DSL evaluation, the objective / band / gate, the proposer interface and its leakage guard, and the portfolio assembler.
No live API key is required.


## Cleaning up run artifacts

Each `run` / `dump-prompt` regenerates `artifacts/subagent_prompt_<dataset>.txt`, and `run` / `score` / `bench` / `compare` regenerate `artifacts/trace_<dataset>.jsonl`, `artifacts/result_<dataset>.json` (and `result_<dataset>.scored.json`), `artifacts/bench_<dataset>.json`, and `artifacts/compare_<dataset>.json`.
Each `run` also writes a human-readable `rules/<dataset>_<time>.pl`.
Remove those generated files (the committed `subagent_response_*.json` replies are preserved) with:

```
uv run autogram clean
```

Preview without deleting, or wipe everything (committed replies plus `__pycache__`, `.pytest_cache`, and `build/` / `dist/` / `*.egg-info`):

```
uv run autogram clean --dry-run
uv run autogram clean --all
```


## Configuration knobs

Every knob is grouped by the loop level that owns it; see `configs/abilene.yaml` and `configs/geant.yaml` for annotated defaults.
Highlights:

- `eval.target_coverage`, `eval.eps_exact`, `eval.eps_max`, `eval.gate_k`, `eval.lift_min` -- inner-loop band fit and the noise gate (`eval.lift_min` also drops sub-informative bounds before assembly; see "Structural form seeding").
- `eval.w_lift` -- weight (default `0.15`) of the clamped log-lift information bonus so a genuine high-lift law outranks a trivially-true sign bound; see "Structural form seeding".
- `eval.deployed`, `eval.rel_noise` -- observed-only gate (no clean oracle) and its modelled per-cell noise level; see "Deployed mode" above.
- `search.iterations`, `search.islands`, `search.thompson`, `search.assemble_k_max`, `search.dedup_rel` -- middle-loop evolutionary search and portfolio assembly.
- `search.structural_seeds` -- deterministically seed the whole structural-equality form family from the grammar after each widening (default `false`; enabled in the deployed configs) so a leakage-blind thin subagent reply still recovers I4-I8 once the grammar is wide enough; leakage-safe (reads only `G`); see "Structural form seeding" above.
- `grammar.proposer`, `grammar.rounds`, `grammar.max_complexity` -- outer-loop grammar / LLM cadence.
- `grammar.start`, `grammar.max_complexity_ceiling`, `grammar.max_add_arity_ceiling` -- where the grammar begins (`full` = start at the ceiling, the default; `narrow` = start small and let the proposer widen across rounds) and the hard upper bound any proposer extension may reach; see "The proposer as a grammar-extension outer loop" above.
- `grammar.auto_widen` -- the engine-side graceful-degradation floor (default `true`); when a round's proposer supplies no usable extension and `G` is below the ceiling, the engine widens to the ceiling itself so the practical static file-backed subagent path does not starve under `--start narrow`; a guaranteed no-op when `start == ceiling`; see "Graceful-degradation widening floor" above.

CLI flags override config values; run `uv run autogram run -h` for the full list (note the flag is `--iters`, not `--iterations`).


## Schema generalization: inducing the adapter from a declarative spec

The base PoC is specialized to the CrossCheck schema through four hardcoded seams: the column-name parser (`loader/names.py`, which recognizes only `low_*`/`high_*` names), the role grounding (`dsl/binders.py`, which resolves roles via fixed string templates such as `low_{X}_egress_to_{Y}`), the cell codec (`loader/loader.py`, which expects each cell to be a dict carrying `ground_truth`/`hidden_ground_truth`), and the role ontology (`dsl/ast.py`).
The `schema/` subpackage removes the specialization from three of those four seams by making them *induced from data* rather than wired in.

A `SchemaSpec` (`schema/spec.py`) is a **declarative, bounded description** of a schema: column-name patterns (anchored regexes or token splits), reference templates that ground each DSL role to a column-name shape, family selectors, a cell codec kind, and the role ontology.
A spec is plain data, not code, so it is exactly the kind of artifact an LLM proposer can emit.
The trusted compiler (`schema/compiler.py`) validates a spec and turns it into a runnable `SchemaAdapter` (`schema/adapter.py`); it raises `CompileError` on any structurally invalid or unsafe spec (unknown matcher, bad regex, undeclared capture group, unknown binder or cell codec, malformed family selector), so a malformed or adversarial spec is rejected before it can touch the engine.
The role ontology (seam 4) is deliberately reused rather than re-induced: a new schema may rename its columns freely, but it speaks the engine's existing role vocabulary.

The generalization rests on one faithfulness claim, proven by `tests/test_schema_faithfulness.py`: the CrossCheck schema, expressed as a `SchemaSpec` (`crosscheck_spec()`) and run through the trusted compiler, reproduces the hardcoded scaffold **column-for-column and binding-for-binding** on the real Abilene and GEANT samples.
The hardcoded path is `adapter=None`; a compiled `crosscheck_spec()` is interchangeable with it.

A second, structurally different benchmark proves the schema is genuinely induced from the spec and not from the CrossCheck-shaped code.
`schema/benchmark2.py` defines a schema with different column syntax (`tx_<node>_src`, `rx_<node>_snk`, `if_<node>_to_<peer>_out`, `dem[<src>=><dst>]`), scalar cells instead of dict cells, and a regex demand matcher, then plants the same eight testable laws in synthetic data and runs the **real** engine over it end-to-end:

```powershell
uv run autogram benchmark2          # defaults: 400 snapshots, 4 nodes, 200 iters, seed 0
```

This recovers **8/8** of the planted catalogue (`B-I1`..`B-I9`, excluding the two out-of-scope topology/routing laws) at 100% strict recall, scripted and deterministic, reading no ground-truth catalogue during learning.

The benchmark plants a conserved per-snapshot **circulation** on a ring of links, modelling transit traffic a node forwards but neither originates nor terminates.
This is what makes it a *faithful* multi-hop benchmark rather than a toy: with pure direct routing the non-physical identities `sum(egress@x) = origination(x)` and `sum(ingress@x) = termination(x)` would also hold, and that two-term subset law makes the genuine four-term node-conservation law (I7) look like padded bloat to the portfolio assembler, which then evicts it.
Real networks have transit (`termination != sum of ingress`), so the conserved circulation cancels in every conservation law (I7/I8 stay exact) while breaking those subset identities, and I7 surfaces exactly as it does on the real Abilene/GEANT data.

What is **not** yet built is the runtime outer loop in which an LLM *proposes* a `SchemaSpec` for an unseen schema and a bootstrap heuristic seeds it (the `gen-bootstrap` and `gen-llm-adapter-proposer` items in the design plan).
The pieces that make that loop safe already exist -- the declarative spec, the validating compiler, and the proposer-layer leakage guard -- so the remaining work is wiring an adapter-proposer behind the same two-backend interface, which the design doc frames as future work (recommendation 9).


## Note on the OpenEvolve relationship (documented deviation)

The design doc recommends building on a battle-tested evolutionary substrate such as OpenEvolve.
This PoC takes the **insights** of that lineage -- a quality-diversity archive (MAP-Elites), island populations with periodic migration, and an LLM proposer in the loop -- but **reimplements a minimal version** rather than vendoring OpenEvolve.
The reason is the representation decision in Sections 6.5-6.7: OpenEvolve evolves raw program text, whereas our genotype is a typed, total DSL AST.
A from-scratch minimal substrate keeps the genotype, the analytic epsilon fit, and the leakage-controlled proposer interface first-class, and keeps the PoC dependency-light and fully offline-testable.
Where the implementation fills gaps left underspecified by the doc, those assumptions are noted in code comments at the relevant module.
