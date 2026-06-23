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
- `cli.py` -- the `run`, `dump-prompt`, and `clean` entry points.
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


## Tests

```
uv run python -m pytest tests -q
```

The suite is fully offline and mockable -- it covers DSL evaluation, the objective / band / gate, the proposer interface and its leakage guard, and the portfolio assembler.
No live API key is required.


## Cleaning up run artifacts

Each `run` / `dump-prompt` regenerates `artifacts/subagent_prompt_<dataset>.txt`.
Remove those generated prompts (the committed `subagent_response_*.json` replies are preserved) with:

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

- `eval.target_coverage`, `eval.eps_exact`, `eval.eps_max`, `eval.gate_k`, `eval.lift_min` -- inner-loop band fit and the noise gate.
- `search.iterations`, `search.islands`, `search.thompson`, `search.assemble_k_max`, `search.dedup_rel` -- middle-loop evolutionary search and portfolio assembly.
- `grammar.proposer`, `grammar.rounds`, `grammar.max_complexity` -- outer-loop grammar / LLM cadence.

CLI flags override config values; run `uv run autogram run -h` for the full list (note the flag is `--iters`, not `--iterations`).


## Note on the OpenEvolve relationship (documented deviation)

The design doc recommends building on a battle-tested evolutionary substrate such as OpenEvolve.
This PoC takes the **insights** of that lineage -- a quality-diversity archive (MAP-Elites), island populations with periodic migration, and an LLM proposer in the loop -- but **reimplements a minimal version** rather than vendoring OpenEvolve.
The reason is the representation decision in Sections 6.5-6.7: OpenEvolve evolves raw program text, whereas our genotype is a typed, total DSL AST.
A from-scratch minimal substrate keeps the genotype, the analytic epsilon fit, and the leakage-controlled proposer interface first-class, and keeps the PoC dependency-light and fully offline-testable.
Where the implementation fills gaps left underspecified by the doc, those assumptions are noted in code comments at the relevant module.
