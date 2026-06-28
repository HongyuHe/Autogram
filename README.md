# Autogram -- open-ended discovery of unknown data invariants

`autogram` discovers **unknown** invariants in tabular data it was *never tuned for*, judging every
candidate from **data alone**: no oracle, no clean frame, no hand-built role vocabulary, and no
pre-tuned constants. Success is not catalogue recall -- it is a stable, parsimonious, name-grounded
portfolio of invariants confirmed on held-out data.

The learner never evolves raw code. A candidate genotype is a typed, total expression in a small
invariant DSL (an AST), so every learned artifact is terminating, side-effect-free, and statically
checkable by construction.

## What it does

Given a structured tabular dataset (columns whose names encode entities and relations), the engine:

1. **Induces a schema from column names** -- entities, relations/directions and families are
   *created* from name semantics (no fixed ontology). The offline inducer is a deterministic
   heuristic; the deployed inducer is an LLM behind the same interface.
2. **Proposes typed candidates** inside the induced schema (LLM proposer + random schema-typed
   mutation). There is no scripted backend, no seed enumeration and no planted decoys.
3. **Judges from data only** -- a self-calibrated tolerance band (coverage read off the residual
   knee), a name-permutation lift percentile (FDR control), cross-split and temporal stability, and
   MDL parsimony.
4. **Archives** accepted invariants in a simple Pareto archive keyed by `(binder, length)` on
   coverage x parsimony, mining its own elites for the next round and re-inducing the schema on a
   stall.

Every learned quantity (the band, its operating coverage, the acceptance bar) is derived from each
rule's own residuals. There is no declared noise scale `eta`, target coverage `kappa*`, `gate_k`,
`eps_exact`, `eps_max`, or `lift_min`.

## Architecture (one small loop)

```
column names + sample rows
        |  induce (heuristic / LLM)
        v
induced SchemaSpec (open roles)  --compile-->  SchemaAdapter
        |  grammar_from_adapter
        v
proposer (random mutation + LLM)  -->  ground on observed data
        |
        v
data-only evaluator: self-calibrated band | name-permutation lift | stability | MDL
        |
        v
Pareto archive (binder x length)  -->  own-elite progress  -->  loop (re-induce on stall)
        |
        v
portfolio of discovered invariants
```

Modules:

- `discovery/` -- `induce.py` (schema induction from names), `synth.py` (synthetic datasets with
  planted invariants), `evaluate.py` (data-only evaluator), `archive.py` (Pareto archive),
  `propose.py` (random/LLM proposer), `loop.py` (the discovery loop), `validate.py` (adversarial
  validation harness).
- `dsl/` -- the typed AST (`ast.py`), parser (`parser.py`), induced-schema type checker
  (`typecheck.py`), binders (`binders.py`), grammar (`grammar.py`), and vectorised grounding
  (`evaluate.py`).
- `schema/` -- the JSON-serialisable `SchemaSpec` interface (`spec.py`), the trusted compiler
  (`compiler.py`) and the compiled adapter (`adapter.py`). The spec is *induced*, never hand-written.
- `evaluator/` -- `band.py` (split-conformal band, now self-calibrating its coverage) and
  `metrics.py` (Wilson interval, name-permutation lift, MDL gain).
- `loader/` -- frames and dataset assembly (`loader.py`) and the schema-agnostic name model
  (`names.py`).
- `search/mutate.py` -- typed AST mutations inside the induced grammar.
- `config.py` -- statistical confidence levels and search budget (no tuned thresholds).
- `cli.py` -- the `discover`, `validate`, and `clean` entry points.

## Install

```
uv sync
```

## Use

Discover invariants on a synthetic dataset (offline, deterministic, no key):

```
uv run autogram discover --entities 6 --snapshots 400 --noise 0.02 --seed 0
```

It induces a schema, proposes typed forms and prints a stable, parsimonious portfolio -- e.g. the
two-end agreement `meas_X_to_Y == meas_Y_from_X` and origination = demand row-sum
`SUM(demand_row) ~= m_src` -- each with its held-out coverage (Wilson CI), name-permutation lift
percentile, stability and MDL gain. Add `--json out.json` to persist the portfolio.

Run the adversarial validation harness (plant-and-recover noise sweep, null dataset, tautology
rejection, rename invariance, ablations):

```
uv run autogram validate --seed 0
```

## Test

```
uv run pytest tests -q
```

All tests are synthetic and offline. They cover the kept primitives (typed DSL + grounding,
split-conformal band, the `SchemaSpec` compiler/adapter) and the new data-only behaviour (heuristic
schema induction, knee-based band/coverage, name-permutation lift percentile, stability, MDL
parsimony, the Pareto archive, and the adversarial checks).
