# Autogram

Autogram finds invariants (laws that hold across many observations) directly from your data, using only the **variable names** and their observed **values**. It is built for **open-ended discovery**: finding rules you did *not* already know, on data you did *not* pre-tune for. An LLM reads your variable names and proposes a small typed vocabulary; the engine then exhaustively enumerates every candidate law expressible in that vocabulary, screens each one with the Z3 solver, and keeps only the laws the data actually supports. The LLM never sees the values and never proposes an answer — it only proposes an alphabet, so the discovered rules are the data's, not the model's.

## What Autogram discovers

Autogram works on any data you can express as **variable–value pairs**: a collection of observations, where each observation assigns a value to each named variable, and the variable names carry semantic structure (for example `flow_A_to_B`, `out_A`). Nothing about the method is specific to any one domain: the variables can be sensor readings, financial quantities, counters, measurements, or anything else you can name and observe repeatedly. From names and values alone Autogram recovers pairwise equalities, family sums, zero constraints, presence pairings, one-sided bounds, conservation balances, ratios, learned proportionality, grouped temporal laws, conditional laws, cross-grain aggregates, sustained predicates, conjunctions, and categorical priority definitions.

Acceptance uses **no ground truth**. A candidate law is kept only if its **hold-rate** — the fraction of observations where `|residual| <= tolerance * scale` holds — clears a threshold with a Wilson confidence interval, and only if it survives Z3's logical screens (tautologies, contradictions, equivalences, and subsumed forms are removed). There is no hidden answer key; "progress" is never defined as "matched a rule we already knew".

## How it works

```
your variables  (flow_A_to_B, flow_B_from_A, out_A, ...)  observed many times
      |
      v
1. induce a GrammarSpec        (one LLM call: roles, families, kinds, directions, capabilities)
      |
      v
2. structural tokenizer        (entity boundaries rebuilt from the real variable names, not an LLM regex)
      |
      v
3. exhaustive enumeration      (every typed candidate law up to the complexity bound)
      |
      v
4. screen each candidate       (Z3 logical admissibility  +  data hold-rate with Wilson CI)
      |
      v
5. Pareto archive -> a non-redundant portfolio of accepted invariants (.dl files)
```

The key structural fact: the **grammar depends only on the variable names**, while the numeric knobs (tolerance, threshold, band) depend on the **values**. So the expensive LLM induction runs **once**, and the calibration loop re-scores the same grammar under many knob settings for free.

## Install

```
uv sync
```

Autogram runs inside a single uv-managed environment. Use `uv run autogram <command>` (the console script) or the equivalent `uv run python -m autogram.cli <command>`. The OpenAI backend is optional: `uv sync --extra openai`.

## Quickstart — discover on your own data

Provide your data as a CSV or pickled pandas DataFrame, where each numeric or Boolean variable is a field and each observation is a row:

```
uv run autogram discover --input path/to/your_data.pkl --name mydata
```

For grouped time series, declare the ordering and grouping columns, expected cadence, allowed windows, and condition columns: `uv run autogram discover --input series.csv --time-index timestamp --group-key tenant_id --cadence-seconds 60 --condition-column label --window 60 --max-lag 60`. An explicit cadence prevents uniformly missing periods from being mistaken for consecutive observations. Add `--advanced --run-length 10 --max-rules 200000` to enumerate sustained, conjunctive, and categorical definitions (advanced discovery requires a finite `--max-rules` budget). This induces a schema, enumerates and screens candidates, prints the accepted portfolio, and writes the learned rules to `rules/mydata_<timestamp>.dl` unless `--no-save-rules` is passed.

### GTIB example

The committed GTIB profile and known declarations are `configs/gtib.yaml`, `configs/gtib_known.yaml`, and `configs/gtib_raw_known.yaml`. Run the complete profile with `uv run autogram calibrate --config configs/gtib.yaml --out gtib_report.json`; explicit CLI flags override values loaded from the YAML file. The CLI also recognizes sibling `timeseries_raw.csv` and `events.csv` files automatically when the input is named `timeseries_derived.csv`. To persist a self-contained profiled pickle first, run `uv run python scripts/gtib_to_wide.py data/gtib-emulation/timeseries_derived.csv --raw data/gtib-emulation/timeseries_raw.csv --out gtib_derived_wide.pkl`.

## The main workflow — calibrate Autogram to your data

Different datasets need different numeric knobs (how much noise to tolerate, how often a law must hold to count). The **calibration loop** tunes those generic knobs for you and then reports how well Autogram recovers a set of invariants you already know — so you can trust it on the rules you *don't* know yet. You provide four things:

| Input | How | Required |
|-------|-----|----------|
| A coding harness | `--harness copilot` (default) / `codex` / `claude`, or env `AUTOGRAM_SUBAGENT_HARNESS` | yes |
| An OpenAI API key | env `OPENAI_API_KEY` with `--schema-backend openai` | only for the OpenAI backend |
| Your known invariants | `--known known_invariants.yaml` (relations over your variable names) | yes |
| A max-iteration budget | `--max-iterations N` (0 = run the whole relaxation ladder) | no (defaults to 0) |

Run it:

```
uv run autogram precheck --harness copilot      # verify the harness/subagent can induce schemas
uv run autogram calibrate \
  --input path/to/your_data.pkl \
  --known path/to/known_invariants.yaml \
  --out   report.json
```

Under the hood the loop tunes tolerance and threshold on synthetic proxies, re-induces the schema, walks a fixed relaxation ladder, and scores known-invariant recovery on a held-out split. One shared global band remains the default, while `--band-mode adaptive` fits a capped band per candidate. The proxy suite is derived only from calibration-split relation shapes and is checked against an algebraic/band null with independent condition labels and related counters, a time-shuffled temporal null that also enumerates conditioned and rolling-ratio hypotheses, and an independent-target definition null. Grammar re-induction widens capabilities monotonically through aggregations, products/ratios and proportionality, grouped temporal terms, and advanced related-grain and Boolean/categorical definitions. The report includes per-proxy evidence, all three null counts (`null_equalities_accepted`, `null_temporal_accepted`, and `null_definitions_accepted`), full and held-out recall, and the learned portfolio. **Recall is only meaningful when every null count remains zero.**

**Band mode.** `autogram calibrate` uses **one shared global band by default** — a single fixed tolerance applied to every candidate — which is simpler, can be more decisive when your invariants really do share a scale, and is noticeably **faster** on large datasets. Pass `--band-mode adaptive` to instead fit a separate, capped tolerance to each candidate from its calibration residuals and score acceptance only on held-out residuals, which suits data where different invariants tolerate different amounts of slack (the adaptive band cost roughly 2–3× the wall time of the global band in our runs). The standalone `DiscoveryConfig` default is the opposite — adaptive — but the calibration loop defaults to the shared global band.

## Writing your known-invariants file

Describe each known invariant as a **relation over your real variable names** — no knowledge of Autogram's internals is needed. YAML or JSON both work; the file has a single `invariants:` list, and each entry is `{name, op, lhs, rhs}`.

`==` declares machine-precision equality, while `~=` declares tolerance-band equality. A discovered exact law can recover an approximate declaration, while an approximate law cannot recover an exact declaration.

| Shape | `op` | `rhs` | Meaning |
|-------|------|-------|---------|
| pairwise equality | `~=` or `==` | a variable name | `lhs` equals another variable |
| reference = family sum | `~=` or `==` | `{sum: [varA, varB, ...]}` | `lhs` equals the sum of a group of variables |
| zero | `~=` or `==` | `0` | `lhs` is structurally zero |
| presence pairing | `<\|>` | a variable name | `lhs` and `rhs` are populated together / absent together |
| one-sided bound | `>=` or `<=` | `0` | `lhs` is non-negative / non-positive |
| ratio identity | `==` or `~=` | `{ratio: [numerator, denominator]}` | `lhs` equals a quotient |
| proportional equality | `~∝` | a variable name | `lhs` equals a robustly fitted multiple of `rhs` |
| temporal bound | `>=`, `>`, `<=`, or `<` | `0` | `lhs: {delta: variable}` constrains a grouped finite difference |
| windowed ratio | `==` or `~=` | ratio whose operands are `{roll_sum: [variable, window]}` | `lhs` equals a grouped trailing ratio of sums |
| conditional law | any supported relation | add `where: {column: value}` or `{column_in: [values...]}` | the relation is scored on one declared subset |
| related aggregate | `==` or `~=` | `{related: role}` | `lhs` equals a declared finer-grain join aggregate |
| Boolean definition | `:=` | `{sustained: ...}` or `{and: [...]}` | a Boolean target equals a sustained predicate or conjunction |
| categorical definition | `:=` | `{priority: [...], default: value}` | a categorical target equals a priority map over Boolean columns, with overlapping cases required to identify precedence |
| operating band | `~band` | `{center: value}` | `lhs` concentrates around a learned center, optionally under an `all` condition |

The example below uses generic entities `A`, `B`, `C` and a directed quantity `flow`; substitute your own variable names (whatever structure they carry):

```yaml
invariants:
  - name: two_end_agreement          # the A->B quantity equals the same link read at B
    op: "=="
    lhs: "flow_A_to_B"
    rhs: "flow_B_from_A"

  - name: total_is_group_sum          # a unit's total-out equals the sum of its outgoing flows
    op: "~="
    lhs: "out_A"
    rhs: { sum: ["flow_A_to_B", "flow_A_to_C", "flow_A_to_D"] }

  - name: zero_self_flow
    op: "=="
    lhs: "flow_A_to_A"
    rhs: 0

  - name: link_presence_symmetry
    op: "<|>"
    lhs: "flow_A_to_B"
    rhs: "flow_B_to_A"

  - name: nonnegative_flow
    op: ">="
    lhs: "flow_A_to_B"
    rhs: 0
```

List one entry per concrete grounding you want scored (for example, one `two_end_agreement` per directed pair of variables). See `docs/known_invariants.example.yaml` for a complete template.

## CLI reference

| Command | What it does | Key flags |
|---------|--------------|-----------|
| `precheck` | Verify the harness/subagent (or OpenAI key) can induce schemas before a run | `--harness`, `--schema-backend` |
| `calibrate` | Tune knobs, (re-)induce, iterate, report known-invariant recall | `--input`, `--known`, `--band-mode`, `--ci-alpha`, `--max-iterations`, `--max-capability-tiers`, `--max-nonlinear-leaves`, `--max-linear-leaves`, `--max-conditioned-rules`, `--validation-frac`, `--out` |
| `discover` | Induce + enumerate + screen a single dataset (or synthetic data) | `--input`, `--raw-input`, `--time-index`, `--group-key`, `--cadence-seconds`, `--condition-column`, `--window`, `--max-lag`, `--run-length`, `--advanced`, `--aggregation`, `--max-nonlinear-leaves`, `--max-linear-leaves`, `--max-conditioned-rules`, `--band-mode`, `--tolerance`, `--hold-rate`, `--ci-alpha`, `--name`, `--json` |
| `validate` | Run the synthetic-proxy self-check | `--seed` |
| `clean` | Remove generated discovery artifacts | `--out` |

All commands accept `--harness {copilot,codex,claude}` and `--schema-backend {subagent,openai}`.

## Harness and backend setup

The default schema backend spawns your **coding harness** as a subagent to induce the grammar; there is no offline fallback, so the harness must be installed and authenticated. Copilot is the default; Codex and Claude are supported via `--harness` or `AUTOGRAM_SUBAGENT_HARNESS`. The alternative `--schema-backend openai` calls the OpenAI SDK directly and needs `OPENAI_API_KEY` (and `uv sync --extra openai`).

| Environment variable | Purpose | Default |
|----------------------|---------|---------|
| `AUTOGRAM_SUBAGENT_HARNESS` | which coding CLI to spawn (`copilot`/`codex`/`claude`) | `copilot` |
| `OPENAI_API_KEY` | credential for `--schema-backend openai` | unset |
| `AUTOGRAM_SUBAGENT_MAX_ATTEMPTS` | induction retries before failing (repair + re-prompt on incomplete schemas) | `5` |

Induction is validated for completeness and self-repaired when a model truncates entity tokens; it retries up to `AUTOGRAM_SUBAGENT_MAX_ATTEMPTS` times and fails loudly rather than guessing.

## Outputs

`discover` and `calibrate` print a human-readable portfolio and write a machine-readable JSON report (`--json` / `--out`). Both also **save the learned invariants by default** as a `.dl` file under `rules/<name>_<timestamp>.dl` (pass `--no-save-rules` to skip, or `--rules-dir` to change the location); `rules/` is git-ignored because these are regenerated every run. The `calibrate` JSON additionally contains the full `learned_invariants` list (each rule with its canonical `rule_payload`, readable text, hold-rate, confidence interval, and fitted tolerance), the path of the saved `rules_file`, the iteration trajectory, the final knobs, recall on the full and held-out splits, and the null-acceptance figure, so the invariants you learned are always persisted in both a readable file and the report.

## For coding agents

Autogram is designed so an agent can drive the whole loop unattended. Two documents formalize the protocol: `AGENTS.md` is the operational playbook (how to run the loop, which knobs the tuner may and may not touch), and `docs/calibration_protocol.md` is the harness-agnostic contract (inputs, per-iteration steps, the calibration/validation split, and the honest-limits rule). An agent should read those, generate a `known_invariants.yaml` from the user's domain knowledge, and iterate `calibrate` until held-out recall plateaus under a low null-acceptance ceiling.

## What it can and can't do

Autogram recovers laws that are **expressible in the induced grammar** and **statistically distinguishable from coincidence** at the tuned band. It cannot recover a true law that is statistically indistinguishable from noise, requires an unavailable variable, exceeds the configured bounded grammar, or depends on an undeclared time/group/relation layout. Exhaustive enumeration can also retain redundant near-equalities. Known-invariant recall is therefore a **lower bound under representativeness**: it is evidence that tuning is sound, not a guarantee that every unknown law will be found.

## Test

```
uv run pytest tests -q
```

Tests exercise the real subagent induction path by default, so the harness must be installed and authenticated. Autogram is version 0.5.0.
