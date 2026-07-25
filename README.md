# Autogram

Autogram finds invariants (laws that hold across many observations) directly from your data, using only the **variable names** and their observed **values**. It is built for **open-ended discovery**: finding rules you did *not* already know, on data you did *not* pre-tune for. An LLM reads your variable names and proposes a small typed vocabulary; the engine then exhaustively enumerates every candidate law expressible in that vocabulary, screens each one with the Z3 solver, and keeps only the laws the data actually supports. The LLM never sees the values and never proposes an answer — it only proposes an alphabet, so the discovered rules are the data's, not the model's.

## What Autogram discovers

Autogram works on any data you can express as **variable–value pairs**: a collection of observations, where each observation assigns a value to each named variable, and the variable names carry semantic structure (for example `flow_A_to_B`, `out_A`). Nothing about the method is specific to any one domain — the variables can be sensor readings, financial quantities, counters, measurements, or anything else you can name and observe repeatedly. From names and values alone Autogram recovers laws such as pairwise equalities, family sums (a reference variable equals the sum of a group of variables), zero constraints, presence pairings (two variables are populated together), one-sided bounds (non-negativity), and conservation balances.

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

Provide your data as a pickled pandas DataFrame, where each variable is a field (a DataFrame column) and each observation is a record (a DataFrame row):

```
uv run autogram discover --input path/to/your_data.pkl --name mydata
```

This induces a schema, enumerates and screens candidates, prints the accepted portfolio, and writes the learned rules to `rules/mydata_<timestamp>.dl` (pass `--no-save-rules` to skip). If your data needs a looser band for approximate laws, raise the tolerance and lower the acceptance threshold, e.g. `--tolerance 0.05 --hold-rate 0.62`.

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

Under the hood the loop tunes tolerance and threshold on synthetic proxies (means, not ends — never your real answers), (re-)induces the schema, then walks a fixed relaxation ladder, scoring recovery of your known invariants on a **held-out** split after each step and stopping early when recovery is complete. Three behaviors are on by default: **one shared global band** (a single fixed tolerance applied to every candidate; `--band-mode adaptive` opts in to a per-candidate band fit from each law's own residuals), a **proxy suite derived from your calibration-split invariant shapes** (the editable `RegimeSpec` supplies the positive tuning proxies on synthetic entities, and an always-on null control is added separately as the false-discovery guard), and **grammar re-induction** (if relaxing the numeric knobs stops improving recall, the loop re-proposes the grammar with widened capabilities — more aggregations, then products and ratios — because a missed law is sometimes a vocabulary gap, not a threshold gap). The report gives you the number of iterations, any grammar re-inductions, the final knobs, the selected proxy shapes with per-proxy recovery evidence (recovery, accepted count, compactness), the grid expansions and the selected null count, recall on the full and held-out splits, the count of accepted rules, and a false-discovery figure (`null_equalities_accepted`) from an always-on random-noise control. **Recall is only meaningful next to a low null-acceptance figure — never maximize recall alone.**

**Band mode.** `autogram calibrate` uses **one shared global band by default** — a single fixed tolerance applied to every candidate — which is simpler, can be more decisive when your invariants really do share a scale, and is noticeably **faster** on large datasets. Pass `--band-mode adaptive` to instead fit a separate, capped tolerance to each candidate from its own residuals, which suits data where different invariants tolerate different amounts of slack (the adaptive band cost roughly 2–3× the wall time of the global band in our runs). The standalone `DiscoveryConfig` default is the opposite — adaptive — but the calibration loop defaults to the shared global band.

## Writing your known-invariants file

Describe each known invariant as a **relation over your real variable names** — no knowledge of Autogram's internals is needed. YAML or JSON both work; the file has a single `invariants:` list, and each entry is `{name, op, lhs, rhs}`.

| Shape | `op` | `rhs` | Meaning |
|-------|------|-------|---------|
| pairwise equality | `~=` or `==` | a variable name | `lhs` equals another variable |
| reference = family sum | `~=` or `==` | `{sum: [varA, varB, ...]}` | `lhs` equals the sum of a group of variables |
| zero | `~=` or `==` | `0` | `lhs` is structurally zero |
| presence pairing | `<\|>` | a variable name | `lhs` and `rhs` are populated together / absent together |
| one-sided bound | `>=` or `<=` | `0` | `lhs` is non-negative / non-positive |

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
| `calibrate` | Tune knobs, (re-)induce, iterate, report known-invariant recall | `--input`, `--known`, `--band-mode`, `--max-iterations`, `--max-capability-tiers`, `--validation-frac`, `--out` |
| `discover` | Induce + enumerate + screen a single dataset (or synthetic data) | `--input`, `--tolerance`, `--hold-rate`, `--max-complexity`, `--name`, `--json`, `--no-save-rules` |
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

`discover` and `calibrate` print a human-readable portfolio and write a machine-readable JSON report (`--json` / `--out`). Both also **save the learned invariants by default** as a `.dl` file under `rules/<name>_<timestamp>.dl` (pass `--no-save-rules` to skip, or `--rules-dir` to change the location); `rules/` is git-ignored because these are regenerated every run. The `calibrate` JSON additionally contains the full `learned_invariants` list (each rule with its hold-rate, confidence interval, and fitted tolerance), the path of the saved `rules_file`, the iteration trajectory, the final knobs, recall on the full and held-out splits, and the null-acceptance figure — so the invariants you learned are always persisted in both a readable file and the report.

## For coding agents

Autogram is designed so an agent can drive the whole loop unattended. Two documents formalize the protocol: `AGENTS.md` is the operational playbook (how to run the loop, which knobs the tuner may and may not touch), and `docs/calibration_protocol.md` is the harness-agnostic contract (inputs, per-iteration steps, the calibration/validation split, and the honest-limits rule). An agent should read those, generate a `known_invariants.yaml` from the user's domain knowledge, and iterate `calibrate` until held-out recall plateaus under a low null-acceptance ceiling.

## What it can and can't do

Autogram recovers laws that are **expressible in the induced grammar** and **statistically distinguishable from coincidence** at the tuned band. It will *not* recover a true law that holds on too small a fraction of observations to separate from noise (an information wall, not a tuning error), that lives outside the grammar's expressiveness (raise the complexity/degree caps or operator set to reach it), or that depends on variables absent from the data. Exhaustive enumeration also over-generates, so expect some redundant near-equalities in the portfolio (for example a genuine sum-conservation law padded with a tiny extra term). Known-invariant recall is therefore a **lower bound under representativeness** — evidence the tuning is sound, not a guarantee that every unknown law will be found.

## Test

```
uv run pytest tests -q
```

Tests exercise the real subagent induction path by default, so the harness must be installed and authenticated. Autogram is version 0.4.0.
