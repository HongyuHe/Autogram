# The Autogram calibration protocol

This document specifies, harness-agnostically, how a coding agent (the **Tuner**) adapts Autogram
to a new dataset and reports how many known invariants it recovers. It is the contract behind
`autogram calibrate`; `AGENTS.md` is the operational summary any coding CLI can follow.

## Roles

| Role | Who | Responsibility |
|------|-----|----------------|
| **Tuner** | the user's coding harness | proposes generic-knob changes and re-runs the loop |
| **Discovery reporter** | `autogram/discovery/known.py` | deterministically scores recovery of known invariants |
| **Engine** | `autogram/discovery/*` | induces a `GrammarSpec`, enumerates, screens with Z3, scores hold-rate |

The Tuner never edits engine code and never hard-codes the user's invariants. It only edits
**generic knobs** and the declarative **`RegimeSpec`** (proxy shapes + regimes).

## Inputs

1. **Harness** (`AUTOGRAM_SUBAGENT_HARNESS`: `copilot`|`codex`|`claude`).
2. **OpenAI API key** (`OPENAI_API_KEY`), optional — only for `--schema-backend openai`.
3. **Known invariants** (`known_invariants.yaml`), relations over your real variable names.
4. **Max iterations** (default: run to completion).

## The loop

1. `autogram precheck` — verify the runtime can induce schemas (harness/subagent/key).
2. Split the known invariants into a **calibration** subset (guides tuning) and a **validation**
   subset (reports honest recall; never tuned against).
3. **Tune generic knobs on the wired proxy suite** (`RegimeSpec`): the approximate-offset proxy
   and the null control are drawn from the RegimeSpec; the tuning grid is adaptive and expands if
   no operating point fits, but the threshold never drops below the data null floor.
4. **(Re-)induce a grammar and discover** on the user's data. The per-candidate **adaptive band**
   is the default — each candidate law is judged at a tolerance fit from its own residuals.
5. **Walk the relaxation ladder** (generic knobs only): exercise the adaptive band, lower the
   threshold toward the null floor, then fall back to a fixed global band (which helps
   systematic-offset laws whose whole population sits at one scale). Score calibration-split recall
   after each rung and stop early at full recall.
6. **Re-induce with widened capabilities on stall.** If relaxing the numeric knobs stops improving
   recall, re-propose the grammar with a higher capability floor (all aggregations, then
   products/ratios via `max_degree=2`) — a missed law can be a vocabulary gap, not a threshold gap.
7. **Report and persist.** Report recovery via the Discovery reporter (per-invariant recovered/missed,
   aggregate recall on the held-out validation split, the number of grammar re-inductions, and a
   false-discovery figure), and **save the learned invariants by default** to
   `rules/<name>_<timestamp>.dl` plus a `learned_invariants` list in the JSON report.

Defaults on: the per-candidate adaptive band (`--band-mode global` opts out when you have a strong
band prior), the wired `RegimeSpec` proxy suite, and grammar re-induction (`--max-capability-tiers`).

## What the Tuner may and may not change

**May:** tolerance/band mode, hold-rate threshold, CI level, complexity/degree caps, aggregation
set, role-exclusion blocklist, and the declarative `RegimeSpec` (shape + regime per proxy).

**May not:** the user's specific invariants (no per-invariant special-casing), the held-out
validation split, or the always-on null proxy (the false-discovery control).

## Guarantees & limits (state these in every report)

- **Recall subject to a false-discovery ceiling** — recall is only meaningful alongside a low
  null-acceptance figure. Never maximize recall alone.
- **Proxies are abstractions, not copies** — a proxy shares a *shape and regime* with a real
  invariant, planted on fresh synthetic entities; it never references the user's real variables.
- **Known-recall is a lower bound under representativeness**, not a guarantee of discovering
  *unknown* invariants: an invariant may be missed if it is weaker than the tuned threshold
  (F1), lives at a different tolerance scale (F2), is not expressible in the grammar (F3), or if
  the proxy suite is unrepresentative of the domain (F4).
