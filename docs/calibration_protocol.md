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
2. Split the known invariants into a **calibration** subset (guides tuning) and a **validation** subset (reports honest recall; never tuned against).
3. **Derive the positive proxy suite from the calibration subset only.** With no custom `RegimeSpec`, calibration reads only the *relation forms* of the calibration-split invariants — never the held-out validation split, and never domain words in the variable names — and plants one generic proxy per form on fresh synthetic entities.
4. **Map each form to a generic shape:** exact/approximate pairs, sums, zero, presence, one-sided bounds, ratios, proportionality, temporal deltas and windowed ratios, conditions, related-grain aggregates, sustained predicates, conjunctions, and categorical priority maps; a supported shape that the known-invariant file cannot express remains reachable through a custom `RegimeSpec`.
5. **Prepare each proxy once, then jointly tune.** Every selected positive proxy and an always-on null control are generated and schema-induced exactly once and then reused across one joint (tolerance, hold-rate threshold) grid, which expands toward the null floor on a stall and fails loudly with per-proxy evidence when nothing qualifies.
6. **Enforce the hard eligibility rule.** A setting qualifies only when every selected positive proxy meets its recovery target, every produced portfolio stays compact and free of scaled-slack inequality variants, and the null control produces zero accepted equalities; among qualifying settings the strictest wins (smallest tolerance, then highest threshold).
7. **(Re-)induce a grammar and discover on the user's data under one shared global band by default.** `--band-mode adaptive` is the opt-in mode that fits a separate, capped tolerance from calibration residuals and scores acceptance only on held-out residuals; the standalone `DiscoveryConfig` default stays adaptive, while `autogram calibrate` defaults to the shared global band.
8. **Walk the relaxation ladder** (generic knobs only): later rungs lower the threshold toward the null floor and widen the tolerance, every rung is re-checked on the *same* prepared null dataset so a rung that accepts even one false equality is disqualified and can never win, and calibration-split recall is scored after each rung with an early stop at full recall.
9. **Re-induce with widened capabilities on stall.** If relaxing the numeric knobs stops improving recall, re-propose the grammar through monotone capability tiers: all aggregations, products/ratios plus proportionality, grouped temporal operators, then related-grain and advanced Boolean/categorical definitions.
10. **Report and persist.** Report recovery via the Discovery reporter (per-invariant recovered/missed, aggregate recall on the held-out validation split, the number of grammar re-inductions, and a false-discovery figure), echo the selected proxy shapes with per-proxy recovery evidence (recovery fraction, accepted count, compactness, and any scaled-slack rules), the grid expansions, and the selected null-equality count, and **save the learned invariants by default** to `rules/<name>_<timestamp>.dl` plus a `learned_invariants` list in the JSON report.

Defaults on: one shared global band (`--band-mode adaptive` opts in to a per-candidate, capped self-calibrated band), the proxy suite derived from calibration-split shapes, a sign-balanced algebraic/band/bound null with independent condition labels and related-frame counters, a time-shuffled temporal null with independent condition labels, an independent-target advanced-definition null, and grammar re-induction (`--max-capability-tiers`). Null grammars cover the full supported capability surface at the configured lag, window, run-length, and conjunction bounds rather than only the proxy shapes present in the calibration split.

## What the Tuner may and may not change

**May:** tolerance/band mode, hold-rate threshold, CI level, complexity/degree caps, explicit nonlinear and linear leaf caps, an explicit conditioned-rule cap (`0` means exhaustive for each), aggregation set, lag/window/run-length bounds, conjunction arity, role-exclusion blocklist, and the declarative `RegimeSpec` (shape + regime per proxy).

**May not:** the user's specific invariants (no per-invariant special-casing), the held-out
validation split, or the always-on null proxy (the false-discovery control).

**Extending the suite.** A caller may pass a `CalibrationConfig.regime`, and the Tuner may add, adjust, or deactivate any of the supported generic proxy entries as a declarative edit.
Adding a genuinely *new* shape is a code change, not a knob: it requires extending the trusted generator, the recovery scorer, the known-invariant shape mapping (when the shape should be auto-derivable), and the tests.

## Guarantees & limits (state these in every report)

- **Recall subject to a false-discovery ceiling** — recall is only meaningful alongside a low
  null-acceptance figure. Never maximize recall alone.
- **Proxies are abstractions, not copies** — a proxy shares a *shape and regime* with a real invariant, planted on fresh synthetic entities, and never references the user's real variables.
  Proxy recovery constrains only the tuned knobs and never restricts which rules the real grammar may enumerate; conversely, a proxy suite that does not represent a real law's noise/support regime can still lead to settings that miss it.
- **Known-recall is a lower bound under representativeness**, not a guarantee of discovering
  *unknown* invariants: an invariant may be missed if it is weaker than the tuned threshold
  (F1), lives at a different tolerance scale (F2), is not expressible in the grammar (F3), or if
  the proxy suite is unrepresentative of the domain (F4).
