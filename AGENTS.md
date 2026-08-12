# Autogram — agent playbook (calibration loop)

You are the **Tuner**. Your job: adapt Autogram's generic knobs to the user's dataset so it recovers their known invariants, then report the result. Autogram is the deterministic instrument; you supply the intelligence. Full protocol: `docs/calibration_protocol.md`.

## Inputs the user provides
1. **Harness** — set `AUTOGRAM_SUBAGENT_HARNESS` to `copilot` (default), `codex`, or `claude`. The subagent backend needs a harness that can spawn nested subagents.
2. **OpenAI API key** *(optional)* — set `OPENAI_API_KEY` and use `--schema-backend openai` only if your harness cannot spawn subagents.
3. **Known invariants** — a `known_invariants.yaml` (see `docs/known_invariants.example.yaml`): relations over the dataset's real variable names.
4. **Max iterations** — how many calibration rounds to run. Default: run to completion.

## Loop
1. **Preflight:** `autogram precheck --harness <h>`. Fix any issue before continuing.
2. **Calibrate:** `autogram calibrate --input data.pkl --known known_invariants.yaml --out report.json`. This tunes generic knobs on a proxy suite derived from your calibration-split invariant shapes (or a custom `RegimeSpec`), (re-)induces a grammar, walks a relaxation ladder under **one shared global band by default**, re-induces with widened capabilities if recall stalls, and reports recall of the known invariants (on a held-out validation split) with a false-discovery figure.
3. **Read the report.** If `recall_validation` is low, adjust **generic knobs only**: pass `--band-mode adaptive`, raise `--max-capability-tiers` to reach nonlinear, temporal, related-grain, or advanced logic tiers, adjust bounded lag/window/run-length/complexity controls, or add/adjust/deactivate entries in the `RegimeSpec` proxy suite. **Never** hard-code the user's specific invariants.
4. Stop when validation recall is satisfactory or the iteration budget is spent.

## Rules (do not break these)
- **Recall subject to false discovery.** Never maximize recall alone; keep `null_equalities_accepted`, `null_temporal_accepted`, and `null_definitions_accepted` at zero.
- **Held-out split.** Never tune against the validation invariants; the calibration/validation gap is your overfitting alarm.
- **Generic knobs only.** Tune tolerance/band, threshold, grid, aggregations, degree, and proxy shapes — not per-invariant special cases.
- **Honest limits.** Known-invariant recall is a lower bound under representativeness, not a guarantee of discovering unknown invariants.
