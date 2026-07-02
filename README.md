# Autogram -- guarantees-first invariant discovery

`autogram` discovers invariants in tabular data from observable names and values only.  v2 uses
hard guarantees wherever the problem is decidable: schema roles are induced by one of two LLM-style
backends (`subagent` or `openai`), the bounded DSL grammar is exhaustively enumerated, and Z3 removes
logical tautologies, contradictions, equivalences, and subsumed forms.  Data is used for one statistic
only: rule hold-rate with a Wilson confidence interval.  MDL is a tie-breaker among surviving rules.

## Architecture

```
column names + observed values
        |
        v
schema induction (subagent | openai)
        |
        v
bounded DSL grammar -> exhaustive enumeration
        |
        v
Z3 logical screens -> data hold-rate + Wilson CI -> MDL tie-break
        |
        v
.dl rules under rules\
```

There is no rule catalogue, clean-frame oracle, hidden ground truth, random mutation proposer, or
permutation reference test in the discovery loop.

## Install

```
uv sync
```

## Use

Synthetic run:

```
uv run autogram discover --entities 6 --snapshots 400 --noise 0.02 --seed 0
```

The default schema backend invokes the real long-context Copilot subagent transport. If Copilot
CLI is not installed/authenticated, schema induction fails hard rather than using a local parser.

CrossCheck sample run (observed `ground_truth` cells only; never `hidden_ground_truth`):

```
uv run autogram discover --input data\crosscheck-samples\abilene_sample_1000.pkl --name abilene
uv run autogram discover --input data\crosscheck-samples\geant_sample_1000.pkl --name geant
```

Both commands save discovered invariants as timestamped `.dl` files in `rules\` unless
`--no-save-rules` is supplied.

## Test

```
uv run pytest tests -q
```
