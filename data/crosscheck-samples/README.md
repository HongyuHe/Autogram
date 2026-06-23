# Network Telemetry Sample Data

Two sample datasets of network telemetry, drawn from the public **Abilene** (2004)
and **GEANT** (2005) research network traffic matrices. Each is the first 1000
contiguous time snapshots from our base dataset.

| File | Rows (snapshots) | Columns | Network |
|------|------------------|---------|---------|
| `abilene_sample_1000.pkl` | 1000 | 237 | Abilene (12 nodes) |
| `geant_sample_1000.pkl`   | 1000 | 681 | GEANT (23 nodes) |

## Format

Each file is a pickled **pandas DataFrame**. Read with:

```python
import pandas as pd
df = pd.read_pickle("geant_sample_1000.pkl")
```

Each **row** is one network snapshot at a given timestamp. Each **column** is one
measured quantity. Columns fall into three groups.

### Low-level counters (`low_*`)

Per-interface byte/packet counters. Naming:

- `low_<NODE>_egress_to_<NEIGHBOR>` and `low_<NODE>_ingress_from_<NEIGHBOR>` —
  directional link counters between two adjacent nodes.
- `low_<NODE>_origination` / `low_<NODE>_termination` — traffic entering/leaving
  the network at that node (external interfaces).

Counts: Abilene has 84 low-level columns; GEANT has 188.

### High-level demands (`high_*`)

Node-to-node traffic demands (the traffic matrix). Naming:
`high_<SRC>_<DST>`. Includes self-pairs (`high_X_X`), which are 0.

Counts: Abilene has 144 demand columns; GEANT has 484.

### Cell structure

Every `low_*` and `high_*` cell is a **dict** with these keys:

```python
{
    'hidden_ground_truth': 249339301.022,  # clean value, before measurement noise
    'ground_truth':        249318181.246,  # value with realistic measurement noise
    'perturbed':           None,           # value after a synthetic fault (unused here)
    'corrected':           None,           # value after our repair algorithm (unused here)
    'confidence':          None,           # repair confidence (unused here)
}
```

- `hidden_ground_truth` is the underlying clean value; `ground_truth` is that value
  with realistic measurement noise applied. They differ in ~84% of low-level cells.
  (For `high_*` demand cells, `hidden_ground_truth` may be `None` — only
  `ground_truth` is populated.)
- `perturbed`, `corrected`, and `confidence` are `None` throughout this sample —
  these are the **base** datasets, before any synthetic fault injection, repair, or
  validation. Use `ground_truth` (or `hidden_ground_truth`) as the data values.

### Metadata columns (9, non-`low_`/`high_`)

`timestamp`, `telemetry_perturbed_type`, `input_perturbed_type`,
`true_detect_inconsistent`, `repair_type`, `repair_confidence`, `validation_type`,
`validation_result`, `validation_confidence`.

In this base sample the perturbation/repair/validation fields are inert
(`'NONE'` / `False` / `None`); only `timestamp` carries data.

## Scale summary

| | Abilene | GEANT |
|--|---------|-------|
| Nodes | 12 | 23 |
| Low-level counter variables | 84 | 188 |
| High-level demand variables | 144 | 484 |
| **Total measured variables / snapshot** | **228** | **672** |
| Snapshots in sample | 1000 | 1000 |

Timestamps: Abilene 2004/03/01–2004/04/08, GEANT 2005/05/04–2005/05/25.
