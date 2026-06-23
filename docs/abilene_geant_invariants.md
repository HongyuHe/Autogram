# Invariants Applicable to the Abilene & GÉANT Datasets

Scope: the two sample datasets under `./data/crosscheck-samples/`
(`abilene_sample_1000.pkl`, `geant_sample_1000.pkl`) — 1000 contiguous snapshots each,
Abilene (12 nodes) and GÉANT (23 nodes). This file lists **every structural invariant that
is applicable and testable on the columns actually present**, with a reproducible hold-rate
(the `threshold` column) and a plain-language meaning.

## How to read this table

- **`formula`** — the invariant stated over named columns, using the shorthand below.
- **`threshold`** — the **percentage of data points for which the invariant holds**, computed
  on the **clean** field (`hidden_ground_truth` for `low_*`; `ground_truth` for `high_*`
  demands, which have no clean field) so that the **injected measurement noise is excluded**.
  Reported as **Abilene / GÉANT**, with the tolerance stated in parentheses. The "data point"
  unit differs per invariant (cell, directed-link×snapshot, node×snapshot, snapshot, …) and is
  named in the `meaning` cell.
- **`meaning`** — what the constraint encodes, its strictness, and (where relevant) the
  CrossCheck-paper counterpart.

**Noise handling.** Each `low_*` cell carries a clean `hidden_ground_truth` and a noisy
`ground_truth = clean + injected noise`. All hold-rates below run on the clean field, so a
sub-100% threshold is a **genuine structural property of the data**, not an artifact of noise.
For one invariant (I4) the noisy hold-rate is also shown to make the clean-vs-noisy split
explicit. Tolerances are relative (`|residual| ≤ tol·|scale| + 1.0`); the `1.0`-byte floor only
matters for near-zero scales.

**Shorthand** (all per snapshot): `o_X = low_X_origination`, `t_X = low_X_termination`,
`e[X→Y] = low_X_egress_to_Y`, `i[X←Y] = low_X_ingress_from_Y`, `H[s,d] = high_s_d`.

## Invariant catalog & hold-rates

| ID | formula | threshold (% of data points holding, Abilene / GÉANT) | meaning |
|----|---------|--------------------------------------------------------|---------|
| **I1** Non-negativity | `v ≥ 0` for every `low_*`, `high_*` value `v` | **100% / 100%** (exact; per cell) | ✅ Every column is a byte/volume count, so negatives are impossible. 0 negative cells anywhere. |
| **I2** Zero self-demand | `H[X,X] = 0` ∀ X | **100% / 100%** (exact; per self-demand cell) | ✅ A node's traffic to itself never enters the backbone, so the demand-matrix diagonal is structurally zero (max\|·\| = 0). |
| **I3** Topology symmetry | `e[X→Y]` exists ⇔ `e[Y→X]` exists (equivalently every `e[X→Y]` pairs with `i[Y←X]`) | **100% / 100%** (exact; per directed link — 30 / 72) | ✅ Backbone links are bidirectional, so each directed counter has its reverse. All 30 (Abilene) / 72 (GÉANT) directed links are reverse-paired. Loose cousin of CrossCheck **CC‑1** (link status), but structural not operational. |
| **I4** Link two-end agreement | `e[X→Y] = i[Y←X]` | **100% / 100%** clean · 16.3% / 17.2% noisy (≤1e‑6 rel; per directed-link×snapshot) | ✅ `e[X→Y]` and `i[Y←X]` are two readings of the **same directed link** at opposite ends; identical on clean data (residual exactly 0). Every noisy violation is injected noise. **≡ CrossCheck CC‑2** (Eq. 2, `l_out^X = l_in^Y`); the tightest invariant. |
| **I5** Origination = demand row-sum | `o_X = Σ_d H[X,d]` | **66.9% / 66.5%** within ±5% (per node×snapshot) | ◑ Everything a node injects must be headed to *some* destination (the matrix row). Holds only **approximately**: a **systematic ~1.9% deficit** (`o_X / Σ_d H[X,d]` median ≈ **0.981 / 0.982**), so it clears ±5% for ~2/3 of points. See sensitivity table. Routing-free boundary corollary of CrossCheck **CC‑4**. |
| **I6** Termination = demand col-sum | `t_X = Σ_s H[s,X]` | **66.7% / 66.5%** within ±5% (per node×snapshot) | ◑ Mirror of I5: everything delivered at X came from *some* source (the matrix column). Same **systematic ~1.9% deficit** (median ratio ≈ 0.981 / 0.982). Routing-free corollary of CrossCheck **CC‑4**. |
| **I7** Node flow conservation | `o_X + Σ_Y i[X←Y] = t_X + Σ_Y e[X→Y]` | **~100% / ~100%** within ±1% (99.99% / 99.7%); 100% / 100% within ±3% (per node×snapshot) | ✅ Kirchhoff's law for traffic: a router neither sources nor sinks bytes over an interval, so inflow = outflow. Holds to ~0.15% mean; the tiny residual is inherited from the I5/I6 layer mismatch. **≡ CrossCheck CC‑3** (Eq. 3) under the convention that interfaces include origination/termination. |
| **I8** Network totals balance | `Σ_X o_X ≈ Σ_X t_X ≈ Σ_{s,d} H[s,d]` | `Σo≈Σt`: **95.1% / 94.3%** (±1%); `Σo≈ΣH`: **93.9% / 97.1%** (±5%) (per snapshot) | ✅/◑ Globally, total in ≈ total out (`Σo/Σt` median ≈ 1.000) — the sum of all I7 balances with internal links cancelling via I4. Origination vs. grand demand total carries the same ~2% deficit (`Σo/ΣH` median ≈ 0.978). Network-wide corollary of CrossCheck **CC‑3**. |
| **I9** Directionality (anti-invariant) | `e[A→B] ≠ e[B→A]` in general | **99.2% / 99.3%** of pairs differ by >1% (per undirected-pair×snapshot) | ✅ A deliberate *anti*-invariant: real traffic is asymmetric, so opposite directions of a link should **not** match. ~99% differ materially, confirming the counters are genuinely directional, not mislabeled duplicates. (Contrast I4, which is *equality* of the two ends of **one** direction.) |
| **I10** Demand → link routing | each `H[S,D]` routes over the topology into the `low_*` link loads | **not directly validated** (inferred; per demand) | ⚪ True by construction (demands were routed onto links to produce the counters), but the **routing/forwarding matrix is absent** from these files, so it cannot be reconstructed and tested here. Full form of CrossCheck **CC‑4** (Eq. 4, `l_demand = l_router`); I5/I6 are its testable boundary corollaries. |

Legend: ✅ holds (exact / within band) · ◑ holds approximately with a systematic offset · ⚪ applicable but not directly validated here.

*GÉANT contains **14 fully-empty snapshots** (all counters and demands = 0, i.e. missing
intervals); they are excluded from the per-snapshot ratio statistics (I8: 986 non-empty) and
are trivially satisfied elsewhere. Abilene has none.*

## Tolerance sensitivity for the approximate invariants (I5 / I6)

I5 and I6 are the only invariants whose hold-rate depends strongly on tolerance, because of
their **systematic ~1.9% structural deficit** (median ratio ≈ 0.981–0.982, i.e. counters run
just under the demand sums). The fraction holding therefore climbs steadily as the band widens:

| invariant | within ±3% | within ±5% | within ±10% |
|-----------|-----------|-----------|------------|
| I5 origination = row-sum (Abilene / GÉANT) | 46.3% / 46.3% | 66.9% / 66.5% | 89.2% / 88.6% |
| I6 termination = col-sum (Abilene / GÉANT) | 46.7% / 47.6% | 66.7% / 66.5% | 88.8% / 88.7% |

The ±5% column matches CrossCheck's `N = 5%` "equal?" convention; ±3% sits right at the median
deviation (~3.3%), which is why only ~46% clear it. Because the injected noise is **zero-mean**,
it cannot create a one-sided deficit — so this offset is a **real SNMP-counter-vs-traffic-matrix
mismatch**, identical in both datasets, and is exactly what keeps I7 from being exact.

## Out of scope (not in the table)

- **I11 Capacity bound** (`per-link load ≤ link capacity`) — **not applicable**: there is no
  capacity column and the counter units are ambiguous, so utilization cannot be formed. Also has
  no CrossCheck counterpart.
- **I12 Measurement-noise model** — a *characterization* of the injected perturbation
  (`(ground_truth − hidden)/hidden` ≈ zero-mean, std ≈ 2%, heavy-tailed, present on ~82% of
  `low_*` cells), not a pass/fail structural invariant, so it has no single hold-rate. It only
  sets the tolerance bands used above.
- **CrossCheck CC‑1 (link status, Eq. 1)** — **not applicable**: requires physical/link
  up/down status (`l_phy`, `l_link`) that these datasets do not contain.

## Method & reproducibility

- **Inputs.** `data/crosscheck-samples/{abilene,geant}_sample_1000.pkl`, read with
  `pandas.read_pickle`; each cell is a dict — the **clean** field (`hidden_ground_truth` for
  `low_*`, `ground_truth` for `high_*`) is used for all structural tests. Read-only; datasets
  unchanged.
- **Hold-rate.** For each invariant a per-point residual and scale are formed, and the threshold
  is `mean( |residual| ≤ tol·|scale| + 1.0 )` over the stated unit (cells / directed-link×snapshot
  / node×snapshot / snapshot / undirected-pair×snapshot). Exact invariants use `tol ≈ 0`
  (`≤ 1e‑6`/`1e‑9`); approximate ones use the relative bands shown.
- **Demand row/column sums** are computed by parsing `high_<src>_<dst>` against the node set and
  summing over destination (row) or source (column).
- The numbers above were produced by the validation harness in the session
  artifacts (`invariants.py`, `thresholds.py`); rerunning them on the same `.pkl` files
  reproduces every cell in the tables.

### Confirmed vs. inferred

- **Confirmed** (direct from the data): I1, I2, I3, I4, I5, I6, I7, I8, I9 hold-rates and the
  ~2% I5/I6 deficit; the directed-link counts (30 / 72) and the GÉANT empty-snapshot count (14).
- **Inferred / assumed**: I10 (true by construction but not reconstructable without the routing
  matrix); the interface convention that makes I7 ≡ CrossCheck CC‑3 (origination/termination
  treated as ingress/egress "links"); the byte/volume semantics of the counters (no units given).
