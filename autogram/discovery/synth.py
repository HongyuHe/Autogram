"""Synthetic tabular datasets with structured column names and known planted invariants.

These datasets exist so the discovery pipeline can be demonstrated and stress-tested *without
ground truth flowing into the engine*.  The generator plants relationships (two-end agreement,
demand row/column sums, zero self-demand, non-negativity); the *planted* structure is returned
separately and is used only by :mod:`autogram.discovery.validate` to judge recovery -- it never
reaches the inducer, proposer or evaluator.

The token spellings (kind/keyword/connector tokens and entity names) are all parameters, so a
consistent rename produces a structurally identical dataset -- the basis of the rename-invariance
check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Vocab:
    """The token spellings used to name columns (everything the inducer must discover)."""
    measurement: str = "measurement"
    demand: str = "flow"
    source: str = "source"
    destination: str = "destination"
    to: str = "to"
    frm: str = "from"
    entity_prefix: str = "n"

    def entity(self, i: int) -> str:
        return f"{self.entity_prefix}{i}"


@dataclass
class Synthetic:
    columns: List[str]
    matrix: np.ndarray                      # (T, d) observed values
    timestamps: np.ndarray
    vocab: Vocab
    entities: List[str]
    planted: Dict[str, object] = field(default_factory=dict)


def make_synthetic(n_entities: int = 6, n_snapshots: int = 400, noise: float = 0.0,
                   seed: int = 0, vocab: Vocab = None, unstable_frac: float = 0.0,
                   families=None, regime_factor: float = 1.6,
                   offset_hold_rate: float = 0.67, offset_factor: float = 0.98,
                   presence_rate: float = 0.6) -> Synthetic:
    """Generate a dataset with planted invariants and ``noise`` relative Gaussian noise.

    Planted (clean) relationships (the default core set includes row/column sums, two-end
    agreement, self-zero, and agg+ref balance; validation may enable one family at a time
    so recovery cannot be dominated by the easiest relation):

    * ``flow_i_i == 0``                         (zero self-demand)
    * ``measurement_i_source ~= sum_{j!=i} flow_i_j``     (origination = demand row sum)
    * ``measurement_i_destination ~= sum_{j!=i} flow_j_i``     (termination = demand column sum)
    * ``measurement_i_to_j == measurement_j_from_i``          (directed two-end agreement)
    * ``measurement_i_to_j ~= measurement_j_from_i``          (systematic-offset approximate pair)
    * ``exists(measurement_i_to_j) <=> exists(measurement_j_from_i)`` (presence pairing)
    * ``measurement_i_source + sum_j flow_j_i == measurement_i_destination + sum_j flow_i_j`` (agg+ref balance)
    * every column ``>= 0``                     (non-negativity)

    ``noise`` is applied to the measured (``measurement_*``) columns only; the demand matrix stays
    clean.  The engine never sees ``noise``; the self-calibrated band must track it.

    ``unstable_frac`` (in [0, 1)) plants a *regime/overfit* trap used only by the drop-stability
    ablation: the directed two-end agreement (``to_ij == from_ji``) holds on the first
    ``1 - unstable_frac`` of snapshots and then shifts to a different linear regime (the
    ``from_*`` side is scaled by ``regime_factor``) for the final ``unstable_frac``.  Because
    agreement still holds on the majority of rows, the pooled median residual stays small -- so
    the pooled residual can still look strong, yet its coverage at the
    rule's own tight tolerance collapses on the late time block, so it is admissible by every
    test *except* stability.  The demand row/column sums are untouched, so the genuinely stable
    invariants are unaffected.
    """
    vocab = vocab or Vocab()
    enabled = set(families or ("row_sum", "col_sum", "two_end", "self_zero", "agg_ref_balance"))
    rng = np.random.default_rng(seed)
    T, N = n_snapshots, n_entities
    ents = [vocab.entity(i) for i in range(N)]

    # demand tensor D[t, i, j] >= 0.  The diagonal is zero only when the zero-self family is
    # intentionally planted; row/column sums below always use off-diagonal demand, matching the
    # induced family selectors and avoiding accidental self-zero recovery in single-family runs.
    D = rng.gamma(shape=2.0, scale=10.0, size=(T, N, N))
    if "self_zero" in enabled:
        for i in range(N):
            D[:, i, i] = 0.0
    D_off = D.copy()
    for i in range(N):
        D_off[:, i, i] = 0.0

    # directed link value L[t, i, j] (i -> j).  The reverse-named side is varied by family:
    # exact equality, systematic multiplicative offset, shared presence mask, or independence.
    L = rng.gamma(shape=2.0, scale=8.0, size=(T, N, N))
    for i in range(N):
        L[:, i, i] = 0.0
    if "two_end" in enabled:
        Lfrm = L.copy()
    elif "offset_pair" in enabled:
        Lfrm = L.copy()
        n_clean = int(round(min(max(offset_hold_rate, 0.0), 1.0) * T))
        if n_clean < T:
            Lfrm[n_clean:, :, :] = L[n_clean:, :, :] * offset_factor
    elif "presence_pair" in enabled:
        mask = rng.random((T, N, N)) < min(max(presence_rate, 0.0), 1.0)
        for i in range(N):
            mask[:, i, i] = False
        L = L * mask
        Lfrm = rng.gamma(shape=2.0, scale=8.0, size=(T, N, N)) * mask
    else:
        Lfrm = rng.gamma(shape=2.0, scale=8.0, size=(T, N, N))
        for i in range(N):
            Lfrm[:, i, i] = 0.0
    if unstable_frac > 0 and "two_end" in enabled:
        n_break = int(round(unstable_frac * T))
        if n_break > 0:
            Lfrm[T - n_break:, :, :] = L[T - n_break:, :, :] * regime_factor

    cols: List[str] = []
    blocks: List[np.ndarray] = []

    def add(name: str, values: np.ndarray):
        cols.append(name)
        blocks.append(values.reshape(T, 1))

    # demand matrix columns
    for i in range(N):
        for j in range(N):
            add(f"{vocab.demand}_{ents[i]}_{ents[j]}", D[:, i, j])

    # single-entity measured columns: origination / termination (row/col sums)
    row_totals = D_off.sum(axis=2)
    col_totals = D_off.sum(axis=1)
    if "agg_ref_balance" in enabled:
        bias = rng.gamma(shape=2.0, scale=10.0, size=(T, N))
        shared_offset = 0.0 if ("row_sum" in enabled or "col_sum" in enabled) else bias
        orig = row_totals + shared_offset
        term = col_totals + shared_offset
    else:
        orig = (row_totals if "row_sum" in enabled
                else rng.gamma(shape=2.0, scale=10.0, size=(T, N)))
        term = (col_totals if "col_sum" in enabled
                else rng.gamma(shape=2.0, scale=10.0, size=(T, N)))
    for i in range(N):
        add(f"{vocab.measurement}_{ents[i]}_{vocab.source}", orig[:, i])
        add(f"{vocab.measurement}_{ents[i]}_{vocab.destination}", term[:, i])

    # directed measured columns: to_ij and from_ji share the same value (unless destabilised)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            add(f"{vocab.measurement}_{ents[i]}_{vocab.to}_{ents[j]}", L[:, i, j])
            add(f"{vocab.measurement}_{ents[j]}_{vocab.frm}_{ents[i]}", Lfrm[:, i, j])

    matrix = np.hstack(blocks)

    # apply relative noise to measured columns only (demand stays clean)
    if noise > 0:
        for k, name in enumerate(cols):
            if name.startswith(vocab.measurement + "_"):
                matrix[:, k] = matrix[:, k] * (1.0 + noise * rng.standard_normal(T))
        matrix = np.maximum(matrix, 0.0)

    all_planted = _planted(vocab, ents, N)
    planted = {k: v for k, v in all_planted.items() if k in enabled}
    ts = np.arange(T)
    return Synthetic(columns=cols, matrix=matrix, timestamps=ts, vocab=vocab,
                     entities=ents, planted=planted)


def make_null(n_entities: int = 6, n_snapshots: int = 400, seed: int = 0,
              vocab: Vocab = None) -> Synthetic:
    """Same structured names, but every column is independent noise (no relationships).

    A correct engine should accept ~no rules here (false-discovery control).
    """
    vocab = vocab or Vocab()
    rng = np.random.default_rng(seed)
    T, N = n_snapshots, n_entities
    ents = [vocab.entity(i) for i in range(N)]
    cols: List[str] = []
    blocks: List[np.ndarray] = []

    def add(name):
        cols.append(name)
        blocks.append(rng.gamma(2.0, 10.0, size=(T, 1)))

    for i in range(N):
        for j in range(N):
            add(f"{vocab.demand}_{ents[i]}_{ents[j]}")
    for i in range(N):
        add(f"{vocab.measurement}_{ents[i]}_{vocab.source}")
        add(f"{vocab.measurement}_{ents[i]}_{vocab.destination}")
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            add(f"{vocab.measurement}_{ents[i]}_{vocab.to}_{ents[j]}")
            add(f"{vocab.measurement}_{ents[j]}_{vocab.frm}_{ents[i]}")
    matrix = np.hstack(blocks)
    ts = np.arange(T)
    return Synthetic(columns=cols, matrix=matrix, timestamps=ts, vocab=vocab,
                     entities=ents, planted={})


def _planted(vocab: Vocab, ents, N) -> Dict[str, object]:
    two_end = set()
    agg_ref_balance = []
    self_zero = []
    row_sum = []
    col_sum = []
    for i in range(N):
        self_zero.append(f"{vocab.demand}_{ents[i]}_{ents[i]}")
        row = frozenset(f"{vocab.demand}_{ents[i]}_{ents[j]}" for j in range(N) if j != i)
        col = frozenset(f"{vocab.demand}_{ents[j]}_{ents[i]}" for j in range(N) if j != i)
        row_sum.append((f"{vocab.measurement}_{ents[i]}_{vocab.source}", row))
        col_sum.append((f"{vocab.measurement}_{ents[i]}_{vocab.destination}", col))
        agg_ref_balance.append(frozenset({
            (f"{vocab.measurement}_{ents[i]}_{vocab.source}", col),
            (f"{vocab.measurement}_{ents[i]}_{vocab.destination}", row),
        }))
        for j in range(N):
            if i == j:
                continue
            two_end.add(frozenset({f"{vocab.measurement}_{ents[i]}_{vocab.to}_{ents[j]}",
                                   f"{vocab.measurement}_{ents[j]}_{vocab.frm}_{ents[i]}"}))
    return {
        "two_end": two_end,
        "offset_pair": two_end,
        "presence_pair": two_end,
        "self_zero": self_zero,
        "row_sum": row_sum,
        "col_sum": col_sum,
        "agg_ref_balance": agg_ref_balance,
    }
