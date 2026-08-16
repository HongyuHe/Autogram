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
import pandas as pd

from ..dsl.evaluate import typed_signature_value


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
    category_target: str = "category"
    category_flags: Tuple[str, str, str] = (
        "flag_a",
        "flag_b",
        "flag_c",
    )
    category_values: Tuple[str, str, str] = (
        "class_a",
        "class_b",
        "class_c",
    )
    category_default: str = "baseline"
    band_group_column: str = "segment"
    band_group_focus: str = "stable"
    band_group_other: str = "variable"
    band_state_column: str = "state"
    band_state_focus: str = "baseline"
    band_state_other: str = "event"
    # Generic, invariant-independent centre for the healthy-band proxy. It must NOT coincide with any
    # specific dataset's healthy threshold (e.g. GTIB's 0.998) -- the proxy exercises the band-fitting
    # capability, so its centre is an arbitrary tuned constant, and the planted signature records this
    # same value so the proxy's data and its expected signature always agree.
    band_center: float = 0.5
    band_spread: float = 0.001

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
    row_context: Dict[str, np.ndarray] = field(default_factory=dict)
    relations: Dict[str, object] = field(default_factory=dict)
    related_aggregates: Dict[str, dict] = field(default_factory=dict)


def make_synthetic(n_entities: int = 6, n_snapshots: int = 400, noise: float = 0.0,
                   seed: int = 0, vocab: Vocab = None, unstable_frac: float = 0.0,
                   families=None, regime_factor: float = 1.6,
                   offset_hold_rate: float = 0.67, offset_factor: float = 0.98,
                   presence_rate: float = 0.6,
                   temporal_window: int = 5) -> Synthetic:
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

    Two explicit one-sided proxy families are also available and, when enabled *on their own*,
    replace all relational structure with sign-constrained independent columns (see below):

    * ``families=("nonneg",)`` -> every column ``>= 0``
    * ``families=("nonpos",)`` -> every column ``<= 0``

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
    if "sum_balance" in enabled:
        D = 0.5 * (D + np.swapaxes(D, 1, 2))
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
    elif "conditional_pair" in enabled:
        pair_regime = np.resize(
            np.array(["paired", "other"], dtype=object),
            T,
        )
        independent_links = rng.gamma(
            shape=2.0,
            scale=8.0,
            size=(T, N, N),
        )
        Lfrm = np.where(
            (pair_regime == "paired")[:, None, None],
            L,
            independent_links,
        )
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
        orig = (row_totals if enabled & {"row_sum", "sum_balance"}
                else rng.gamma(shape=2.0, scale=10.0, size=(T, N)))
        term = (col_totals if "col_sum" in enabled
                else rng.gamma(shape=2.0, scale=10.0, size=(T, N)))
    if "ratio" in enabled:
        diagonal = D[:, np.arange(N), np.arange(N)]
        orig = term / np.maximum(diagonal, 1e-6)
    if "proportional" in enabled:
        orig = 1.7 * term
    if "conditional_proportional" in enabled:
        regime = np.resize(
            np.array(["proportional", "other"], dtype=object),
            T,
        )
        independent = rng.gamma(shape=2.0, scale=10.0, size=(T, N))
        orig = np.where(
            (regime == "proportional")[:, None],
            1.7 * term,
            independent,
        )
    if "conditional_pair" in enabled:
        independent = rng.gamma(shape=2.0, scale=10.0, size=(T, N))
        orig = np.where(
            (pair_regime == "paired")[:, None],
            term,
            independent,
        )
    if "monotone" in enabled:
        orig = np.cumsum(rng.uniform(1.0, 3.0, size=(T, N)), axis=0)
    if "windowed_ratio" in enabled:
        diagonal = D[:, np.arange(N), np.arange(N)]
        orig = np.full_like(term, np.nan)
        for entity in range(N):
            numerator = np.convolve(term[:, entity], np.ones(temporal_window), mode="valid")
            denominator = np.convolve(
                diagonal[:, entity],
                np.ones(temporal_window),
                mode="valid",
            )
            orig[temporal_window - 1:, entity] = numerator / denominator
    row_context: Dict[str, np.ndarray] = {}
    relations: Dict[str, object] = {}
    related_aggregates: Dict[str, dict] = {}
    proxy_timestamps = None
    if "conditional_proportional" in enabled:
        row_context["regime"] = regime
    if "conditional_pair" in enabled:
        row_context["regime"] = pair_regime
    if enabled & {"conditional_positive", "conditional_zero"}:
        regime = np.resize(
            np.array(["positive", "zero", "other"], dtype=object),
            T,
        )
        increment = np.select(
            [regime == "positive", regime == "zero"],
            [2.0, 0.0],
            default=-2.0,
        )
        factors = np.arange(1.0, N + 1.0)
        orig = np.cumsum(increment[:, None] * factors[None, :], axis=0)
        row_context["regime"] = regime
    if "sustained" in enabled:
        cycle = np.concatenate([
            np.array([0.8]),
            np.linspace(0.2, 0.45, 2 * temporal_window),
            np.array([0.9]),
        ])
        pattern = np.resize(cycle, T)
        term = np.repeat(pattern[:, None], N, axis=1)
        orig = np.zeros((T, N), dtype=float)
        for entity in range(N):
            below = term[:, entity] < 0.5
            for index in range(temporal_window - 1, T):
                orig[index, entity] = float(
                    np.all(below[index - temporal_window + 1:index + 1])
                )
    if "conjunction" in enabled:
        term = np.repeat(np.linspace(0.9, 0.1, T)[:, None], N, axis=1)
        for entity in range(N):
            D[:, entity, entity] = 0.05
        orig = np.zeros((T, N), dtype=float)
        for entity in range(N):
            deficit = pd.Series(
                term[:, entity] - D[:, entity, entity]
            ).rolling(temporal_window, min_periods=temporal_window).sum().to_numpy()
            slope = pd.Series(term[:, entity]).diff(temporal_window).to_numpy()
            orig[:, entity] = (
                (term[:, entity] < 0.5)
                & (deficit > 0.0)
                & (slope <= 0.0)
            ).astype(float)
    if "categorical" in enabled:
        first = np.resize(
            np.array([False, True, True, False, False, False]),
            T,
        )
        second = np.resize(
            np.array([False, True, False, True, False, False]),
            T,
        )
        third = np.resize(
            np.array([False, False, True, True, False, True]),
            T,
        )
        category = np.full(
            T,
            vocab.category_default,
            dtype=object,
        )
        category[third] = vocab.category_values[2]
        category[second] = vocab.category_values[1]
        category[first] = vocab.category_values[0]
        row_context.update({
            vocab.category_flags[0]: first,
            vocab.category_flags[1]: second,
            vocab.category_flags[2]: third,
            vocab.category_target: category,
        })
    if "cross_grain" in enabled:
        proxy_timestamps = pd.date_range("2026-01-01", periods=T, freq="1min").to_numpy()
        child_rows = []
        for shard, increment in (("s0", 10.0), ("s1", 20.0)):
            cumulative = 0.0
            for step in range(T * 6):
                cumulative += increment
                child_rows.append({
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=10 * step),
                    "shard_id": shard,
                    "counter": cumulative,
                    "reset_flag": False,
                })
        relations["raw"] = pd.DataFrame(child_rows)
        related_aggregates["proxy_raw_sum"] = {
            "relation": "raw",
            "column": "counter",
            "mode": "sum_delta",
            "parent_keys": (),
            "child_keys": (),
            "partition_keys": ("shard_id",),
            "parent_time": "__time__",
            "child_time": "timestamp",
            "window_seconds": 60,
            "reset_column": "reset_flag",
            "validity_columns": ("counter",),
        }
        row_context["__time__"] = proxy_timestamps
        orig = np.full((T, N), 180.0)
        orig[0, :] = np.nan
    if "healthy_band" in enabled:
        group = np.resize(np.array([
            vocab.band_group_focus,
            vocab.band_group_focus,
            vocab.band_group_other,
        ]), T)
        state = np.resize(np.array([
            vocab.band_state_focus,
            vocab.band_state_focus,
            vocab.band_state_focus,
            vocab.band_state_other,
        ]), T)
        healthy = rng.normal(vocab.band_center, vocab.band_spread, size=(T, N))
        broad = rng.normal(0.9, 0.08, size=(T, N))
        mask = (
            (group == vocab.band_group_focus)
            & (state == vocab.band_state_focus)
        )
        orig = np.where(mask[:, None], healthy, broad)
        row_context.update({
            vocab.band_group_column: group,
            vocab.band_state_column: state,
        })
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
            if (
                name.startswith(vocab.measurement + "_")
                and not (
                    enabled & {
                        "conditional_positive",
                        "conditional_zero",
                        "sustained",
                        "conjunction",
                        "cross_grain",
                        "ratio",
                        "windowed_ratio",
                        "monotone",
                        "lag_bound",
                        "conditional_proportional",
                    }
                    and name.endswith("_" + vocab.source)
                )
                and not (
                    enabled & {"ratio", "windowed_ratio"}
                    and name.endswith("_" + vocab.destination)
                )
                and not (
                    "conditional_pair" in enabled
                    and name.endswith((
                        "_" + vocab.source,
                        "_" + vocab.destination,
                    ))
                )
            ):
                matrix[:, k] = matrix[:, k] * (1.0 + noise * rng.standard_normal(T))
        matrix = np.maximum(matrix, 0.0)

    # Boolean-definition targets are derived columns. Recompute them from the final observed
    # operands so measurement noise cannot silently invalidate the exact planted definition.
    if enabled & {"sustained", "conjunction"}:
        index_by_name = {
            name: index
            for index, name in enumerate(cols)
        }
        for entity in ents:
            source_index = index_by_name[
                f"{vocab.measurement}_{entity}_{vocab.source}"
            ]
            destination = matrix[
                :,
                index_by_name[
                    f"{vocab.measurement}_{entity}_{vocab.destination}"
                ],
            ]
            if "conjunction" in enabled:
                demand_self = matrix[
                    :,
                    index_by_name[f"{vocab.demand}_{entity}_{entity}"],
                ]
                deficit = pd.Series(
                    destination - demand_self
                ).rolling(
                    temporal_window,
                    min_periods=temporal_window,
                ).sum().to_numpy()
                slope = pd.Series(destination).diff(
                    temporal_window
                ).to_numpy()
                target = (
                    (destination < 0.5)
                    & (deficit > 0.0)
                    & (slope <= 0.0)
                )
            else:
                target = np.zeros(T, dtype=bool)
                below = destination < 0.5
                for snapshot in range(temporal_window - 1, T):
                    target[snapshot] = bool(np.all(
                        below[
                            snapshot - temporal_window + 1:
                            snapshot + 1
                        ]
                    ))
            matrix[:, source_index] = target.astype(float)

    # explicit one-sided proxies (item: nonneg/nonpos).  When the run enables *only* a one-sided
    # family, sign-constrain every column and apply an independent per-cell dropout to exact zeros.
    # Zero is both >= 0 and <= 0, so the sign law is preserved, while the independent dropout masks
    # give each column its own presence pattern -- so no pair, sum, zero, balance, or presence
    # relation is accidentally planted (only the one-sided sign relation holds).
    one_sided = enabled & {"nonneg", "nonpos"}
    if one_sided and not (enabled - {"nonneg", "nonpos"}):
        keep = rng.random(matrix.shape) >= 0.25
        matrix = np.abs(matrix) * keep
        if "nonpos" in enabled:
            matrix = -matrix

    all_planted = _planted(vocab, ents, N, temporal_window=temporal_window)
    planted = {k: v for k, v in all_planted.items() if k in enabled}
    if "nonneg" in enabled:
        planted["nonneg"] = frozenset(cols)
    if "nonpos" in enabled:
        planted["nonpos"] = frozenset(cols)
    ts = proxy_timestamps if proxy_timestamps is not None else np.arange(T)
    return Synthetic(columns=cols, matrix=matrix, timestamps=ts, vocab=vocab,
                     entities=ents, planted=planted, row_context=row_context,
                     relations=relations, related_aggregates=related_aggregates)


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
        blocks.append(rng.normal(0.0, 10.0, size=(T, 1)))

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


def enrich_null(
    data: Synthetic,
    *,
    seed: int = 0,
    conditions: bool = False,
    related: bool = False,
) -> Synthetic:
    """Attach independent context needed to exercise conditional and related null hypotheses."""

    rng = np.random.default_rng(seed + 31337)
    n_snapshots = int(data.matrix.shape[0])
    if conditions:
        data.row_context.update({
            "regime": rng.choice(
                np.array(["positive", "zero", "other"], dtype=object),
                size=n_snapshots,
            ),
            data.vocab.band_group_column: rng.choice(
                np.array([
                    data.vocab.band_group_focus,
                    data.vocab.band_group_other,
                ], dtype=object),
                size=n_snapshots,
            ),
            data.vocab.band_state_column: rng.choice(
                np.array([
                    data.vocab.band_state_focus,
                    data.vocab.band_state_other,
                ], dtype=object),
                size=n_snapshots,
            ),
        })
    if related:
        timestamps = pd.date_range(
            "2026-01-01",
            periods=n_snapshots,
            freq="1min",
        ).to_numpy()
        child_rows = []
        for shard in ("s0", "s1"):
            cumulative = 0.0
            for step in range(n_snapshots * 6):
                cumulative += float(rng.gamma(shape=2.0, scale=8.0))
                child_rows.append({
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(seconds=10 * step),
                    "shard_id": shard,
                    "counter": cumulative,
                    "reset_flag": False,
                })
        data.timestamps = timestamps
        data.row_context["__time__"] = timestamps
        data.relations["raw"] = pd.DataFrame(child_rows)
        data.related_aggregates["proxy_raw_sum"] = {
            "relation": "raw",
            "column": "counter",
            "mode": "sum_delta",
            "parent_keys": (),
            "child_keys": (),
            "partition_keys": ("shard_id",),
            "parent_time": "__time__",
            "child_time": "timestamp",
            "window_seconds": 60,
            "reset_column": "reset_flag",
            "validity_columns": ("counter",),
        }
    return data


def make_temporal_null(
    n_entities: int = 4,
    n_snapshots: int = 160,
    seed: int = 0,
) -> Synthetic:
    """A monotone-by-row proxy whose time order is shuffled, destroying monotonicity."""

    data = make_synthetic(
        n_entities=n_entities,
        n_snapshots=n_snapshots,
        noise=0.0,
        seed=seed,
        families=("monotone",),
    )
    rng = np.random.default_rng(seed + 991)
    data.timestamps = rng.permutation(np.asarray(data.timestamps))
    data.planted = {}
    return data


def make_definition_null(
    n_entities: int = 3,
    n_snapshots: int = 240,
    seed: int = 0,
) -> Synthetic:
    """Independent Boolean/categorical targets and signals for definition false-discovery checks."""

    data = make_null(
        n_entities=n_entities,
        n_snapshots=n_snapshots,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1771)
    for index, column in enumerate(data.columns):
        if column.endswith("_source"):
            data.matrix[:, index] = (rng.random(n_snapshots) < 0.5).astype(float)
    data.row_context = {
        data.vocab.category_flags[0]: rng.random(n_snapshots) < 0.5,
        data.vocab.category_flags[1]: rng.random(n_snapshots) < 0.5,
        data.vocab.category_flags[2]: rng.random(n_snapshots) < 0.5,
        data.vocab.category_target: rng.choice(
            np.array([
                data.vocab.category_default,
                *data.vocab.category_values,
            ], dtype=object),
            size=n_snapshots,
            replace=True,
        ),
    }
    return data


def _planted(vocab: Vocab, ents, N, temporal_window: int = 5) -> Dict[str, object]:
    two_end = set()
    agg_ref_balance = []
    self_zero = []
    row_sum = []
    col_sum = []
    ratios = set()
    proportionals = set()
    monotone = set()
    lag_bound = set()
    sum_balance = set()
    windowed_ratios = set()
    conditional_positive = set()
    conditional_zero = set()
    conditional_proportional = set()
    conditional_pair = set()
    sustained = set()
    conjunction = set()
    cross_grain = set()
    healthy_band = set()
    for i in range(N):
        self_zero.append(f"{vocab.demand}_{ents[i]}_{ents[i]}")
        row = frozenset(f"{vocab.demand}_{ents[i]}_{ents[j]}" for j in range(N) if j != i)
        col = frozenset(f"{vocab.demand}_{ents[j]}_{ents[i]}" for j in range(N) if j != i)
        row_sum.append((f"{vocab.measurement}_{ents[i]}_{vocab.source}", row))
        col_sum.append((f"{vocab.measurement}_{ents[i]}_{vocab.destination}", col))
        ratios.add((
            f"{vocab.measurement}_{ents[i]}_{vocab.source}",
            f"{vocab.measurement}_{ents[i]}_{vocab.destination}",
            f"{vocab.demand}_{ents[i]}_{ents[i]}",
        ))
        proportionals.add((
            f"{vocab.measurement}_{ents[i]}_{vocab.source}",
            f"{vocab.measurement}_{ents[i]}_{vocab.destination}",
        ))
        monotone.add((
            f"{vocab.measurement}_{ents[i]}_{vocab.source}",
            1,
            ">=",
        ))
        lag_bound.add((
            f"{vocab.measurement}_{ents[i]}_{vocab.source}",
            1,
            ">=",
        ))
        sum_balance.add(frozenset({row, col}))
        windowed_ratios.add((
            f"{vocab.measurement}_{ents[i]}_{vocab.source}",
            f"{vocab.measurement}_{ents[i]}_{vocab.destination}",
            f"{vocab.demand}_{ents[i]}_{ents[i]}",
            temporal_window,
        ))
        conditional_positive.add((
            (
                "regime",
                "==",
                (typed_signature_value("positive"),),
            ),
            (
                "delta_bound",
                (f"{vocab.measurement}_{ents[i]}_{vocab.source}", 1, ">="),
            ),
        ))
        conditional_zero.add((
            (
                "regime",
                "==",
                (typed_signature_value("zero"),),
            ),
            (
                "delta_zero",
                (f"{vocab.measurement}_{ents[i]}_{vocab.source}", 1),
            ),
        ))
        conditional_proportional.add((
            (
                "regime",
                "==",
                (typed_signature_value("proportional"),),
            ),
            (
                "proportional",
                (
                    f"{vocab.measurement}_{ents[i]}_{vocab.source}",
                    f"{vocab.measurement}_{ents[i]}_{vocab.destination}",
                ),
            ),
        ))
        conditional_pair.add((
            (
                "regime",
                "==",
                (typed_signature_value("paired"),),
            ),
            (
                "pair",
                frozenset({
                    f"{vocab.measurement}_{ents[i]}_{vocab.source}",
                    f"{vocab.measurement}_{ents[i]}_{vocab.destination}",
                }),
            ),
        ))
        source = f"{vocab.measurement}_{ents[i]}_{vocab.source}"
        destination = f"{vocab.measurement}_{ents[i]}_{vocab.destination}"
        demand_self = f"{vocab.demand}_{ents[i]}_{ents[i]}"
        sustained.add((
            source,
            (
                "sustained",
                temporal_window,
                ("bound", ("ref", destination), "<", "threshold"),
            ),
        ))
        conjunction.add((
            source,
            (
                ("bound", ("ref", destination), "<", "threshold"),
                (
                    "bound",
                    (
                        "rolling",
                        "SUM",
                        temporal_window,
                        (
                            "difference",
                            ("ref", destination),
                            ("ref", demand_self),
                        ),
                    ),
                    ">",
                    "threshold",
                ),
                (
                    "bound",
                    ("delta", ("ref", destination), temporal_window),
                    "<=",
                    "threshold",
                ),
            ),
        ))
        cross_grain.add((
            f"{vocab.measurement}_{ents[i]}_{vocab.source}",
            "proxy_raw_sum",
        ))
        healthy_band.add((
            (
                "all",
                tuple(sorted((
                    (
                        vocab.band_group_column,
                        "==",
                        (typed_signature_value(vocab.band_group_focus),),
                    ),
                    (
                        vocab.band_state_column,
                        "==",
                        (typed_signature_value(vocab.band_state_focus),),
                    ),
                ), key=str)),
            ),
            (
                "healthy_band",
                (f"{vocab.measurement}_{ents[i]}_{vocab.source}", vocab.band_center),
            ),
        ))
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
        "ratio": ratios,
        "proportional": proportionals,
        "monotone": monotone,
        "lag_bound": lag_bound,
        "sum_balance": sum_balance,
        "windowed_ratio": windowed_ratios,
        "conditional_positive": conditional_positive,
        "conditional_zero": conditional_zero,
        "conditional_proportional": conditional_proportional,
        "conditional_pair": conditional_pair,
        "sustained": sustained,
        "conjunction": conjunction,
        "categorical": {
            (
                vocab.category_target,
                tuple(zip(
                    vocab.category_flags,
                    (
                        typed_signature_value(value)
                        for value in vocab.category_values
                    ),
                )),
                typed_signature_value(vocab.category_default),
            ),
        },
        "cross_grain": cross_grain,
        "healthy_band": healthy_band,
    }
