"""Configuration for the open-ended discovery engine.

Every quantity that used to bake in a dataset assumption is gone.  There is no declared noise
scale ``eta``, no target coverage ``kappa*``, no ``gate_k`` / ``eps_exact`` / ``eps_max`` /
``lift_min``, and no fixed stability/support cut-off.  What remains are statistical protocol
choices (permutation count, significance level, split counts) and search-budget knobs -- never
thresholds tuned to a particular dataset's structure.

The band's operating coverage and tolerance are read off each rule's own residual distribution
(:func:`autogram.evaluator.band.fit_band_auto`); acceptance is a self-calibrated comparison
against the name-permutation null plus cross-split / temporal stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiscoveryConfig:
    """Data-only evaluator + acceptance knobs (statistical levels, not tuned constants)."""

    holdout_frac: float = 0.5       # split-conformal calibration/eval split (generic)
    n_perm: int = 24                # permutations for the name-permutation null
    n_splits: int = 4               # disjoint cross-splits for stability
    n_time_blocks: int = 3          # contiguous time blocks for temporal stability
    alpha: float = 0.05             # significance level for the lift percentile / FDR control
    require_lift: bool = True       # ablation seam: disable only to measure lift/null necessity
    require_null_support: bool = True  # ablation seam: disable with lift to count pre-null admits
    require_stability: bool = True  # ablation seam: disable only to measure stability necessity
    require_parsimony: bool = True  # ablation seam: disable only for pre-MDL lift ablations
    subsample: int = 0              # 0 => use all points; else cap residual points per rule
    seed: int = 0


@dataclass
class SearchConfig:
    """Discovery loop (proposer + Pareto archive) budget knobs."""

    rounds: int = 6                 # outer rounds (own-elite progress checked each round)
    proposals_per_round: int = 120  # candidate rules proposed per round
    p_mutate: float = 0.7           # fraction of proposals from mutation vs fresh random
    proposer: str = "random"        # random (offline) or portfolio (LLM + random)
    max_complexity: int = 12
    max_add_arity: int = 3
    stall_patience: int = 2         # rounds without Pareto progress before re-inducing schema
    seed: int = 0


@dataclass
class RunConfig:
    """Top-level run configuration."""

    name: str = "synthetic"
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    seed: int = 0

    def reseed(self) -> None:
        self.discovery.seed = self.seed
        self.search.seed = self.seed
