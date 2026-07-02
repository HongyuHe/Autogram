"""Configuration for the guarantees-first discovery engine."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_MAX_COMPLEXITY = 10
DEFAULT_MAX_ADD_ARITY = 2


@dataclass
class DiscoveryConfig:
    """Logic + hold-rate evaluation knobs.

    The evaluator has one statistic: empirical hold-rate with a Wilson confidence interval.
    Z3 handles logical truth, equivalence and subsumption; MDL is only a final tie-breaker.
    """

    tolerance: float = 0.05             # dimensionless epsilon for ~=, ==, <=, >=
    separation_tolerance: float = 1e-6  # minimum relative gap for != separations
    presence_tolerance: float = 1e-9    # relative non-zero cutoff for <|> pairings
    hold_rate_threshold: float = 0.62   # Wilson lower bound required for approximate-law acceptance
    ci_alpha: float = 0.05              # Wilson interval confidence level
    subsample: int = 0                  # 0 => use every grounded point
    seed: int = 0


@dataclass
class SearchConfig:
    """Bounded enumeration controls."""

    proposer: str = "enumeration"
    max_complexity: int = DEFAULT_MAX_COMPLEXITY
    max_add_arity: int = DEFAULT_MAX_ADD_ARITY
    max_rules: int = 0                  # 0 => exhaust the bounded grammar
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
