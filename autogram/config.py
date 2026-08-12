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

    tolerance: float = 0.05             # dimensionless epsilon (fallback / global-band relative tolerance)
    band_mode: str = "adaptive"         # DEFAULT "adaptive" (per-candidate knee band, capped at `tolerance` so it can only tighten). "global" = one fixed shared tolerance. Adaptive fits each rule's band to its own residual spread, so the monotone-completeness transfer becomes a bound on hold-rate alone (cleanliness is absorbed by the band, not required as a precondition).
    band_holdout_frac: float = 0.3      # split-conformal holdout for the adaptive band
    separation_tolerance: float = 1e-6  # minimum relative gap for != separations
    presence_tolerance: float = 1e-9    # relative non-zero cutoff for <|> pairings
    ordering_tolerance: float = 1e-12   # strict < and > margin on the relative residual
    hold_rate_threshold: float = 0.62   # Wilson lower bound required for approximate-law acceptance
    ci_alpha: float = 0.05              # Wilson interval confidence level
    subsample: int = 0                  # 0 => use every grounded point
    seed: int = 0
    min_condition_points: int = 20
    min_condition_fraction: float = 0.01
    parameter_holdout_frac: float = 0.3
    min_proportional_points: int = 8
    definition_min_lift: float = 0.05
    # Learned-threshold sweep ceiling. A Boolean/definition bound with a learned threshold is fit
    # over the fit-split midpoints between consecutive distinct effective-term values (plus below-
    # min / above-max edge sentinels), which is exhaustive for a single stump and jointly exact
    # across a conjunction's learned bounds. 0 keeps that exact default; a positive value is a
    # fail-loud safety ceiling that raises SearchSpaceTruncatedError when a bound exposes more
    # fit-split candidate thresholds than the cap, so the sweep can never be silently coarsened.
    max_threshold_candidates: int = 0

    # Acceptance-threshold policy for the hold-rate bar theta. DEFAULT "global" applies one flat
    # hold_rate_threshold to every rule, so theta is shared across candidates; this makes the
    # monotone-completeness transfer unconditional on complexity (the (C3) side-condition becomes
    # trivial). "per_rule" instead raises the bar for structurally fragile candidates (a per-law
    # precision gate) at the cost of reintroducing that complexity side-condition in the guarantee.
    threshold_policy: str = "global"
    thr_base_complexity: int = 6        # no complexity penalty at/below this AST size
    thr_complexity_penalty: float = 0.01   # added per unit of complexity above the baseline
    thr_separation_penalty: float = 0.05   # scaled by fitted eps / tolerance cap (adaptive band only)
    thr_min_bindings: int = 8           # below this many bindings, apply a fragility penalty
    thr_low_support_penalty: float = 0.03
    thr_ceiling: float = 0.95           # per-rule bar never exceeds this


@dataclass
class SearchConfig:
    """Bounded enumeration controls."""

    proposer: str = "enumeration"
    max_complexity: int = DEFAULT_MAX_COMPLEXITY
    max_add_arity: int = DEFAULT_MAX_ADD_ARITY
    max_rules: int = 0                  # 0 => no ceiling; positive values fail if the bounded grammar is larger
    max_nonlinear_leaves: int = 0       # positive => fail if the declared nonlinear leaves exceed it
    max_linear_leaves: int = 0          # positive => fail if the declared scaled/additive leaves exceed it
    max_conditioned_rules: int = 0      # 0 => condition every eligible temporal rule
    seed: int = 0
    max_lag: int = 0                    # 0 => preserve the induced/profiled grammar bound
    windows: tuple[int, ...] = ()       # empty => preserve the induced/profiled windows


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
