"""Central configuration: every tuning knob the engine exposes (design Sec. 10.4).

Knobs are grouped by the loop level that owns them so the three-level structure of
Sec. 10.4 is explicit in code:

* :class:`EvalConfig`   -- inner loop: analytic band fit + scoring (no search here).
* :class:`SearchConfig` -- middle loop: evolutionary rule-set search.
* :class:`GrammarConfig`-- outer loop: grammar/LLM extension cadence.

Everything is seedable for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Inner-loop evaluator knobs.  The band is *fit*, never searched (Sec. 5.4)."""

    target_coverage: float = 0.90      # kappa*: global coverage the band is fit to hit
    holdout_frac: float = 0.5          # split-conformal calibration/eval split
    gate_k: float = 6.0                # |bias| or scale must exceed gate_k * sigma_prop
    eps_exact: float = 0.01            # band <= eps_exact => report as an exact invariant
    eps_max: float = 0.15              # band > eps_max with no structure => reject
    lift_min: float = 2.0              # min name-semantic lift to accept a tight fit
    n_perm: int = 8                    # permutations for the name-blind null (lift)
    # MDL description-length weights (bits-ish); only relative magnitudes matter.
    mdl_w_complexity: float = 1.0
    mdl_w_band: float = 1.0
    mdl_w_residual: float = 1.0
    # combined scalar score weights (used only to rank *within* a QD cell).
    w_confidence: float = 1.0
    w_tightness: float = 1.0
    w_support: float = 0.5
    w_mdl: float = 0.25
    subsample: int = 0                 # 0 => use all points; else cap points per rule
    seed: int = 0


@dataclass
class SearchConfig:
    """Middle-loop evolutionary search knobs."""

    iterations: int = 200
    population: int = 24
    islands: int = 4
    migration_every: int = 25
    p_mutate: float = 0.9
    elites_per_cell: int = 1
    thompson: bool = True              # Thompson budget allocation over islands
    assemble_k_max: int = 16           # cap on the assembled portfolio size
    dedup_rel: float = 0.05            # near-duplicate residual-corr threshold
    seed_from_grammar: bool = False    # also inject enumerate_candidates(G) as seeds?
    bootstrap_random: int = 12         # blind random rules added to the seed pool
    seed: int = 0


@dataclass
class GrammarConfig:
    """Outer-loop grammar/LLM-extension knobs."""

    proposer: str = "scripted"         # scripted | openai | subagent
    rounds: int = 1                    # outer grammar-extension rounds
    proposals_per_round: int = 12
    max_complexity: int = 12           # admits the complexity-10 Kirchhoff flow law (I3)
    seed: int = 0


@dataclass
class RunConfig:
    """Top-level run configuration."""

    dataset: str = "abilene"
    data_path: str = ""
    eval: EvalConfig = field(default_factory=EvalConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)
    seed: int = 0

    def reseed(self) -> None:
        """Propagate the top-level seed to every sub-config."""
        self.eval.seed = self.seed
        self.search.seed = self.seed
        self.grammar.seed = self.seed
