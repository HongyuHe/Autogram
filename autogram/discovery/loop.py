"""The discovery loop: induce -> propose -> ground -> evaluate -> archive -> progress (smallest loop).

One stage of schema induction up front, then a single loop whose only judge is the data.  The
loop mines its own elites for the next round's proposals and measures *progress* as the
improvement of the archive's coverage x parsimony front -- a success signal that needs no
answer key.  A stalled archive triggers schema re-induction (and a mild grammar widening) so
the loop can break out of a plateau, exactly as the design specifies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..config import DiscoveryConfig, SearchConfig
from ..dsl import ast as A
from ..dsl.grammar import Grammar, grammar_from_adapter
from ..loader.loader import Dataset, build_dataset
from .archive import ParetoArchive
from .evaluate import DataOnlyEvaluator, Evaluation
from .induce import HeuristicInducer, SchemaInducer, induce_spec
from .propose import LLMProposer, PortfolioProposer, RandomProposer
from ..schema.compiler import compile_spec


@dataclass
class DiscoveryResult:
    portfolio: List[Evaluation]
    archive: ParetoArchive
    dataset: Dataset
    grammar: Grammar
    rounds_run: int
    progress_history: List[float] = field(default_factory=list)
    reinductions: int = 0

    def report(self) -> str:
        lines = [f"Discovered invariants on {self.dataset.name!r} "
                 f"({len(self.portfolio)} accepted, {self.rounds_run} rounds, "
                 f"{self.reinductions} re-inductions):", "-" * 78]
        for ev in self.portfolio:
            lines.append("  " + ev.summary())
        if not self.portfolio:
            lines.append("  (none)")
        lines.append("-" * 78)
        lines.append(f"  progress trace: "
                     + ", ".join(f"{p:.2f}" for p in self.progress_history))
        return "\n".join(lines)


def _dataset_from_columns(columns: Sequence[str], matrix: np.ndarray, adapter,
                          name: str, timestamps=None) -> Dataset:
    return build_dataset(columns, matrix, adapter, name, timestamps)


def _make_proposer(G: Grammar, scfg: SearchConfig, llm_responder=None):
    if scfg.proposer == "random":
        return RandomProposer(G, scfg.p_mutate)
    if scfg.proposer in ("portfolio", "llm"):
        return PortfolioProposer([
            LLMProposer(G, responder=llm_responder),
            RandomProposer(G, scfg.p_mutate),
        ])
    raise ValueError(f"unknown proposer mode {scfg.proposer!r}")


def discover(columns: Sequence[str], matrix: np.ndarray, *,
             inducer: Optional[SchemaInducer] = None,
             proposer=None,
             llm_responder=None,
             discovery_cfg: Optional[DiscoveryConfig] = None,
             search_cfg: Optional[SearchConfig] = None,
             name: str = "synthetic", timestamps=None,
             sample_rows=None) -> DiscoveryResult:
    """Run open-ended discovery on a named, structured tabular dataset.

    ``columns``/``matrix`` are the *only* inputs; the schema, the proposals and the verdicts are
    all derived from them.  Deterministic given the seeds in the configs.
    """
    inducer = inducer or HeuristicInducer()
    dcfg = discovery_cfg or DiscoveryConfig()
    scfg = search_cfg or SearchConfig()
    rng = random.Random(scfg.seed)

    spec = induce_spec(columns, inducer, sample_rows)
    adapter = compile_spec(spec)
    ds = _dataset_from_columns(columns, matrix, adapter, name, timestamps)
    G = grammar_from_adapter(adapter, scfg.max_complexity, scfg.max_add_arity)
    evaluator = DataOnlyEvaluator(ds, dcfg)
    proposer_obj = proposer or _make_proposer(G, scfg, llm_responder)
    archive = ParetoArchive()

    progress_history: List[float] = []
    stall = 0
    reinductions = 0
    rounds_run = 0

    for r in range(scfg.rounds):
        rounds_run = r + 1
        seeds = [e.rule for e in archive.elites()]
        candidates = proposer_obj.propose(scfg.proposals_per_round, seeds, rng)
        for rule in candidates:
            ev = evaluator.evaluate(rule)
            archive.add(ev)
        prog = archive.progress()
        progress_history.append(prog)
        improved = prog > (progress_history[-2] + 1e-9) if len(progress_history) > 1 else prog > 0
        if improved:
            stall = 0
        else:
            stall += 1
        if stall >= scfg.stall_patience and r < scfg.rounds - 1:
            G, evaluator, ds, adapter, reinductions = _reinduce(
                columns, matrix, inducer, dcfg, scfg, name, timestamps,
                sample_rows, reinductions)
            if proposer is None:
                proposer_obj = _make_proposer(G, scfg, llm_responder)
            stall = 0

    return DiscoveryResult(
        portfolio=archive.portfolio(non_redundant=True), archive=archive, dataset=ds,
        grammar=G, rounds_run=rounds_run, progress_history=progress_history,
        reinductions=reinductions)


def _reinduce(columns, matrix, inducer, dcfg, scfg, name, timestamps, sample_rows,
              reinductions):
    """Re-induce the schema and mildly widen the grammar to escape a stalled archive."""
    spec = induce_spec(columns, inducer, sample_rows)
    adapter = compile_spec(spec)
    ds = build_dataset(columns, matrix, adapter, name, timestamps)
    widened = scfg.max_complexity + 2
    G = grammar_from_adapter(adapter, widened, scfg.max_add_arity)
    evaluator = DataOnlyEvaluator(ds, dcfg)
    return G, evaluator, ds, adapter, reinductions + 1


def discover_synthetic(synth, *, discovery_cfg=None, search_cfg=None,
                       inducer=None, proposer=None, llm_responder=None) -> DiscoveryResult:
    """Convenience: run discovery on a :class:`autogram.discovery.synth.Synthetic` dataset."""
    return discover(synth.columns, synth.matrix, inducer=inducer, proposer=proposer,
                    llm_responder=llm_responder,
                    discovery_cfg=discovery_cfg, search_cfg=search_cfg,
                    name="synthetic", timestamps=synth.timestamps)
