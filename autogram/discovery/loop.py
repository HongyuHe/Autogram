"""Discovery loop: LLM schema induction -> exhaustive enumeration -> solver/data evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..config import DiscoveryConfig, SearchConfig
from ..dsl.grammar import Grammar, grammar_from_adapter
from ..loader.loader import Dataset, build_dataset, load_dataframe
from ..schema.compiler import compile_spec
from .archive import ParetoArchive
from .evaluate import DataOnlyEvaluator, Evaluation
from .induce import SchemaInducer, induce_spec, make_inducer
from .propose import EnumerationProposer


@dataclass
class DiscoveryResult:
    portfolio: List[Evaluation]
    archive: ParetoArchive
    dataset: Dataset
    grammar: Grammar
    rounds_run: int
    progress_history: List[float] = field(default_factory=list)
    reinductions: int = 0
    diagnostics: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Discovered invariants on {self.dataset.name!r} "
                 f"({len(self.portfolio)} accepted, exhaustive enumeration):", "-" * 78]
        for ev in self.portfolio:
            lines.append("  " + ev.summary())
        if not self.portfolio:
            lines.append("  (none)")
        if self.diagnostics:
            lines.append("-" * 78)
            lines.append("  diagnostics:")
            for msg in self.diagnostics:
                lines.append("  - " + msg)
        lines.append("-" * 78)
        lines.append("  statistic: hold-rate with Wilson confidence interval")
        return "\n".join(lines)


def _dataset_from_columns(columns: Sequence[str], matrix: np.ndarray, adapter, name: str, timestamps=None) -> Dataset:
    return build_dataset(columns, matrix, adapter, name, timestamps)


def _make_proposer(G: Grammar, scfg: SearchConfig):
    if scfg.proposer != "enumeration":
        raise ValueError("v2 supports only proposer='enumeration'")
    return EnumerationProposer(G)


def _run_dataset(ds: Dataset, G: Grammar, *, proposer, dcfg: DiscoveryConfig, scfg: SearchConfig) -> DiscoveryResult:
    evaluator = DataOnlyEvaluator(ds, dcfg)
    proposer_obj = proposer or _make_proposer(G, scfg)
    archive = ParetoArchive()
    candidates = proposer_obj.propose(0, (), None)
    if scfg.max_rules and scfg.max_rules > 0:
        candidates = candidates[:scfg.max_rules]
    diagnostics: List[str] = []
    seen_diagnostics = set()
    for rule in candidates:
        ev = evaluator.evaluate(rule)
        if "grounded 0 points" in ev.reason:
            key = (ev.rule.binder, ev.reason)
            if key not in seen_diagnostics and len(diagnostics) < 20:
                diagnostics.append(f"{ev.rule.unparse()}: {ev.reason}")
                seen_diagnostics.add(key)
        archive.add(ev)
    portfolio = archive.portfolio(non_redundant=True)
    return DiscoveryResult(
        portfolio=portfolio, archive=archive, dataset=ds, grammar=G,
        rounds_run=1, progress_history=[archive.progress()], reinductions=0,
        diagnostics=diagnostics)


def discover(columns: Sequence[str], matrix: np.ndarray, *,
             inducer: Optional[SchemaInducer] = None,
             proposer=None,
             llm_responder=None,
             discovery_cfg: Optional[DiscoveryConfig] = None,
             search_cfg: Optional[SearchConfig] = None,
             name: str = "synthetic", timestamps=None,
             sample_rows=None) -> DiscoveryResult:
    """Run guarantees-first discovery from an observed numeric matrix."""
    dcfg = discovery_cfg or DiscoveryConfig()
    scfg = search_cfg or SearchConfig()
    if inducer is None:
        inducer = make_inducer("subagent", responder=llm_responder) if llm_responder is not None else make_inducer("subagent")
    spec = induce_spec(columns, inducer, sample_rows)
    adapter = compile_spec(spec)
    ds = _dataset_from_columns(columns, matrix, adapter, name, timestamps)
    G = grammar_from_adapter(adapter, scfg.max_complexity, scfg.max_add_arity)
    return _run_dataset(ds, G, proposer=proposer, dcfg=dcfg, scfg=scfg)


def discover_dataframe(df, *, inducer: Optional[SchemaInducer] = None,
                       proposer=None, discovery_cfg: Optional[DiscoveryConfig] = None,
                       search_cfg: Optional[SearchConfig] = None,
                       name: str = "dataframe") -> DiscoveryResult:
    """Run discovery on a pandas DataFrame, decoding only observed ``ground_truth`` values."""
    dcfg = discovery_cfg or DiscoveryConfig()
    scfg = search_cfg or SearchConfig()
    inducer = inducer or make_inducer("subagent")
    spec = induce_spec(list(df.columns), inducer, sample_rows=None)
    adapter = compile_spec(spec)
    timestamps = df["timestamp"].values if "timestamp" in df.columns else None
    ds = load_dataframe(df, adapter, name, timestamps=timestamps)
    G = grammar_from_adapter(adapter, scfg.max_complexity, scfg.max_add_arity)
    return _run_dataset(ds, G, proposer=proposer, dcfg=dcfg, scfg=scfg)


def discover_synthetic(synth, *, discovery_cfg=None, search_cfg=None,
                       inducer=None, proposer=None, llm_responder=None) -> DiscoveryResult:
    return discover(synth.columns, synth.matrix, inducer=inducer, proposer=proposer,
                    llm_responder=llm_responder, discovery_cfg=discovery_cfg,
                    search_cfg=search_cfg, name="synthetic", timestamps=synth.timestamps)
