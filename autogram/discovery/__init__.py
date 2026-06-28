"""Open-ended invariant discovery package.

This is the redesigned engine: schema induction from column names, an LLM/random proposer
inside the induced schema, a data-only evaluator, a simple Pareto archive, and a discovery loop
whose only judge is the data.  Nothing here reads a clean oracle, a known-invariant catalogue,
or a pre-tuned constant.
"""

from .induce import HeuristicInducer, SchemaInducer, induce_adapter, induce_spec
from .evaluate import DataOnlyEvaluator, Evaluation
from .archive import ParetoArchive
from .loop import DiscoveryResult, discover
from . import synth, validate

__all__ = [
    "SchemaInducer",
    "HeuristicInducer",
    "induce_spec",
    "induce_adapter",
    "DataOnlyEvaluator",
    "Evaluation",
    "ParetoArchive",
    "discover",
    "DiscoveryResult",
    "synth",
    "validate",
]
