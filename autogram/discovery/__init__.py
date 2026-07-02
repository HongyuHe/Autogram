"""Guarantees-first invariant discovery package."""

from .induce import (
    OpenAISchemaInducer,
    SchemaInducer,
    SubagentSchemaInducer,
    available_inducer_backends,
    induce_adapter,
    induce_spec,
    make_inducer,
)
from .evaluate import DataOnlyEvaluator, Evaluation
from .archive import ParetoArchive
from .loop import DiscoveryResult, discover, discover_dataframe
from .subagent import AutogramSubagentRunner
from . import synth, validate

__all__ = [
    "SchemaInducer",
    "SubagentSchemaInducer",
    "OpenAISchemaInducer",
    "AutogramSubagentRunner",
    "available_inducer_backends",
    "make_inducer",
    "induce_spec",
    "induce_adapter",
    "DataOnlyEvaluator",
    "Evaluation",
    "ParetoArchive",
    "discover",
    "discover_dataframe",
    "DiscoveryResult",
    "synth",
    "validate",
]
