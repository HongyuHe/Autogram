"""Proposer backends and a lazy factory.

Kept minimal and import-light on purpose: backends are imported *inside* :func:`make_proposer`
so importing this package never eagerly pulls in the OpenAI SDK or the search module (avoids
an import cycle, since the scripted/subagent backends import from ``..dsl`` and ``..search``).
"""

from __future__ import annotations

from typing import Callable, Optional

from .base import (GrammarExtension, LeakageError, Proposal, ProposalContext, Proposer,
                   assert_no_leakage, build_context)

__all__ = [
    "Proposer", "Proposal", "ProposalContext", "GrammarExtension", "LeakageError",
    "assert_no_leakage", "build_context", "make_proposer",
]


def make_proposer(name: str, *, work_dir: str = ".", dataset: str = "dataset",
                  model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                  responder: Optional[Callable[[str], str]] = None) -> Proposer:
    """Construct a proposer backend by name (``scripted`` | ``openai`` | ``subagent``)."""
    key = (name or "scripted").lower()
    if key == "scripted":
        from .scripted_backend import ScriptedProposer
        return ScriptedProposer()
    if key == "openai":
        from .openai_backend import OpenAIProposer
        return OpenAIProposer(model=model, api_key=api_key)
    if key == "subagent":
        from .subagent_backend import SubagentProposer
        return SubagentProposer(work_dir=work_dir, dataset=dataset, responder=responder)
    raise ValueError(f"unknown proposer backend {name!r}")
