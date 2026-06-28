"""OpenAI-API proposer backend (one of the two swappable LLM backends).

The API key is **never hardcoded**: it is read from the ``OPENAI_API_KEY`` environment
variable (or passed explicitly via config).  If no key is present the backend degrades
gracefully to an empty proposal so the engine still runs (the evolutionary search then
proceeds from whatever seeds the round already has).

SDK note (verified against openai-python v2.x, 2024-06): the modern surface is
``from openai import OpenAI; client = OpenAI(api_key=...)`` followed by
``client.chat.completions.create(model=..., messages=[...], response_format={"type":"json_object"})``.
The key is also picked up from ``OPENAI_API_KEY`` automatically when ``api_key`` is omitted.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import (Proposal, Proposer, ProposalContext, assert_no_leakage,
                   parse_proposal_json, render_proposal_prompt)


class OpenAIProposer(Proposer):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 temperature: float = 0.4, max_tokens: int = 1500) -> None:
        # Blank/configurable by design: read from env when not supplied; do NOT hardcode.
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_notes = ""                          # proposer feedback for the run report

    def propose(self, ctx: ProposalContext) -> Proposal:
        if not self.api_key:
            self.last_notes = "no OPENAI_API_KEY set; openai backend skipped"
            return Proposal(seeds=[], notes=self.last_notes)
        prompt = render_proposal_prompt(ctx)          # leakage-checked
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            self.last_notes = f"openai SDK unavailable: {exc}"
            return Proposal(seeds=[], notes=self.last_notes)
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You output only compact JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or "{}"
        assert_no_leakage(text, "openai response")     # defensive
        prop = parse_proposal_json(text, ctx.grammar, ctx.ceiling)
        detail = f" | {prop.notes}" if prop.notes else ""
        prop.notes = f"openai:{self.model} -> {len(prop.seeds)} admissible forms{detail}"
        self.last_notes = prop.notes
        return prop
