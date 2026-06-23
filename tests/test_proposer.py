"""Proposer backends + leakage discipline (task: two swappable backends, isolation).

No live API is ever called: the OpenAI path is exercised through a monkeypatched SDK client,
and the subagent path through an in-process ``responder`` callback.
"""

from __future__ import annotations

import json

import pytest

from autogram.dsl.grammar import default_grammar
from autogram.proposer import build_context, make_proposer
from autogram.proposer.base import (LeakageError, assert_no_leakage, parse_proposal_json,
                                    render_proposal_prompt)

# Two valid forms + one trivial self-comparison that the admissibility filter must drop.
_CANNED = json.dumps({
    "rules": [
        {"binder": "node", "op": "==",
         "left": {"k": "Ref", "role": "demand_self"}, "right": {"k": "Const", "value": 0}},
        {"binder": "link", "op": "~=",
         "left": {"k": "Ref", "role": "egress"}, "right": {"k": "Ref", "role": "ingress_rev"}},
        {"binder": "node", "op": "~=",
         "left": {"k": "Ref", "role": "origination"},
         "right": {"k": "Ref", "role": "origination"}},
    ],
    "notes": "mock proposal",
})


# ------------------------------------------------------------------------- JSON contract

def test_parse_proposal_drops_inadmissible():
    G = default_grammar()
    prop = parse_proposal_json(_CANNED, G)
    # self-comparison dropped -> exactly the two valid forms survive.
    assert len(prop.seeds) == 2
    sigs = {r.signature() for r in prop.seeds}
    assert any("H[X,X]" in s for s in sigs)


# ----------------------------------------------------------------------- leakage guards

@pytest.mark.parametrize("bad", [
    "consider rule I5 over the rows",
    "see docs/abilene_geant_invariants.md",
    "the ground truth says origination equals the row sum",
    "this is the oracle answer",
    "a 1.9% structural deficit",
])
def test_assert_no_leakage_raises(bad):
    with pytest.raises(LeakageError):
        assert_no_leakage(bad, "test")


def test_clean_prompt_has_no_leakage(abilene):
    G = default_grammar()
    ctx = build_context(abilene, G)
    prompt = render_proposal_prompt(ctx)        # raises if it leaks
    assert "demand_self" in prompt or "origination" in prompt
    # the catalogue's identifiers must not appear in a learning-time prompt.
    assert "I5" not in prompt and "abilene_geant_invariants" not in prompt


# --------------------------------------------------------------- subagent backend (mock)

def test_subagent_responder_parses_forms(abilene, tmp_path):
    G = default_grammar()
    ctx = build_context(abilene, G)
    prop_holder = {}

    def responder(prompt: str) -> str:
        prop_holder["prompt"] = prompt
        return _CANNED

    sub = make_proposer("subagent", work_dir=str(tmp_path), dataset="abilene",
                        responder=responder)
    prop = sub.propose(ctx)
    assert sub.used_real_subagent is True
    assert len(prop.seeds) == 2
    # the prompt the subagent saw must itself be leakage-free.
    assert_no_leakage(prop_holder["prompt"], "captured prompt")


def test_subagent_rescans_response_for_leakage(abilene, tmp_path):
    G = default_grammar()
    ctx = build_context(abilene, G)
    leaky = json.dumps({"rules": [], "notes": "this matches catalogue rule I7"})
    sub = make_proposer("subagent", work_dir=str(tmp_path), dataset="abilene",
                        responder=lambda p: leaky)
    with pytest.raises(LeakageError):
        sub.propose(ctx)


def test_subagent_empty_when_no_reply(abilene, tmp_path):
    G = default_grammar()
    ctx = build_context(abilene, G)
    sub = make_proposer("subagent", work_dir=str(tmp_path), dataset="abilene")
    prop = sub.propose(ctx)                      # no responder, no response file
    assert prop.seeds == []
    assert sub.used_real_subagent is False       # honestly reported


# ------------------------------------------------------------------ openai backend (mock)

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def create(self, **kw):
        return _FakeResp(_CANNED)


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    def __init__(self, *a, **k):
        self.chat = _FakeChat()


def test_openai_backend_with_mocked_sdk(abilene, monkeypatch):
    openai = pytest.importorskip(
        "openai", reason="OpenAI backend is optional; run `uv sync --extra openai` to test it"
    )
    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    G = default_grammar()
    ctx = build_context(abilene, G)
    prop = make_proposer("openai", api_key="sk-test-not-real").propose(ctx)
    assert len(prop.seeds) == 2                  # same contract, same admissibility filter


def test_openai_backend_skips_without_key(abilene, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    G = default_grammar()
    ctx = build_context(abilene, G)
    prop = make_proposer("openai", api_key="").propose(ctx)
    assert prop.seeds == []                       # graceful no-op, no network call
