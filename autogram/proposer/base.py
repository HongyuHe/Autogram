"""LLM/subagent proposer interface (design Sec. 6.4, 10.4; task: two swappable backends).

The proposer is the *outer loop*: it proposes grammar extensions and candidate invariant
*forms* (never thresholds -- those are fit by the evaluator).  Two backends implement this
ABC interchangeably (OpenAI API, isolated subagent) plus a deterministic scripted default
for tests/reproducible runs.

**Leakage discipline (task-critical).**  A proposer must only ever see information that is
legitimately available during learning: column *names* and inferred types, the current
grammar, the node count, and a few *observed* (noisy) sample rows.  It must NEVER see the
ground-truth catalog ``docs/abilene_geant_invariants.md`` or any derived oracle.
:func:`build_context` constructs the context from the dataset's leakage-safe
``observable_summary`` only, and :func:`assert_no_leakage` is a defensive guard that scans
any proposer-facing text for forbidden oracle tokens and raises if one appears.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..loader.loader import Dataset
from ..dsl.ast import Rule
from ..dsl.grammar import Grammar

# Tokens that would only appear if a proposer had read the ground-truth catalog.
# (Labels I1..I10, the systematic-deficit finding, the catalog filename, oracle words.)
_FORBIDDEN = [
    r"\bI(?:10|[1-9])\b", r"abilene_geant_invariants", r"ground[\s_-]*truth",
    r"hidden_ground_truth", r"\b1\.9\s*%", r"systematic deficit", r"structural deficit",
    r"oracle", r"crosscheck invariant", r"\bCC-\d",
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in _FORBIDDEN]


class LeakageError(RuntimeError):
    """Raised when proposer-facing text contains a forbidden oracle token."""


def assert_no_leakage(text: str, where: str = "") -> None:
    for rx in _FORBIDDEN_RE:
        m = rx.search(text or "")
        if m:
            raise LeakageError(f"forbidden oracle token {m.group(0)!r} in {where or 'text'}")


@dataclass
class GrammarExtension:
    """A proposed widening of the search space (all fields are unions onto ``G``)."""
    enable_ref_roles: tuple = ()
    enable_fam_roles: tuple = ()
    enable_ops: tuple = ()
    enable_binders: tuple = ()
    enable_agg_kinds: tuple = ()
    max_complexity: Optional[int] = None
    max_add_arity: Optional[int] = None
    note: str = ""


@dataclass
class Proposal:
    """The proposer's output: an optional grammar extension plus candidate forms."""
    seeds: List[Rule] = field(default_factory=list)
    extension: Optional[GrammarExtension] = None
    notes: str = ""


@dataclass
class ProposalContext:
    """Leakage-safe information handed to a proposer (no oracle, no clean frame)."""
    dataset_name: str
    grammar: Grammar
    columns: List[dict]                 # [{name, kind, role/dir, ...}] from names only
    node_count: int
    sample_rows: List[dict]             # a few OBSERVED rows (noisy), column->value
    n_rows: int
    binder_vocab: dict                  # binder -> {ref_roles, fam_roles} currently enabled
    notes: str = ""

    def as_prompt_text(self) -> str:
        """A compact, leakage-checked textual rendering for LLM/subagent backends."""
        lines = [
            f"dataset: {self.dataset_name}", f"nodes: {self.node_count}",
            f"snapshots: {self.n_rows}", "columns (name : kind : direction/role):",
        ]
        for c in self.columns[:60]:
            lines.append(f"  {c.get('name')} : {c.get('kind')} : {c.get('detail','')}")
        lines.append("current grammar binders/roles:")
        for b, v in self.binder_vocab.items():
            lines.append(f"  {b}: refs={sorted(v['ref_roles'])} fams={sorted(v['fam_roles'])}")
        text = "\n".join(lines)
        assert_no_leakage(text, "ProposalContext.as_prompt_text")
        return text


class Proposer(ABC):
    """Abstract proposer; backends are drop-in interchangeable via config."""
    name: str = "abstract"

    @abstractmethod
    def propose(self, ctx: ProposalContext) -> Proposal:
        ...


# ---------------------------------------------------------------------------
# Shared JSON proposal contract (used by the OpenAI + subagent backends)
# ---------------------------------------------------------------------------

PROPOSAL_INSTRUCTIONS = """\
You are proposing candidate INVARIANT FORMS for a network-traffic dataset.
Return ONLY a JSON object: {"rules": [<rule>, ...], "notes": "<short>"}.

A <rule> is {"binder","op","left","right"} where left/right are TERMS:
  {"k":"Ref","role":<ref_role>}                     a single column for the binder
  {"k":"Const","value":<number>}                    a constant (use 0 for bounds)
  {"k":"Agg","kind":<SUM|MIN|MAX|AVG>,"family_role":<fam_role>}   aggregate of a family
  {"k":"Scale","coeff":<number>,"term":<term>}      scalar multiple
  {"k":"Add","terms":[<term>,...]}                  n-ary sum of terms

Rules:
- Propose FORMS ONLY. Do NOT propose numeric tolerances/thresholds; those are fit from data.
- Use only the binders, operators, ref_roles, family_roles and aggregation kinds listed
  in the vocabulary below. A term must be valid for its binder.
- Operators: '~=' approx-equal, '==' exact-equal, '<=' / '>=' bounds, '!=' separation.
- Aim for structural relationships suggested by the column names/types (conservation,
  pairing of the two ends of a link, a counter vs a sum of demands, network totals,
  non-negativity). Include a few plausible-but-wrong forms too; the evaluator filters them.
- Only use information in this prompt. Do NOT read any repository file or external answer key.
"""


def vocabulary_block(G: "Grammar") -> str:
    """Render the grammar's legal vocabulary per binder (leakage-free)."""
    from ..dsl import ast as A
    lines = [f"operators: {list(G.ops)}", f"aggregation kinds: {list(G.agg_kinds)}",
             f"max_complexity: {G.max_complexity}  max_add_arity: {G.max_add_arity}",
             "binders and their legal roles:"]
    for b in G.binders:
        refs = [r for r in A.REF_ROLES.get(b, ()) if r in G.ref_roles]
        fams = [r for r in A.FAM_ROLES.get(b, ()) if r in G.fam_roles]
        lines.append(f"  {b}: ref_roles={refs} family_roles={fams}")
    return "\n".join(lines)


def render_proposal_prompt(ctx: ProposalContext) -> str:
    """Full leakage-checked prompt string for an LLM/subagent backend."""
    text = (PROPOSAL_INSTRUCTIONS + "\n\nVOCABULARY\n" + vocabulary_block(ctx.grammar)
            + "\n\nCONTEXT\n" + ctx.as_prompt_text())
    assert_no_leakage(text, "render_proposal_prompt")
    return text


def parse_proposal_json(text: str, G: "Grammar") -> Proposal:
    """Parse a backend's JSON reply into admissible rules (drops invalid ones)."""
    import json
    from ..dsl.parser import rule_from_dict
    from ..dsl.typecheck import is_admissible
    assert_no_leakage(text, "parse_proposal_json")
    # tolerate code-fenced or chatty replies: extract the outermost JSON object
    s = text.strip()
    if "```" in s:
        s = s.split("```")[1] if s.count("```") >= 2 else s
        s = s[s.find("{"):]
    start, end = s.find("{"), s.rfind("}")
    obj = json.loads(s[start:end + 1]) if (start >= 0 and end > start) else {"rules": []}
    seeds: List[Rule] = []
    for rd in obj.get("rules", []):
        try:
            r = rule_from_dict({"binder": rd["binder"], "op": rd["op"],
                                "left": rd["left"], "right": rd["right"],
                                "tag": rd.get("tag", "proposed")})
        except Exception:
            continue
        ok, _ = is_admissible(r, G)
        if ok:
            seeds.append(r)
    return Proposal(seeds=seeds, notes=str(obj.get("notes", "")))


def build_context(ds: Dataset, G: Grammar, n_sample_rows: int = 3) -> ProposalContext:
    """Build a leakage-safe :class:`ProposalContext` from observable data only."""
    summary = ds.observable_summary()
    nm = ds.name_model
    # per-column descriptors built from the NAME parser only (no oracle, no values)
    cols = []
    for c in ds.observed.names:
        sem = nm.by_name.get(c)
        if sem is None:
            cols.append({"name": c, "kind": "meta", "detail": ""})
        else:
            cols.append({"name": c, "kind": sem.kind, "detail": sem.direction})
    vocab = {}
    from ..dsl import ast as A
    for b in G.binders:
        vocab[b] = {
            "ref_roles": [r for r in A.REF_ROLES.get(b, ()) if r in G.ref_roles],
            "fam_roles": [r for r in A.FAM_ROLES.get(b, ()) if r in G.fam_roles],
        }
    # a few OBSERVED rows only (never the clean frame)
    obs = ds.observed
    names = obs.names[: min(len(obs.names), 40)]
    rows = []
    for t in range(min(n_sample_rows, obs.n_rows)):
        rows.append({col_name: float(obs.col(col_name)[t]) for col_name in names})
    ctx = ProposalContext(
        dataset_name=ds.name, grammar=G, columns=cols,
        node_count=len(summary.get("nodes", [])), sample_rows=rows,
        n_rows=obs.n_rows, binder_vocab=vocab,
    )
    # defensive: ensure nothing oracle-derived slipped into the rendered context
    ctx.as_prompt_text()
    return ctx
