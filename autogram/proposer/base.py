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
from ..dsl.grammar import Grammar, full_ceiling

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
class SearchFeedback:
    """Leakage-safe progress signal handed back to the proposer between outer rounds.

    Every field is derived from the engine's OWN discovered elites -- counts, best score,
    which binders already carry an accepted rule, a coarse plateau flag -- never from the
    ground-truth catalog or the clean frame, so disclosing it is legitimate search feedback,
    not oracle access.
    It exists so the proposer can tell *when* the current grammar has been mined out and
    *which* part of the space is still empty, and widen ``G`` accordingly (Sec. 10.4 outer
    loop) instead of re-proposing forms the middle loop has already explored.
    """
    round_index: int                       # the outer round that just finished (0-based)
    n_accepted: int = 0                    # distinct accepted elites discovered so far
    best_score: float = 0.0                # best combined score among those elites
    filled_cells: int = 0                  # MAP-Elites descriptor cells occupied
    binders_covered: tuple = ()            # binders with >=1 accepted elite
    binders_idle: tuple = ()               # ENABLED binders still with no elite (widen here)
    roles_used: int = 0                    # distinct roles appearing in elite rules
    improved: bool = True                  # did best_score improve vs. the previous round?
    stagnant: bool = False                 # plateau: no improvement and/or idle binders
    hint: str = ""                         # short, leakage-free natural-language hint

    def as_prompt_text(self) -> str:
        """Compact, leakage-checked rendering for the proposer prompt."""
        lines = [
            f"outer round just completed: {self.round_index}",
            f"accepted invariants so far: {self.n_accepted}",
            f"best score so far: {self.best_score:.4f}",
            f"diversity cells filled: {self.filled_cells}",
            f"distinct roles used by accepted rules: {self.roles_used}",
            f"binders with an accepted rule: {list(self.binders_covered)}",
            f"enabled binders still EMPTY: {list(self.binders_idle)}",
            f"improved since previous round: {self.improved}",
            f"stagnant (widen the grammar if so): {self.stagnant}",
        ]
        if self.hint:
            lines.append(f"hint: {self.hint}")
        text = "\n".join(lines)
        assert_no_leakage(text, "SearchFeedback.as_prompt_text")
        return text


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
    ceiling: Optional["Grammar"] = None      # the widen-toward boundary (full_ceiling)
    feedback: Optional["SearchFeedback"] = None   # leakage-safe progress from prior rounds
    round_index: int = 0                     # outer-loop round (0-based); selects per-round reply

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
Return ONLY a JSON object: {"rules": [<rule>, ...], "notes": "<short>", "extension": <optional>}.

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
- For '!=' (separation), compare a quantity ONLY to its own reverse-direction counterpart
  within the SAME measurement family (the forward vs reverse measurement of one quantity).
  Disequalities across different families, or against an aggregate, are rejected as
  uninformative -- do not propose those.
- Aim for structural relationships suggested by the column names/types (conservation,
  pairing of the two ends of a link, a counter vs a sum of demands, network totals,
  non-negativity). Include a few plausible-but-wrong forms too; the evaluator filters them.
- Only use information in this prompt. Do NOT read any repository file or external answer key.

Grammar widening (optional "extension"):
- If the AVAILABLE TO ENABLE block below lists ANY binders/roles/operators/aggregation kinds
  (i.e. the current grammar is narrower than the ceiling), you SHOULD propose an "extension"
  that widens toward them -- the richer structural laws (link pairing, conservation, network
  totals, directional separation) cannot even be expressed until those are enabled. Propose
  the widening on the FIRST round; do not wait. Also widen whenever the SEARCH FEEDBACK reports
  stagnation or an enabled binder with no accepted rule.
- If AVAILABLE TO ENABLE is empty (the grammar already spans the ceiling), OMIT "extension" and
  just propose rules inside the current vocabulary.
- To widen, add: "extension": {"enable_binders":[...], "enable_ref_roles":[...],
  "enable_fam_roles":[...], "enable_ops":[...], "enable_agg_kinds":[...],
  "max_complexity":<int>, "max_add_arity":<int>, "note":"<why>"}.
- Only enable items listed under AVAILABLE TO ENABLE below, and only raise the size caps up
  to the stated ceilings. Anything else (unknown or already-enabled tokens, caps beyond the
  ceiling) is dropped. Prefer the binders reported EMPTY by the feedback when present.
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


def extension_block(G: "Grammar", ceiling: Optional["Grammar"]) -> str:
    """Render what the proposer may legally enable to WIDEN ``G`` toward the ceiling.

    Lists exactly the binders/roles/operators/aggregation kinds present in the ceiling but not
    yet enabled in ``G``, plus the size-cap headroom.
    When ``G`` already equals the ceiling (the default ``start = "full"`` case) every list is
    empty and the caps are maxed, so a proposed extension can only be a no-op -- which is why
    turning the parser on does not perturb the default-grammar runs.
    """
    if ceiling is None:
        ceiling = full_ceiling(G.max_complexity, G.max_add_arity)
    avail_binders = [b for b in ceiling.binders if b not in G.binders]
    avail_refs = [r for r in ceiling.ref_roles if r not in G.ref_roles]
    avail_fams = [r for r in ceiling.fam_roles if r not in G.fam_roles]
    avail_ops = [o for o in ceiling.ops if o not in G.ops]
    avail_aggs = [a for a in ceiling.agg_kinds if a not in G.agg_kinds]
    lines = [
        "AVAILABLE TO ENABLE (widen toward these only):",
        f"  binders: {avail_binders}",
        f"  ref_roles: {avail_refs}",
        f"  family_roles: {avail_fams}",
        f"  operators: {avail_ops}",
        f"  aggregation kinds: {avail_aggs}",
        f"  max_complexity ceiling: {ceiling.max_complexity} (current {G.max_complexity})",
        f"  max_add_arity ceiling: {ceiling.max_add_arity} (current {G.max_add_arity})",
    ]
    if not (avail_binders or avail_refs or avail_fams or avail_ops or avail_aggs):
        lines.append("  (grammar already at the ceiling -- no widening available)")
    return "\n".join(lines)


def render_proposal_prompt(ctx: ProposalContext) -> str:
    """Full leakage-checked prompt string for an LLM/subagent backend."""
    parts = [PROPOSAL_INSTRUCTIONS,
             "\n\nVOCABULARY\n" + vocabulary_block(ctx.grammar),
             "\n\n" + extension_block(ctx.grammar, ctx.ceiling)]
    if ctx.feedback is not None:
        parts.append("\n\nSEARCH FEEDBACK\n" + ctx.feedback.as_prompt_text())
    parts.append("\n\nCONTEXT\n" + ctx.as_prompt_text())
    text = "".join(parts)
    assert_no_leakage(text, "render_proposal_prompt")
    return text


def parse_grammar_extension(ext_obj, G: "Grammar",
                            ceiling: Optional["Grammar"]) -> Optional[GrammarExtension]:
    """Validate a proposed ``extension`` against the ceiling; return ``None`` if it is a no-op.

    Each ``enable_*`` list is intersected with the ceiling vocabulary (unknown tokens dropped)
    and with the complement of what ``G`` already enables (already-on tokens dropped); the size
    caps may only be RAISED and only up to the ceiling.
    This is the trusted gate that keeps an LLM-proposed widening sound: a backend can never
    introduce a role/operator the evaluator cannot ground, nor blow the size caps past the
    configured ceiling.
    """
    if not isinstance(ext_obj, dict):
        return None
    if ceiling is None:
        ceiling = full_ceiling(G.max_complexity, G.max_add_arity)

    def _pick(key, ceil_vocab, cur_vocab):
        raw = ext_obj.get(key, []) or []
        if isinstance(raw, str):
            raw = [raw]
        seen, out = set(), []
        for tok in raw:
            if tok in ceil_vocab and tok not in cur_vocab and tok not in seen:
                out.append(tok)
                seen.add(tok)
        return tuple(out)

    enable_binders = _pick("enable_binders", set(ceiling.binders), set(G.binders))
    enable_refs = _pick("enable_ref_roles", set(ceiling.ref_roles), set(G.ref_roles))
    enable_fams = _pick("enable_fam_roles", set(ceiling.fam_roles), set(G.fam_roles))
    enable_ops = _pick("enable_ops", set(ceiling.ops), set(G.ops))
    enable_aggs = _pick("enable_agg_kinds", set(ceiling.agg_kinds), set(G.agg_kinds))

    def _cap(key, cur, ceil):
        val = ext_obj.get(key)
        if not isinstance(val, (int, float)):
            return None
        val = int(val)
        if val <= cur:                  # only ever RAISE a cap
            return None
        return min(val, ceil)           # never exceed the ceiling

    max_complexity = _cap("max_complexity", G.max_complexity, ceiling.max_complexity)
    max_add_arity = _cap("max_add_arity", G.max_add_arity, ceiling.max_add_arity)

    if not (enable_binders or enable_refs or enable_fams or enable_ops or enable_aggs
            or max_complexity or max_add_arity):
        return None
    note = str(ext_obj.get("note", ""))
    return GrammarExtension(
        enable_ref_roles=enable_refs, enable_fam_roles=enable_fams, enable_ops=enable_ops,
        enable_binders=enable_binders, enable_agg_kinds=enable_aggs,
        max_complexity=max_complexity, max_add_arity=max_add_arity, note=note,
    )


def parse_proposal_json(text: str, G: "Grammar",
                        ceiling: Optional["Grammar"] = None) -> Proposal:
    """Parse a backend's JSON reply into admissible rules + an optional grammar extension.

    Invalid rules are dropped (typed-admissibility check); a proposed ``extension`` is routed
    through :func:`parse_grammar_extension` so only ceiling-legal widenings survive.
    ``ceiling`` defaults to ``G``'s own maximal vocabulary, so a 2-argument call (the test/
    backward-compatible path) can still attach an extension but cannot raise the size caps.
    """
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
    dropped: List[str] = []
    for rd in obj.get("rules", []):
        try:
            r = rule_from_dict({"binder": rd["binder"], "op": rd["op"],
                                "left": rd["left"], "right": rd["right"],
                                "tag": rd.get("tag", "proposed")})
        except Exception as exc:
            dropped.append(f"unparseable ({type(exc).__name__})")
            continue
        ok, why = is_admissible(r, G)
        if ok:
            seeds.append(r)
        else:
            dropped.append(why or "inadmissible")
    notes = str(obj.get("notes", ""))
    if dropped:
        from collections import Counter
        summary = "; ".join(f"{n}x {reason}" for reason, n in Counter(dropped).most_common(4))
        notes = (notes + f" | rejected {len(dropped)} form(s): {summary}").strip(" |")
    extension = parse_grammar_extension(obj.get("extension"), G, ceiling)
    return Proposal(seeds=seeds, notes=notes, extension=extension)


def build_context(ds: Dataset, G: Grammar, n_sample_rows: int = 3,
                  feedback: Optional["SearchFeedback"] = None,
                  ceiling: Optional["Grammar"] = None,
                  round_index: int = 0) -> ProposalContext:
    """Build a leakage-safe :class:`ProposalContext` from observable data only.

    ``ceiling`` and ``feedback`` are the two outer-loop additions: ``ceiling`` tells the
    proposer how far it may widen ``G`` (defaults to ``G``'s own maximal vocabulary, i.e. no
    widening), and ``feedback`` is the leakage-safe progress signal from prior rounds (``None``
    on the first round).
    """
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
    if ceiling is None:
        ceiling = full_ceiling(G.max_complexity, G.max_add_arity)
    ctx = ProposalContext(
        dataset_name=ds.name, grammar=G, columns=cols,
        node_count=len(summary.get("nodes", [])), sample_rows=rows,
        n_rows=obs.n_rows, binder_vocab=vocab, ceiling=ceiling, feedback=feedback,
        round_index=round_index,
    )
    # defensive: ensure nothing oracle-derived slipped into the rendered context
    ctx.as_prompt_text()
    if feedback is not None:
        feedback.as_prompt_text()   # leakage-guard the feedback rendering too
    return ctx
