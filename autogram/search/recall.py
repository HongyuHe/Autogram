"""Score a learned portfolio against the *known* invariant catalogue (recall).

This module is the **validation oracle**: unlike the proposer (which must never see the
answers, Sec. 6.4), the recall scorer is allowed to reference the catalogue in
``docs/abilene_geant_invariants.md`` because it *is* the grader.  It is kept in a separate
module precisely so that nothing on the learning path imports it.

Each catalogue target is encoded by its **residual fingerprint** -- the binder, the relation
kind (equality / sign-bound / anti), and the multiset of measurement terms left after moving
both sides of the (in)equality onto one side and cancelling like terms.  The fingerprint is
computed with the very same ``assemble._residual_keys`` used during portfolio assembly, so
the grader cannot drift from the learner's own notion of "the same relation".

A target counts as

* **FULL**   -- a portfolio rule matches its form *and* the expected strictness
  (e.g. I5 recovered as ``SOFT_STRUCTURAL`` with a small negative deficit), or
* **PARTIAL** -- a rule matches the form / term-pair but with a different strictness or band
  (right shape, wrong threshold), or
* **MISSED** -- no rule matches.

The two structural catalogue entries that the numeric per-point DSL *cannot* express --
**I3** (reverse-link existence) and **I10** (demand->link routing, no routing matrix in the
sample) -- are reported as ``OUT_OF_SCOPE`` and excluded from the recall denominator, exactly
as the catalogue itself flags them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from ..evaluator.evaluator import EvaluationResult
from ..evaluator.gate import Verdict
from .assemble import _residual_keys


def _kind(op: str) -> str:
    if op in ("~=", "=="):
        return "eq"
    if op == "!=":
        return "anti"
    if op == ">=":
        return "ge"
    if op == "<=":
        return "le"
    return op


# --- catalogue targets (docs/abilene_geant_invariants.md) -------------------------------
# Each target: id, label, binder, relation kind, the set of acceptable residual keysets,
# and the expected strictness verdict(s).  Keys mix Ref role-names (e.g. "egress",
# "origination", "self", "demand_self") with Agg unparses (e.g. "SUM(H[X,*])") exactly as
# assemble._residual_keys emits them.

@dataclass(frozen=True)
class Target:
    tid: str
    label: str
    binder: str
    kind: str
    keysets: Tuple[frozenset, ...]
    expect: Tuple[Verdict, ...]
    # I1 is "any single measurement >= 0"; a size-1 ge residual matches regardless of which
    # measurement, so its keysets tuple is empty and a wildcard flag is used instead.
    wildcard_single_ge: bool = False


TESTABLE_TARGETS: Tuple[Target, ...] = (
    Target("I1", "non-negativity  v >= 0", "any", "ge",
           (), (Verdict.EXACT,), wildcard_single_ge=True),
    Target("I2", "zero self-demand  H[X,X] = 0", "node", "eq",
           (frozenset({"demand_self"}),), (Verdict.EXACT,)),
    Target("I4", "link two-end agreement  e[X->Y] = i[Y<-X]", "link", "eq",
           (frozenset({"egress", "ingress_rev"}),), (Verdict.EXACT,)),
    Target("I5", "origination = demand row-sum  o(X) = SUM H[X,*]", "node", "eq",
           (frozenset({"origination", "SUM(H[X,*])"}),), (Verdict.SOFT_STRUCTURAL,)),
    Target("I6", "termination = demand col-sum  t(X) = SUM H[*,X]", "node", "eq",
           (frozenset({"termination", "SUM(H[*,X])"}),), (Verdict.SOFT_STRUCTURAL,)),
    Target("I7", "node flow conservation  o+SUM i = t+SUM e", "node", "eq",
           (frozenset({"origination", "SUM(i[X<-*])",
                       "termination", "SUM(e[X->*])"}),), (Verdict.EXACT,)),
    Target("I8", "network totals balance  SUM o ~ SUM t ~ SUM H", "network", "eq",
           (frozenset({"SUM(o(*))", "SUM(t(*))"}),
            frozenset({"SUM(t(*))", "SUM(H[*,*])"}),
            frozenset({"SUM(o(*))", "SUM(H[*,*])"})),
           (Verdict.EXACT, Verdict.SOFT_STRUCTURAL, Verdict.SOFT)),
    Target("I9", "directionality anti  e[X->Y] != e[Y->X]", "link", "anti",
           (frozenset({"egress", "egress_rev"}),), (Verdict.ANTI,)),
)

OUT_OF_SCOPE = (
    ("I3", "topology reverse-link existence (structural, not a per-point numeric relation)"),
    ("I10", "demand->link routing (no routing matrix in the sample)"),
)


@dataclass
class Match:
    tid: str
    label: str
    status: str                       # FULL | PARTIAL | MISSED | OUT_OF_SCOPE
    rule_summary: Optional[str] = None
    note: str = ""


@dataclass
class RecallReport:
    matches: List[Match] = field(default_factory=list)
    n_targets: int = 0
    n_full: int = 0
    n_recovered: int = 0              # full + partial

    @property
    def recall(self) -> float:
        return self.n_recovered / self.n_targets if self.n_targets else 0.0

    @property
    def strict_recall(self) -> float:
        return self.n_full / self.n_targets if self.n_targets else 0.0


def _rule_fingerprint(r: EvaluationResult) -> Tuple[str, str, Set[str]]:
    return (r.rule.binder, _kind(r.rule.atom.op), _residual_keys(r.rule))


def _matches_form(t: Target, binder: str, kind: str, keys: Set[str]) -> bool:
    if kind != t.kind:
        return False
    if t.binder != "any" and binder != t.binder:
        return False
    if t.wildcard_single_ge:
        return len(keys) == 1            # a single non-negative measurement vs 0
    return any(set(ks) == keys for ks in t.keysets)


def score_recall(portfolio: List[EvaluationResult]) -> RecallReport:
    """Match a learned ``portfolio`` against the testable catalogue targets."""
    return score_against(portfolio, TESTABLE_TARGETS, OUT_OF_SCOPE)


def score_against(portfolio: List[EvaluationResult],
                  targets: Tuple[Target, ...],
                  out_of_scope: Tuple[Tuple[str, str], ...] = ()) -> RecallReport:
    """Generic recall scorer: match a portfolio against an arbitrary ``targets`` set.

    Factored out of :func:`score_recall` so a *different* benchmark (e.g. the schema-general
    second benchmark, which carries its own planted ground truth) can be graded by the very
    same form-and-strictness matcher without importing the CrossCheck catalogue.
    """
    fps = [(_rule_fingerprint(r), r) for r in portfolio]
    rep = RecallReport(n_targets=len(targets))

    for t in targets:
        full: Optional[EvaluationResult] = None
        partial: Optional[EvaluationResult] = None
        for (binder, kind, keys), r in fps:
            if not _matches_form(t, binder, kind, keys):
                continue
            if r.verdict in t.expect:
                full = full or r
            else:
                partial = partial or r
        if full is not None:
            rep.matches.append(Match(t.tid, t.label, "FULL", full.summary()))
            rep.n_full += 1
            rep.n_recovered += 1
        elif partial is not None:
            rep.matches.append(Match(
                t.tid, t.label, "PARTIAL", partial.summary(),
                note="form recovered, strictness/band differs from catalogue"))
            rep.n_recovered += 1
        else:
            rep.matches.append(Match(t.tid, t.label, "MISSED"))

    for tid, why in out_of_scope:
        rep.matches.append(Match(tid, why, "OUT_OF_SCOPE",
                                 note="not expressible in the numeric per-point DSL"))
    return rep


def format_report(rep: RecallReport) -> str:
    """Render an ASCII recall table (console is cp1252 -- no non-ASCII glyphs)."""
    lines = []
    lines.append("Recall vs. the known invariant catalogue (I1-I10)")
    lines.append("-" * 78)
    for m in rep.matches:
        head = f"  {m.tid:<4} {m.status:<12} {m.label}"
        lines.append(head)
        if m.rule_summary:
            lines.append(f"           -> {m.rule_summary.strip()}")
        if m.note:
            lines.append(f"           ({m.note})")
    lines.append("-" * 78)
    lines.append(
        f"  testable targets : {rep.n_targets}"
        f"   |  recovered (full+partial): {rep.n_recovered}"
        f"   |  full-strictness: {rep.n_full}")
    lines.append(
        f"  recall           : {rep.recall:6.2%}"
        f"   |  strict recall          : {rep.strict_recall:6.2%}")
    lines.append("  out-of-scope     : I3 (topology existence), I10 (routing) "
                 "-- excluded from denominator")
    return "\n".join(lines)
