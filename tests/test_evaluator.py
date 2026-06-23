"""Multi-objective evaluator verdicts on real data (design Sec. 10.1).

These pin the *strictness* the evaluator must assign to the catalogue forms: exact laws come
back EXACT, the ~1.9% structural-deficit laws come back SOFT_STRUCTURAL, the directionality
form comes back ANTI, and a planted decoy is not accepted.
"""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.evaluator.evaluator import evaluate
from autogram.evaluator.gate import Verdict


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right), tag="t")


def test_zero_self_demand_is_exact(abilene, eval_cfg):
    r = _rule("node", A.Ref("demand_self"), "==", A.Const(0))
    res = evaluate(r, abilene, eval_cfg)
    assert res.verdict is Verdict.EXACT
    assert res.accepted


def test_nonnegativity_is_exact(abilene, eval_cfg):
    r = _rule("cell", A.Ref("self"), ">=", A.Const(0))
    res = evaluate(r, abilene, eval_cfg)
    assert res.verdict is Verdict.EXACT
    assert res.accepted


def test_link_agreement_is_exact(abilene, eval_cfg):
    r = _rule("link", A.Ref("egress"), "~=", A.Ref("ingress_rev"))
    res = evaluate(r, abilene, eval_cfg)
    assert res.verdict is Verdict.EXACT


def test_origination_rowsum_is_soft_structural(abilene, eval_cfg):
    """I5 carries a small, *systematic* negative deficit -> SOFT_STRUCTURAL, not EXACT."""
    r = _rule("node", A.Ref("origination"), "~=", A.Agg("SUM", "demand_row"))
    res = evaluate(r, abilene, eval_cfg)
    assert res.verdict is Verdict.SOFT_STRUCTURAL
    assert res.delta < 0           # a deficit, not a surplus


def test_directionality_is_anti(abilene, eval_cfg):
    r = _rule("link", A.Ref("egress"), "!=", A.Ref("egress_rev"))
    res = evaluate(r, abilene, eval_cfg)
    assert res.verdict is Verdict.ANTI


def test_node_flow_conservation_is_exact(abilene, eval_cfg):
    left = A.Add((A.Ref("origination"), A.Agg("SUM", "ingress_fam")))
    right = A.Add((A.Ref("termination"), A.Agg("SUM", "egress_fam")))
    r = _rule("node", left, "~=", right)
    res = evaluate(r, abilene, eval_cfg)
    assert res.verdict is Verdict.EXACT


def test_planted_decoy_not_accepted(abilene, eval_cfg):
    """A spurious equality between unrelated node aggregates must not be accepted as exact."""
    r = _rule("node", A.Ref("origination"), "~=", A.Ref("termination"))
    res = evaluate(r, abilene, eval_cfg)
    # origination != termination per node (only the network *totals* balance); so this is
    # either rejected or, at best, a loose soft fit -- never EXACT.
    assert res.verdict is not Verdict.EXACT
