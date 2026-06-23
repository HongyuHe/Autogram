"""Recall scorer: portfolio vs. the known catalogue (FULL / PARTIAL / MISSED).

Uses lightweight stand-ins for ``EvaluationResult`` -- the scorer only reads ``.rule``,
``.verdict`` and ``.summary()`` -- so this stays a fast unit test independent of a full run.
"""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.evaluator.gate import Verdict
from autogram.search.recall import score_recall


class _Fake:
    def __init__(self, rule, verdict):
        self.rule = rule
        self.verdict = verdict

    def summary(self):
        return self.rule.signature()


def _r(binder, left, op, right, verdict):
    return _Fake(A.Rule(binder, A.Compare(left, op, right), tag="t"), verdict)


def _full_portfolio():
    return [
        _r("cell", A.Ref("self"), ">=", A.Const(0), Verdict.EXACT),                       # I1
        _r("node", A.Ref("demand_self"), "==", A.Const(0), Verdict.EXACT),                # I2
        _r("link", A.Ref("egress"), "~=", A.Ref("ingress_rev"), Verdict.EXACT),          # I4
        _r("node", A.Ref("origination"), "~=", A.Agg("SUM", "demand_row"),
           Verdict.SOFT_STRUCTURAL),                                                       # I5
        _r("node", A.Ref("termination"), "~=", A.Agg("SUM", "demand_col"),
           Verdict.SOFT_STRUCTURAL),                                                       # I6
        _r("node", A.Add((A.Ref("origination"), A.Agg("SUM", "ingress_fam"))), "~=",
           A.Add((A.Ref("termination"), A.Agg("SUM", "egress_fam"))), Verdict.EXACT),     # I7
        _r("network", A.Agg("SUM", "all_orig"), "~=", A.Agg("SUM", "all_term"),
           Verdict.EXACT),                                                                 # I8
        _r("link", A.Ref("egress"), "!=", A.Ref("egress_rev"), Verdict.ANTI),            # I9
    ]


def test_full_portfolio_scores_eight_of_eight():
    rep = score_recall(_full_portfolio())
    assert rep.n_targets == 8
    assert rep.n_full == 8
    assert rep.recall == 1.0
    assert rep.strict_recall == 1.0


def test_wrong_strictness_is_partial_not_full():
    port = _full_portfolio()
    # report I5 as EXACT instead of SOFT_STRUCTURAL -> form recovered, strictness wrong.
    port[3] = _r("node", A.Ref("origination"), "~=", A.Agg("SUM", "demand_row"),
                 Verdict.EXACT)
    rep = score_recall(port)
    assert rep.n_recovered == 8         # still recovered (form matches)
    assert rep.n_full == 7              # but not at full strictness
    i5 = next(m for m in rep.matches if m.tid == "I5")
    assert i5.status == "PARTIAL"


def test_missing_rule_is_missed():
    port = _full_portfolio()
    del port[-1]                        # drop I9
    rep = score_recall(port)
    assert rep.n_recovered == 7
    i9 = next(m for m in rep.matches if m.tid == "I9")
    assert i9.status == "MISSED"


def test_out_of_scope_excluded_from_denominator():
    rep = score_recall(_full_portfolio())
    oos = {m.tid for m in rep.matches if m.status == "OUT_OF_SCOPE"}
    assert oos == {"I3", "I10"}
    assert rep.n_targets == 8           # I3/I10 not counted
