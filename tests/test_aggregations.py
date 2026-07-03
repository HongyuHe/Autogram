"""Item 5: the enumerator emits every aggregation the grammar declares (not just SUM)."""

from __future__ import annotations

from autogram.dsl.grammar import Grammar
from autogram.discovery.propose import EnumerationProposer


def _grammar(agg_kinds):
    return Grammar(
        binders=("node",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"node": ("m_a",)},
        fam_roles={"node": ("fam_x",)},
        agg_kinds=tuple(agg_kinds),
        scale_coeffs=(-1.0,),
        max_complexity=10,
        max_add_arity=2,
    )


def _rendered(agg_kinds):
    rules = EnumerationProposer(_grammar(agg_kinds)).propose(0, (), None)
    return "\n".join(r.unparse() for r in rules)


def test_sum_only_grammar_emits_only_sum():
    text = _rendered(("SUM",))
    assert "SUM(fam_x)" in text
    for k in ("MIN", "MAX", "AVG"):
        assert f"{k}(fam_x)" not in text


def test_grammar_with_all_aggregations_emits_all():
    text = _rendered(("SUM", "MIN", "MAX", "AVG"))
    for k in ("SUM", "MIN", "MAX", "AVG"):
        assert f"{k}(fam_x)" in text, f"{k} not enumerated"


def test_proposer_choice_is_respected():
    text = _rendered(("SUM", "MAX"))
    assert "MAX(fam_x)" in text
    assert "MIN(fam_x)" not in text and "AVG(fam_x)" not in text
