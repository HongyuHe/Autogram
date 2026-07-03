"""Item 2: role co-occurrence blocklist (default-allow, proposer-denied pairs)."""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.dsl.grammar import Grammar
from autogram.dsl.typecheck import is_admissible


def _grammar(exclusions=()):
    return Grammar(
        binders=("node",),
        ops=("~=", "==", "<=", ">="),
        ref_roles={"node": ("temp", "count", "demand")},
        fam_roles={"node": ()},
        agg_kinds=("SUM",),
        scale_coeffs=(-1.0,),
        max_complexity=12,
        max_add_arity=2,
        max_degree=2,
        role_exclusions=tuple(exclusions),
    )


def test_excluded_pair_is_rejected():
    G = _grammar([frozenset({"temp", "count"})])
    bad = A.Rule("node", A.Compare(A.Ref("temp"), "~=", A.Ref("count")))
    assert is_admissible(bad, G)[0] is False


def test_unrelated_pair_still_allowed():
    G = _grammar([frozenset({"temp", "count"})])
    ok = A.Rule("node", A.Compare(A.Ref("temp"), "~=", A.Ref("demand")))
    assert is_admissible(ok, G)[0] is True


def test_exclusion_covers_product_operands():
    # temp * count is blocked because _leaf_set recurses into the product
    G = _grammar([frozenset({"temp", "count"})])
    bad = A.Rule("node", A.Compare(A.Mul(A.Ref("temp"), A.Ref("count")), "~=", A.Ref("demand")))
    assert is_admissible(bad, G)[0] is False


def test_no_exclusions_allows_everything():
    G = _grammar([])
    r = A.Rule("node", A.Compare(A.Ref("temp"), "~=", A.Ref("count")))
    assert is_admissible(r, G)[0] is True
