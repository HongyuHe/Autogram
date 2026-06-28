"""Assembly helpers: residual fingerprints + cancellation detection (design Sec. 10.3)."""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.search.assemble import (
    _bound_residual,
    _has_cancellation,
    _pure_sign_bound,
    _residual_keys,
    _sign_tautology,
)


def _rule(binder, left, op, right):
    return A.Rule(binder, A.Compare(left, op, right), tag="t")


def test_residual_keys_cancel_shared_term():
    """o(X) + SUM(H[X,*])  ~=  SUM(H[X,*])  reduces, after cancellation, to {origination}."""
    left = A.Add((A.Ref("origination"), A.Agg("SUM", "demand_row")))
    right = A.Agg("SUM", "demand_row")
    keys = _residual_keys(_rule("node", left, "~=", right))
    assert keys == {"origination"}


def test_has_cancellation_flags_padded_equality():
    left = A.Add((A.Ref("origination"), A.Agg("SUM", "demand_row")))
    right = A.Agg("SUM", "demand_row")
    assert _has_cancellation(_rule("node", left, "~=", right)) is True


def test_clean_law_has_no_cancellation():
    r = _rule("node", A.Ref("origination"), "~=", A.Agg("SUM", "demand_row"))
    assert _has_cancellation(r) is False
    assert _residual_keys(r) == {"origination", "SUM(H[X,*])"}


def test_kirchhoff_residual_keys():
    left = A.Add((A.Ref("origination"), A.Agg("SUM", "ingress_fam")))
    right = A.Add((A.Ref("termination"), A.Agg("SUM", "egress_fam")))
    keys = _residual_keys(_rule("node", left, "~=", right))
    assert keys == {"origination", "SUM(i[X<-*])", "termination", "SUM(e[X->*])"}


# --- sign-entailment tautology filter (design Sec. 10.3 / poc-eval E2) ----------------------

def test_pure_sign_bound_is_the_nonnegativity_schema():
    """``v >= 0`` (target I1): a sign tautology that reduces to a single measured term."""
    r = _rule("cell", A.Ref("self"), ">=", A.Const(0))
    assert _sign_tautology(r) is True
    assert _pure_sign_bound(r) is True
    assert _bound_residual(r) == {"self": 1.0}


def test_scaled_nonnegativity_is_still_pure_sign():
    """``3*v >= 0`` is the same schema as ``v >= 0`` (one term, no offset)."""
    r = _rule("cell", A.Scale(3.0, A.Ref("self")), ">=", A.Const(0))
    assert _pure_sign_bound(r) is True
    # ...but it is the bulkier form: the parsimony tie-break prefers the bare ``v >= 0``.
    assert r.complexity() > _rule("cell", A.Ref("self"), ">=", A.Const(0)).complexity()


def test_compound_sign_tautology_is_not_pure_and_is_dropped():
    """``i >= -H`` holds for every non-negative assignment but carries two terms -> dropped."""
    r = _rule("node", A.Ref("origination"),
              ">=", A.Scale(-1.0, A.Agg("SUM", "demand_row")))
    assert _sign_tautology(r) is True          # true by signs alone (o + SUM(H) >= 0)
    assert _pure_sign_bound(r) is False         # ...but not the single-term schema
    assert len([k for k in _bound_residual(r) if k != "#const"]) == 2


def test_genuine_bound_with_negative_coeff_is_not_a_tautology():
    """``MIN(t) <= SUM(o)`` is data-supported (not entailed by signs) -> kept as genuine."""
    r = _rule("network", A.Agg("MIN", "all_term"), "<=", A.Agg("SUM", "all_orig"))
    assert _sign_tautology(r) is False
    assert _pure_sign_bound(r) is False


def test_equality_is_never_a_sign_tautology():
    r = _rule("link", A.Ref("egress"), "~=", A.Ref("ingress_rev"))
    assert _sign_tautology(r) is False
    assert _bound_residual(r) == {}
