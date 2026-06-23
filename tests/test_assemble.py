"""Assembly helpers: residual fingerprints + cancellation detection (design Sec. 10.3)."""

from __future__ import annotations

from autogram.dsl import ast as A
from autogram.search.assemble import _has_cancellation, _residual_keys


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
