"""Static admissibility / typing for DSL rules.

Totality, determinism, and absence of side effects are guaranteed *by construction* of the
AST (no recursion, no unbounded loops, no I/O), so "type checking" here reduces to two cheap,
decidable checks:

1. **Well-formedness w.r.t. the binder** -- every ``Ref`` role and ``Agg`` family role used by
   the atom is one the current binder actually exposes *in the induced grammar* (no fixed
   vocabulary), and operators/agg kinds are in the grammar's allowed sets.
2. **Dimensional admissibility** -- all measured quantities are treated as one comparable
   dimension, so the only restriction is that a comparison's two sides are both measured
   (a bare non-zero constant may appear only behind a scalar multiply or against an ordering
   operator, so ``v >= 0`` is fine but ``v ~= 5`` is rejected as meaningless).

Roles are validated against the *induced* schema carried by the grammar, never a hardcoded
role table -- this is what lets genuinely new roles (invented for a fresh dataset) pass.
"""

from __future__ import annotations

from . import ast as A


def _roles_ok(term: A.Term, binder: str, G) -> bool:
    if isinstance(term, A.Const):
        return True
    if isinstance(term, A.Ref):
        return term.role in G.refs_for(binder)
    if isinstance(term, A.Scale):
        return _roles_ok(term.term, binder, G)
    if isinstance(term, A.Add):
        return len(term.terms) <= G.max_add_arity and all(
            _roles_ok(t, binder, G) for t in term.terms)
    if isinstance(term, A.Agg):
        return term.kind in G.agg_kinds and term.family_role in G.fams_for(binder)
    return False


def _has_measured(term: A.Term) -> bool:
    """True if the term references at least one column (i.e. is not a pure const)."""
    if isinstance(term, A.Const):
        return False
    if isinstance(term, A.Ref):
        return True
    if isinstance(term, A.Scale):
        return _has_measured(term.term)
    if isinstance(term, A.Add):
        return any(_has_measured(t) for t in term.terms)
    if isinstance(term, A.Agg):
        return True
    return False


def _base_family(role: str) -> str:
    """Strip a direction suffix so paired roles share a base measurement family
    (``egress``/``egress_rev`` -> ``egress``).  A meaningful anti-invariant is a directional
    asymmetry claim *within* one family; cross-family disequality is trivially separated and
    carries no information."""
    return role[:-4] if role.endswith("_rev") else role


def _leaf_set(term: A.Term) -> set:
    """Multiset-free set of measured leaves in a term (Ref roles and typed Agg)."""
    if isinstance(term, A.Ref):
        return {("r", term.role)}
    if isinstance(term, A.Agg):
        return {("a", term.kind, term.family_role)}
    if isinstance(term, A.Scale):
        return _leaf_set(term.term)
    if isinstance(term, A.Add):
        out: set = set()
        for t in term.terms:
            out |= _leaf_set(t)
        return out
    return set()


def _leaf_list(term: A.Term) -> list:
    """Measured leaves with multiplicity, used to reject algebraically reducible terms."""
    if isinstance(term, A.Ref):
        return [("r", term.role)]
    if isinstance(term, A.Agg):
        return [("a", term.kind, term.family_role)]
    if isinstance(term, A.Scale):
        return _leaf_list(term.term)
    if isinstance(term, A.Add):
        out = []
        for t in term.terms:
            out.extend(_leaf_list(t))
        return out
    return []


def _has_duplicate_leaf(term: A.Term) -> bool:
    leaves = _leaf_list(term)
    return len(leaves) != len(set(leaves))


def is_admissible(rule: A.Rule, G) -> tuple:
    """Return ``(ok, reason)``; ``ok`` is False with a short reason if inadmissible."""
    if rule.binder not in G.binders:
        return False, f"binder {rule.binder!r} not enabled"
    atom = rule.atom
    if atom.op not in G.ops:
        return False, f"op {atom.op!r} not enabled"
    if not _roles_ok(atom.left, rule.binder, G) or not _roles_ok(atom.right, rule.binder, G):
        return False, "role/family not valid for binder"
    # a comparison of a term with itself is non-informative (tautology for ==/~=/<=/>=,
    # contradiction for !=); reject structurally so search never spends budget on it.
    if atom.left == atom.right:
        return False, "trivial self-comparison"
    # self-referential / algebraically reducible: if the two sides share a measured leaf the
    # comparison reduces to a smaller (often trivial) claim (e.g. ``2v ~= v`` -> ``v ~= 0``,
    # ``v + w ~= v`` -> ``w ~= 0``).  A genuine relational invariant relates DISTINCT
    # measurements, so reject any overlap.  This is a pure-form check (no oracle).
    if _leaf_set(atom.left) & _leaf_set(atom.right):
        return False, "self-referential / reducible comparison"
    if _has_duplicate_leaf(atom.left) or _has_duplicate_leaf(atom.right):
        return False, "self-referential / reducible comparison"
    if rule.complexity() > G.max_complexity:
        return False, "exceeds max complexity"
    # dimensional: approximate/exact equality and disequality need both sides measured,
    # except a measured side compared to the additive identity 0.
    if atom.op in ("~=", "==", "!="):
        lm, rm = _has_measured(atom.left), _has_measured(atom.right)
        zero_r = isinstance(atom.right, A.Const) and atom.right.value == 0
        zero_l = isinstance(atom.left, A.Const) and atom.left.value == 0
        if not ((lm and rm) or (lm and zero_r) or (rm and zero_l)):
            return False, "dimensionally meaningless comparison"
        # A separation/anti-invariant asserts a *directional asymmetry* between two refs of
        # the SAME base measurement family; cross-family or aggregate disequality is trivially
        # separated, so restrict '!=' to same-family Ref-vs-Ref leaves.
        if atom.op == "!=":
            if not (isinstance(atom.left, A.Ref) and isinstance(atom.right, A.Ref)):
                return False, "separation (!=) only between atomic measured refs"
            if _base_family(atom.left.role) != _base_family(atom.right.role):
                return False, "separation (!=) only within one measurement family"
    return True, ""
