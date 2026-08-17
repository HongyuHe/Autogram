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
from .evaluate import typed_binary_domain, typed_group_key


def _roles_ok(term: A.Term, binder: str, G) -> bool:
    if isinstance(term, A.Const):
        return True
    if isinstance(term, A.Ref):
        return term.role in G.refs_for(binder)
    if isinstance(term, A.Scale):
        return _roles_ok(term.term, binder, G)
    if isinstance(term, A.Add):
        cap = (
            G.add_arity_cap(binder)
            if hasattr(G, "add_arity_cap")
            else getattr(G, "max_add_arity", len(term.terms))
        )
        return len(term.terms) <= cap and all(
            _roles_ok(t, binder, G) for t in term.terms)
    if isinstance(term, A.Agg):
        return (
            term.kind in A.AGG_KINDS
            and term.kind in G.agg_kinds
            and term.family_role in G.fams_for(binder)
        )
    if isinstance(term, A.Mul):
        return _roles_ok(term.left, binder, G) and _roles_ok(term.right, binder, G)
    if isinstance(term, A.Div):
        return _roles_ok(term.num, binder, G) and _roles_ok(term.den, binder, G)
    if isinstance(term, (A.Lag, A.Diff)):
        return (
            bool(getattr(G, "temporal_enabled", False))
            and 0 < int(term.steps) <= int(getattr(G, "max_lag", 0))
            and _roles_ok(term.term, binder, G)
        )
    if isinstance(term, A.Rolling):
        return (
            bool(getattr(G, "temporal_enabled", False))
            and term.kind in ("SUM", "MIN", "MAX", "AVG")
            and int(term.window) in tuple(getattr(G, "windows", ()))
            and _roles_ok(term.term, binder, G)
        )
    if isinstance(term, A.RelatedAgg):
        return term.role in G.related_for(binder)
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
    if isinstance(term, A.Mul):
        return _has_measured(term.left) or _has_measured(term.right)
    if isinstance(term, A.Div):
        return _has_measured(term.num) or _has_measured(term.den)
    if isinstance(term, (A.Lag, A.Diff, A.Rolling)):
        return _has_measured(term.term)
    if isinstance(term, A.RelatedAgg):
        return True
    return False


def _has_boolean_ref(term: A.Term, binder: str, G) -> bool:
    boolean_roles = set(G.booleans_for(binder))
    if isinstance(term, A.Ref):
        return term.role in boolean_roles
    if isinstance(term, A.RelatedAgg):
        return term.role in set(G.boolean_related_for(binder))
    if isinstance(term, (A.Scale, A.Lag, A.Diff, A.Rolling)):
        return _has_boolean_ref(term.term, binder, G)
    if isinstance(term, A.Add):
        return any(
            _has_boolean_ref(child, binder, G)
            for child in term.terms
        )
    if isinstance(term, A.Mul):
        return (
            _has_boolean_ref(term.left, binder, G)
            or _has_boolean_ref(term.right, binder, G)
        )
    if isinstance(term, A.Div):
        return (
            _has_boolean_ref(term.num, binder, G)
            or _has_boolean_ref(term.den, binder, G)
        )
    return False


def _is_boolean_atomic(term: A.Term, binder: str, G) -> bool:
    return (
        isinstance(term, A.Ref)
        and term.role in set(G.booleans_for(binder))
    ) or (
        isinstance(term, A.RelatedAgg)
        and term.role in set(G.boolean_related_for(binder))
    )


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
    if isinstance(term, A.Mul):
        return _leaf_set(term.left) | _leaf_set(term.right)
    if isinstance(term, A.Div):
        return _leaf_set(term.num) | _leaf_set(term.den)
    if isinstance(term, A.Lag):
        return {("t", "lag", term.steps, term.term.unparse())}
    if isinstance(term, A.Diff):
        return {("t", "diff", term.steps, term.term.unparse())}
    if isinstance(term, A.Rolling):
        return {("t", "rolling", term.kind, term.window, term.term.unparse())}
    if isinstance(term, A.RelatedAgg):
        return {("related", term.role)}
    return set()


def _predicate_leaf_set(predicate: A.Predicate) -> set:
    if isinstance(predicate, A.Bound):
        return _leaf_set(predicate.term)
    if isinstance(predicate, A.Sustained):
        return _predicate_leaf_set(predicate.predicate)
    if isinstance(predicate, A.Conjunction):
        return set().union(*(
            _predicate_leaf_set(item)
            for item in predicate.predicates
        ))
    return set()


def _excluded_roles(leaves, G) -> bool:
    roles = {
        leaf[1] if leaf[0] == "r" else leaf[-1]
        for leaf in leaves
    }
    return any(
        len(pair) == 2 and set(pair) <= roles
        for pair in getattr(G, "role_exclusions", ())
    )


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
    if isinstance(term, A.Mul):
        return _leaf_list(term.left) + _leaf_list(term.right)
    if isinstance(term, A.Div):
        return _leaf_list(term.num) + _leaf_list(term.den)
    if isinstance(term, A.Lag):
        return [("t", "lag", term.steps, term.term.unparse())]
    if isinstance(term, A.Diff):
        return [("t", "diff", term.steps, term.term.unparse())]
    if isinstance(term, A.Rolling):
        return [("t", "rolling", term.kind, term.window, term.term.unparse())]
    if isinstance(term, A.RelatedAgg):
        return [("related", term.role)]
    return []


def _has_duplicate_leaf(term: A.Term) -> bool:
    leaves = _leaf_list(term)
    return len(leaves) != len(set(leaves))


def is_admissible(rule: A.Rule, G) -> tuple:
    """Return ``(ok, reason)``; ``ok`` is False with a short reason if inadmissible."""
    if rule.binder not in G.binders:
        return False, f"binder {rule.binder!r} not enabled"
    atom = rule.atom
    if rule.condition is not None:
        condition = rule.condition
        if not getattr(G, "conditional_enabled", False):
            return False, "conditional rules are not enabled"
        if condition.op == "all":
            if not condition.values or any(
                not isinstance(value, A.Condition)
                or value.op == "all"
                for value in condition.values
            ):
                return False, "condition conjunction must contain simple conditions"
            max_terms = int(getattr(G, "max_condition_conjunction", 4))
            if not (2 <= len(condition.values) <= max_terms):
                return False, "condition conjunction arity is outside the configured bound"
            if len({value.column for value in condition.values}) != len(condition.values):
                return False, "condition conjunction must use distinct columns"
            for value in condition.values:
                probe = A.Rule(rule.binder, rule.atom, condition=value)
                ok, reason = is_admissible(probe, G)
                if not ok:
                    return False, reason
            condition = None
        if condition is None:
            pass
        else:
            allowed = getattr(G, "condition_columns", {})
            if condition.column not in allowed:
                return False, f"condition column {condition.column!r} is not enabled"
            if condition.op not in ("==", "in"):
                return False, f"condition operator {condition.op!r} is not enabled"
            if not condition.values:
                return False, "condition must contain at least one value"
            if len(condition.values) > int(getattr(G, "max_condition_values", 4)):
                return False, "condition exceeds the value cap"
            allowed_values = {
                typed_group_key(value)
                for value in allowed[condition.column]
            }
            condition_values = {
                typed_group_key(value)
                for value in condition.values
            }
            if not condition_values <= allowed_values:
                return False, "condition value is not observed for the declared column"
            if condition.op == "==" and len(condition.values) != 1:
                return False, "equality condition requires exactly one value"
            if condition.op == "in" and not (
                2 <= len(condition_values) < len(allowed_values)
            ):
                return False, "membership condition requires a proper multi-value subset"
    if isinstance(atom, A.BooleanDefinition):
        if not getattr(G, "advanced_enabled", False):
            return False, "advanced Boolean definitions are not enabled"
        if not isinstance(atom.target, A.Ref) or atom.target.role not in G.booleans_for(rule.binder):
            return False, "Boolean definition target must be a declared Boolean ref"
        if not _roles_ok(atom.target, rule.binder, G):
            return False, "Boolean definition target is not valid for binder"
        ok, reason = _predicate_admissible(atom.predicate, rule.binder, G)
        if not ok:
            return False, reason
        if not _definition_predicate_enumerable(atom.predicate):
            return False, "Boolean predicate is outside the bounded definition grammar"
        if _learned_bound_count(atom.predicate) > 2:
            return False, "Boolean definition has too many learned thresholds"
        if _excluded_roles(
            _leaf_set(atom.target)
            | _predicate_leaf_set(atom.predicate),
            G,
        ):
            return False, "excluded role co-occurrence"
        if rule.complexity() > G.complexity_cap(rule.binder):
            return False, "exceeds max complexity"
        return True, ""
    if isinstance(atom, A.CategoryDefinition):
        if not getattr(G, "advanced_enabled", False):
            return False, "categorical definitions are not enabled"
        columns = getattr(G, "condition_columns", {})
        if atom.target_column not in columns:
            return False, "categorical target column is not declared"
        target_values = {
            typed_group_key(value)
            for value in columns[atom.target_column]
        }
        if typed_group_key(atom.default) not in target_values:
            return False, "categorical default is not an observed target value"
        if not atom.cases:
            return False, "categorical definition needs at least one case"
        for column, value in atom.cases:
            if column not in columns or not typed_binary_domain(columns[column]):
                return False, "categorical cases must use declared Boolean columns"
            if typed_group_key(value) not in target_values:
                return False, "categorical case emits an unknown target value"
        if rule.complexity() > G.complexity_cap(rule.binder):
            return False, "exceeds max complexity"
        return True, ""
    if isinstance(atom, A.BandDefinition):
        if not getattr(G, "band_enabled", False):
            return False, "distribution bands are not enabled"
        if not _roles_ok(atom.term, rule.binder, G):
            return False, "band term is not valid for binder"
        if not _has_measured(atom.term):
            return False, "band requires a measured term"
        if _has_boolean_ref(atom.term, rule.binder, G):
            return False, "band requires a numeric term"
        if rule.complexity() > G.complexity_cap(rule.binder):
            return False, "exceeds max complexity"
        return True, ""
    if not isinstance(atom, A.Compare):
        return False, "unknown rule atom"
    if atom.op not in A.OPS:
        return False, f"unknown intrinsic op {atom.op!r}"
    if atom.op not in G.ops:
        return False, f"op {atom.op!r} not enabled"
    if not _roles_ok(atom.left, rule.binder, G) or not _roles_ok(atom.right, rule.binder, G):
        return False, "role/family not valid for binder"
    left_boolean = _has_boolean_ref(atom.left, rule.binder, G)
    right_boolean = _has_boolean_ref(atom.right, rule.binder, G)
    if left_boolean or right_boolean:
        if not (
            _is_boolean_atomic(atom.left, rule.binder, G)
            and _is_boolean_atomic(atom.right, rule.binder, G)
            and atom.op in ("==", "!=", "<|>")
        ):
            return False, "Boolean refs only support Boolean equality, separation, or presence"
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
    cap_c = G.complexity_cap(rule.binder) if hasattr(G, "complexity_cap") else getattr(G, "max_complexity", 12)
    if rule.complexity() > cap_c:
        return False, "exceeds max complexity"
    cap_d = G.degree_cap(rule.binder) if hasattr(G, "degree_cap") else getattr(G, "max_degree", 1)
    if max(atom.left.degree(), atom.right.degree()) > cap_d:
        return False, "exceeds max polynomial degree"
    for side in (atom.left, atom.right):
        if isinstance(side, A.Div) and not _has_measured(side.den):
            return False, "ratio denominator must be a measured term"
        if isinstance(side, A.Mul) and not (_has_measured(side.left) and _has_measured(side.right)):
            return False, "product operands must both be measured"
    if _excluded_roles(
        _leaf_set(atom.left) | _leaf_set(atom.right),
        G,
    ):
        return False, "excluded role co-occurrence"
    # dimensional: every comparison needs at least one measured side.  A bare constant may only
    # be the additive identity 0, which admits one-sided non-negativity / non-positivity laws.
    lm, rm = _has_measured(atom.left), _has_measured(atom.right)
    zero_r = isinstance(atom.right, A.Const) and atom.right.value == 0
    zero_l = isinstance(atom.left, A.Const) and atom.left.value == 0
    if not ((lm and rm) or (lm and zero_r) or (rm and zero_l)):
        return False, "dimensionally meaningless comparison"
    if atom.op in ("~=", "==", "!="):
        # A separation/anti-invariant asserts a *directional asymmetry* between two refs of
        # the SAME base measurement family; cross-family or aggregate disequality is trivially
        # separated, so restrict '!=' to same-family Ref-vs-Ref leaves.
        if atom.op == "!=":
            if not (isinstance(atom.left, A.Ref) and isinstance(atom.right, A.Ref)):
                return False, "separation (!=) only between atomic measured refs"
            if _base_family(atom.left.role) != _base_family(atom.right.role):
                return False, "separation (!=) only within one measurement family"
    if atom.op == "<|>":
        if not (isinstance(atom.left, A.Ref) and isinstance(atom.right, A.Ref)):
            return False, "existence pairing only between atomic measured refs"
    if atom.op == "~∝":
        if not (isinstance(atom.left, A.Ref) and isinstance(atom.right, A.Ref)):
            return False, "proportional equality only between atomic measured refs"
    return True, ""


def _predicate_admissible(predicate: A.Predicate, binder: str, G) -> tuple[bool, str]:
    if isinstance(predicate, A.Bound):
        if predicate.op not in ("<", "<=", ">", ">="):
            return False, f"unsupported predicate bound {predicate.op!r}"
        if not _roles_ok(predicate.term, binder, G):
            return False, "predicate term is not valid for binder"
        if _has_boolean_ref(predicate.term, binder, G):
            return False, "predicate bounds require numeric terms"
        return True, ""
    if isinstance(predicate, A.Sustained):
        if predicate.window not in tuple(getattr(G, "run_lengths", ())):
            return False, "sustained window is not enabled"
        return _predicate_admissible(predicate.predicate, binder, G)
    if isinstance(predicate, A.Conjunction):
        if not (2 <= len(predicate.predicates) <= int(getattr(G, "max_conjunction_terms", 3))):
            return False, "conjunction arity is outside the configured bound"
        for item in predicate.predicates:
            ok, reason = _predicate_admissible(item, binder, G)
            if not ok:
                return ok, reason
        return True, ""
    return False, "unknown Boolean predicate"


def _learned_bound_count(predicate: A.Predicate) -> int:
    if isinstance(predicate, A.Bound):
        return int(predicate.threshold is None)
    if isinstance(predicate, A.Sustained):
        return _learned_bound_count(predicate.predicate)
    if isinstance(predicate, A.Conjunction):
        return sum(
            _learned_bound_count(item)
            for item in predicate.predicates
        )
    return 0


def _enumerable_definition_operand(term) -> bool:
    """Operand terms allowed under a bounded Boolean-definition bound.

    A generic, bounded family parameterised over columns and windows -- NOT a fixed per-invariant
    shape: a bare column, a finite difference of a column, a rolling aggregate of a column, or a
    rolling aggregate of a pairwise column difference. Every alert-defining conjunction the plan
    targets is expressible as a conjunction of bounds over these operands, and the enumeration is
    kept finite by the grammar's window / arity / learned-threshold caps and the fail-loud max_rules.
    """
    if isinstance(term, A.Ref):
        return True
    if isinstance(term, A.Diff) and isinstance(term.term, A.Ref):
        return True
    if isinstance(term, A.Rolling):
        inner = term.term
        if isinstance(inner, A.Ref):
            return True
        if isinstance(inner, A.Add) and len(inner.terms) == 2:
            direct = [t for t in inner.terms if isinstance(t, A.Ref)]
            negated = [
                t.term for t in inner.terms
                if isinstance(t, A.Scale)
                and float(t.coeff) == -1.0
                and isinstance(t.term, A.Ref)
            ]
            if len(direct) == 1 and len(negated) == 1 and direct[0] != negated[0]:
                return True
    return False


def _enumerable_definition_bound(bound) -> bool:
    return (
        isinstance(bound, A.Bound)
        and _enumerable_definition_operand(bound.term)
        and bound.threshold in (None, 0.0)
    )


def _definition_predicate_enumerable(
    predicate: A.Predicate,
) -> bool:
    """Structural membership in the bounded Boolean-definition grammar.

    This is a GENERIC shape check, not a per-invariant template: a single bound (or a run-length
    sustained bound) over an admissible operand, or a conjunction whose every conjunct is such a
    bound. Arity, learned-threshold count (<= 2), role validity and complexity are enforced
    separately (``_predicate_admissible``, ``_learned_bound_count``, complexity cap), so a generic
    two-term ``(a < ?) AND (b > ?)`` and the multi-term trajectory alert are admitted uniformly --
    no conjunct's operator, position, or term is hard-coded to a specific known invariant.
    """
    if isinstance(predicate, A.Bound):
        return _enumerable_definition_bound(predicate)
    if isinstance(predicate, A.Sustained):
        return (
            isinstance(predicate.predicate.term, A.Ref)
            and predicate.predicate.threshold is None
        )
    if isinstance(predicate, A.Conjunction):
        return len(predicate.predicates) >= 2 and all(
            _enumerable_definition_bound(item)
            for item in predicate.predicates
        )
    return False
