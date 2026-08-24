"""Z3-backed logical checks for bounded DSL rules."""

from __future__ import annotations

from typing import Dict, Tuple

import z3

from ..dsl import ast as A
from ..dsl.evaluate import typed_group_key


def _leaf_key(term: A.Term) -> Tuple:
    if isinstance(term, A.Ref):
        return ("ref", term.role)
    if isinstance(term, A.Agg):
        return ("agg", term.kind, term.family_role)
    raise TypeError(f"not a measured leaf: {term!r}")


def _var_name(key: Tuple) -> str:
    """Injectively encode a structural leaf key as a Z3 symbol name.

    Distinct keys must map to distinct symbol names, otherwise Z3 conflates two different measured
    leaves into one variable and reports unrelated rules as equivalent/subsuming. Each tuple
    component is encoded as the hex of its UTF-8 bytes and joined with ``_``; hex digits never
    include ``_``, so the join is unambiguous and the whole encoding is a total injection over
    arbitrary component strings (including control characters), not merely over a restricted
    alphabet.
    """
    return "x_" + "_".join(
        str(component).encode("utf-8").hex()
        for component in key
    )


def _category_definition_key(atom: A.CategoryDefinition) -> Tuple:
    key = [
        "category_definition",
        "target",
        atom.target_column,
        "case_count",
        len(atom.cases),
    ]
    for index, (column, value) in enumerate(atom.cases):
        key.extend((
            "case",
            index,
            column,
            typed_group_key(value),
        ))
    key.extend((
        "default",
        typed_group_key(atom.default),
    ))
    return tuple(key)


def _term_expr(term: A.Term, env: Dict[Tuple, z3.ArithRef]) -> z3.ArithRef:
    if isinstance(term, A.Const):
        return z3.RealVal(str(float(term.value)))
    if isinstance(term, (A.Ref, A.Agg)):
        key = _leaf_key(term)
        if key not in env:
            env[key] = z3.Real(_var_name(key))
        return env[key]
    if isinstance(term, A.Scale):
        return z3.RealVal(str(float(term.coeff))) * _term_expr(term.term, env)
    if isinstance(term, A.Add):
        if not term.terms:
            return z3.RealVal("0")
        return sum((_term_expr(t, env) for t in term.terms), z3.RealVal("0"))
    if isinstance(term, A.Mul):
        return _term_expr(term.left, env) * _term_expr(term.right, env)
    if isinstance(term, A.Div):
        # ratios are opaque fresh reals for the (sound but incomplete) screening
        key = ("div", term.unparse())
        if key not in env:
            env[key] = z3.Real(_var_name(key))
        return env[key]
    if isinstance(term, (A.Lag, A.Diff, A.Rolling)):
        key = ("temporal", term.unparse())
        if key not in env:
            env[key] = z3.Real(_var_name(key))
        return env[key]
    if isinstance(term, A.RelatedAgg):
        key = ("related", term.role)
        if key not in env:
            env[key] = z3.Real(_var_name(key))
        return env[key]
    raise TypeError(f"unknown term {term!r}")


def atom_expr(atom, env: Dict[Tuple, z3.ArithRef] | None = None) -> z3.BoolRef:
    env = env if env is not None else {}
    if isinstance(atom, A.BooleanDefinition):
        target = _term_expr(atom.target, env) != 0
        return target == _predicate_expr(atom.predicate, env)
    if isinstance(atom, A.CategoryDefinition):
        key = _category_definition_key(atom)
        if key not in env:
            env[key] = z3.Bool(_var_name(key))
        return env[key]
    if isinstance(atom, A.BandDefinition):
        key = ("band_definition", atom.unparse())
        if key not in env:
            env[key] = z3.Bool(_var_name(key))
        return env[key]
    if not isinstance(atom, A.Compare):
        raise TypeError(f"unknown atom {atom!r}")
    left = _term_expr(atom.left, env)
    right = _term_expr(atom.right, env)
    if atom.op == "==":
        return left == right
    if atom.op == "~=":
        if atom.left == atom.right:
            return z3.BoolVal(True)
        operands = tuple(sorted((
            atom.left.unparse(),
            atom.right.unparse(),
        )))
        key = ("approximate_equality", *operands)
        if key not in env:
            env[key] = z3.Bool(_var_name(key))
        return env[key]
    if atom.op == "<=":
        return left <= right
    if atom.op == ">=":
        return left >= right
    if atom.op == "<":
        return left < right
    if atom.op == ">":
        return left > right
    if atom.op == "!=":
        return left != right
    if atom.op == "<|>":
        return (left != 0) == (right != 0)
    if atom.op == "~∝":
        key = ("proportional_coefficient", atom.left.unparse(), atom.right.unparse())
        if key not in env:
            env[key] = z3.Real(_var_name(key))
        return left == env[key] * right
    raise ValueError(f"unknown op {atom.op!r}")


def _predicate_expr(predicate: A.Predicate, env) -> z3.BoolRef:
    if isinstance(predicate, A.Bound):
        term = _term_expr(predicate.term, env)
        if predicate.threshold is None:
            key = ("learned_threshold", predicate.unparse())
            if key not in env:
                env[key] = z3.Real(_var_name(key))
            threshold = env[key]
        else:
            threshold = z3.RealVal(str(float(predicate.threshold)))
        if predicate.op == "<":
            return term < threshold
        if predicate.op == "<=":
            return term <= threshold
        if predicate.op == ">":
            return term > threshold
        if predicate.op == ">=":
            return term >= threshold
    if isinstance(predicate, A.Sustained):
        key = ("sustained", predicate.unparse())
        if key not in env:
            env[key] = z3.Bool(_var_name(key))
        return env[key]
    if isinstance(predicate, A.Conjunction):
        return z3.And(*(_predicate_expr(item, env) for item in predicate.predicates))
    raise TypeError(f"unknown predicate {predicate!r}")


def _leaves(term: A.Term) -> set[Tuple]:
    if isinstance(term, A.Ref):
        return {("ref", term.role)}
    if isinstance(term, A.Agg):
        return {("agg", term.kind, term.family_role)}
    if isinstance(term, A.Scale):
        return _leaves(term.term)
    if isinstance(term, A.Add):
        out: set[Tuple] = set()
        for t in term.terms:
            out |= _leaves(t)
        return out
    if isinstance(term, (A.Mul, A.Div)):
        return _leaves(term.left) | _leaves(term.right) if isinstance(term, A.Mul) else (
            _leaves(term.num) | _leaves(term.den)
        )
    if isinstance(term, (A.Lag, A.Diff, A.Rolling)):
        return {("temporal", term.unparse())}
    if isinstance(term, A.RelatedAgg):
        return {("related", term.role)}
    return set()


def _valid(expr: z3.BoolRef) -> bool:
    solver = z3.Solver()
    solver.add(z3.Not(expr))
    return solver.check() == z3.unsat


def _condition_expr(condition: A.Condition, env: Dict[Tuple, z3.ArithRef]) -> z3.BoolRef:
    from ..dsl.evaluate import typed_condition_key

    key = ("condition", typed_condition_key(condition))
    if key not in env:
        env[key] = z3.Bool(_var_name(key))
    return env[key]


def _rule_expr(rule: A.Rule, env) -> z3.BoolRef:
    atom = atom_expr(rule.atom, env)
    if rule.condition is None:
        return atom
    return z3.Implies(_condition_expr(rule.condition, env), atom)


def is_tautology(rule: A.Rule) -> bool:
    """True when the rule is valid for all assignments of its measured leaves."""
    if not isinstance(rule.atom, A.Compare):
        return _valid(_rule_expr(rule, {}))
    if rule.atom.left == rule.atom.right and rule.atom.op in ("~=", "==", "<=", ">="):
        return True
    if _leaves(rule.atom.left).isdisjoint(_leaves(rule.atom.right)):
        return False
    return _valid(_rule_expr(rule, {}))


def is_contradiction(rule: A.Rule) -> bool:
    """True when the rule can never hold for any assignment of its measured leaves."""
    if rule.condition is not None:
        return False
    if not isinstance(rule.atom, A.Compare):
        return _valid(z3.Not(_rule_expr(rule, {})))
    if rule.atom.left == rule.atom.right and rule.atom.op == "!=":
        return True
    if _leaves(rule.atom.left).isdisjoint(_leaves(rule.atom.right)):
        return False
    return _valid(z3.Not(atom_expr(rule.atom, {})))


def is_trivial(rule: A.Rule) -> bool:
    """Logical triviality includes tautologies and contradictions."""
    return is_tautology(rule) or is_contradiction(rule)


def _legacy_approximate_as_exact(rule: A.Rule) -> A.Rule:
    if not isinstance(rule.atom, A.Compare) or rule.atom.op != "~=":
        return rule
    return A.Rule(
        rule.binder,
        A.Compare(rule.atom.left, "==", rule.atom.right),
        tag=rule.tag,
        condition=rule.condition,
    )


def legacy_is_trivial(rule: A.Rule) -> bool:
    return is_trivial(_legacy_approximate_as_exact(rule))


def equivalent(a: A.Rule, b: A.Rule) -> bool:
    if a.binder != b.binder:
        return False
    env: Dict[Tuple, z3.ArithRef] = {}
    ea = _rule_expr(a, env)
    eb = _rule_expr(b, env)
    return _valid(ea == eb)


def legacy_equivalent(a: A.Rule, b: A.Rule) -> bool:
    return equivalent(
        _legacy_approximate_as_exact(a),
        _legacy_approximate_as_exact(b),
    )


def subsumes(a: A.Rule, b: A.Rule) -> bool:
    """True when every assignment satisfying ``a`` also satisfies ``b``."""
    if a.binder != b.binder:
        return False
    env: Dict[Tuple, z3.ArithRef] = {}
    ea = _rule_expr(a, env)
    eb = _rule_expr(b, env)
    return _valid(z3.Implies(ea, eb))


def legacy_subsumes(a: A.Rule, b: A.Rule) -> bool:
    return subsumes(
        _legacy_approximate_as_exact(a),
        _legacy_approximate_as_exact(b),
    )
