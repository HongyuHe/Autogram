"""Z3-backed logical checks for bounded DSL rules."""

from __future__ import annotations

import re
from typing import Dict, Tuple

import z3

from ..dsl import ast as A


def _leaf_key(term: A.Term) -> Tuple:
    if isinstance(term, A.Ref):
        return ("ref", term.role)
    if isinstance(term, A.Agg):
        return ("agg", term.kind, term.family_role)
    raise TypeError(f"not a measured leaf: {term!r}")


def _var_name(key: Tuple) -> str:
    raw = "_".join(str(x) for x in key)
    return "x_" + re.sub(r"[^0-9A-Za-z_]", "_", raw)


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
    raise TypeError(f"unknown term {term!r}")


def atom_expr(atom: A.Compare, env: Dict[Tuple, z3.ArithRef] | None = None) -> z3.BoolRef:
    env = env if env is not None else {}
    left = _term_expr(atom.left, env)
    right = _term_expr(atom.right, env)
    if atom.op in ("~=", "=="):
        return left == right
    if atom.op == "<=":
        return left <= right
    if atom.op == ">=":
        return left >= right
    if atom.op == "!=":
        return left != right
    if atom.op == "<|>":
        return (left != 0) == (right != 0)
    raise ValueError(f"unknown op {atom.op!r}")


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
    return set()


def _valid(expr: z3.BoolRef) -> bool:
    solver = z3.Solver()
    solver.add(z3.Not(expr))
    return solver.check() == z3.unsat


def is_tautology(rule: A.Rule) -> bool:
    """True when the rule is valid for all assignments of its measured leaves."""
    if rule.atom.left == rule.atom.right and rule.atom.op in ("~=", "==", "<=", ">="):
        return True
    if _leaves(rule.atom.left).isdisjoint(_leaves(rule.atom.right)):
        return False
    return _valid(atom_expr(rule.atom, {}))


def is_contradiction(rule: A.Rule) -> bool:
    """True when the rule can never hold for any assignment of its measured leaves."""
    if rule.atom.left == rule.atom.right and rule.atom.op == "!=":
        return True
    if _leaves(rule.atom.left).isdisjoint(_leaves(rule.atom.right)):
        return False
    return _valid(z3.Not(atom_expr(rule.atom, {})))


def is_trivial(rule: A.Rule) -> bool:
    """Logical triviality includes tautologies and contradictions."""
    return is_tautology(rule) or is_contradiction(rule)


def equivalent(a: A.Rule, b: A.Rule) -> bool:
    if a.binder != b.binder:
        return False
    env: Dict[Tuple, z3.ArithRef] = {}
    ea = atom_expr(a.atom, env)
    eb = atom_expr(b.atom, env)
    return _valid(ea == eb)


def subsumes(a: A.Rule, b: A.Rule) -> bool:
    """True when every assignment satisfying ``a`` also satisfies ``b``."""
    if a.binder != b.binder:
        return False
    env: Dict[Tuple, z3.ArithRef] = {}
    ea = atom_expr(a.atom, env)
    eb = atom_expr(b.atom, env)
    return _valid(z3.Implies(ea, eb))
