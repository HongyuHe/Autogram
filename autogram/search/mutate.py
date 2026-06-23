"""Typed AST mutations and random rule generation (design Sec. 10.4, "vary").

Every operator returns an *admissible* :class:`~autogram.dsl.ast.Rule` (re-checked with
``typecheck.is_admissible``) or ``None`` when no admissible result was found within a few
attempts.  Because the genotype is a typed, total DSL AST -- not raw code -- mutation can
only ever produce another well-formed, terminating invariant: there is no way to mutate
into an unbounded loop or a side effect (Sec. 6.5).
"""

from __future__ import annotations

import random
from typing import List, Optional

from ..dsl import ast as A
from ..dsl.ast import Add, Agg, Compare, Const, Ref, Rule, Scale
from ..dsl.grammar import Grammar
from ..dsl.typecheck import is_admissible

_SCALARS = (0.5, 2.0, -1.0)


# ---------------------------------------------------------------------------
# Random generation (random restarts + family seeds for the islands)
# ---------------------------------------------------------------------------

def _ref_roles(binder: str, G: Grammar) -> List[str]:
    return [r for r in A.REF_ROLES.get(binder, ()) if r in G.ref_roles]


def _fam_roles(binder: str, G: Grammar) -> List[str]:
    return [r for r in A.FAM_ROLES.get(binder, ()) if r in G.fam_roles]


def random_term(binder: str, G: Grammar, rng: random.Random, depth: int = 2) -> Optional[A.Term]:
    """Build a random admissible term for ``binder`` within ``depth`` levels."""
    refs = _ref_roles(binder, G)
    fams = _fam_roles(binder, G)
    leaves: List[A.Term] = [Ref(r) for r in refs]
    leaves += [Agg(k, f) for f in fams for k in G.agg_kinds]
    if not leaves:
        return None
    if depth <= 0 or rng.random() < 0.55:
        return rng.choice(leaves)
    choice = rng.random()
    if choice < 0.45:                                   # n-ary add
        arity = rng.randint(2, max(2, G.max_add_arity))
        parts = []
        for _ in range(arity):
            t = random_term(binder, G, rng, depth - 1)
            if t is None:
                return None
            parts.append(t)
        return Add(tuple(parts))
    if choice < 0.7:                                    # scalar multiply
        inner = random_term(binder, G, rng, depth - 1)
        return None if inner is None else Scale(rng.choice(_SCALARS), inner)
    return rng.choice(leaves)


def random_rule(G: Grammar, rng: random.Random, binder: Optional[str] = None,
                tries: int = 24) -> Optional[Rule]:
    """Draw a random admissible rule (optionally pinned to ``binder``)."""
    for _ in range(tries):
        b = binder or rng.choice(G.binders)
        op = rng.choice(G.ops)
        left = random_term(b, G, rng)
        # right side: 0 for ordering ops, otherwise another term
        if op in (">=", "<=") and rng.random() < 0.6:
            right: Optional[A.Term] = Const(0)
        else:
            right = random_term(b, G, rng)
        if left is None or right is None:
            continue
        r = Rule(b, Compare(left, op, right), tag="rand")
        ok, _ = is_admissible(r, G)
        if ok:
            return r
    return None


# ---------------------------------------------------------------------------
# Local mutations
# ---------------------------------------------------------------------------

def _subterms(term: A.Term):
    """Yield ``(term, rebuild)`` pairs so a mutation can replace one node in place."""
    yield term, lambda new: new
    if isinstance(term, Scale):
        for sub, rb in _subterms(term.term):
            yield sub, (lambda new, rb=rb: Scale(term.coeff, rb(new)))
    elif isinstance(term, Add):
        for i, t in enumerate(term.terms):
            for sub, rb in _subterms(t):
                def make(new, i=i, rb=rb):
                    items = list(term.terms)
                    items[i] = rb(new)
                    return Add(tuple(items))
                yield sub, make


def _mutate_term(term: A.Term, binder: str, G: Grammar, rng: random.Random) -> A.Term:
    refs = _ref_roles(binder, G)
    fams = _fam_roles(binder, G)
    if isinstance(term, Ref) and refs:
        return Ref(rng.choice(refs))
    if isinstance(term, Agg):
        roll = rng.random()
        if roll < 0.5:
            return Agg(rng.choice(G.agg_kinds), term.family_role)
        if fams:
            return Agg(term.kind, rng.choice(fams))
        return term
    if isinstance(term, Scale):
        if rng.random() < 0.5:
            return Scale(rng.choice(_SCALARS), term.term)
        return term.term                                # unwrap
    if isinstance(term, Add):
        items = list(term.terms)
        if len(items) > 2 and rng.random() < 0.5:
            items.pop(rng.randrange(len(items)))        # drop a summand
            return Add(tuple(items)) if len(items) > 1 else items[0]
        if len(items) < G.max_add_arity:
            extra = random_term(binder, G, rng, depth=1)
            if extra is not None:
                items.append(extra)
                return Add(tuple(items))
    return term


def _replace_side(rule: Rule, side: str, new: A.Term) -> Rule:
    a = rule.atom
    atom = Compare(new, a.op, a.right) if side == "left" else Compare(a.left, a.op, new)
    return Rule(rule.binder, atom, tag=rule.tag)


def mutate_rule(rule: Rule, G: Grammar, rng: random.Random, tries: int = 12) -> Optional[Rule]:
    """Return a mutated admissible neighbour of ``rule`` (or ``None``)."""
    for _ in range(tries):
        kind = rng.random()
        if kind < 0.2:                                  # swap operator
            atom = Compare(rule.atom.left, rng.choice(G.ops), rule.atom.right)
            cand = Rule(rule.binder, atom, tag=rule.tag)
        elif kind < 0.85:                               # mutate one side's subterm
            side = "left" if rng.random() < 0.5 else "right"
            term = rule.atom.left if side == "left" else rule.atom.right
            subs = list(_subterms(term))
            target, rebuild = rng.choice(subs)
            cand = _replace_side(rule, side, rebuild(_mutate_term(target, rule.binder, G, rng)))
        else:                                           # wrap a side in a fresh Add
            side = "left" if rng.random() < 0.5 else "right"
            term = rule.atom.left if side == "left" else rule.atom.right
            extra = random_term(rule.binder, G, rng, depth=1)
            if extra is None:
                continue
            cand = _replace_side(rule, side, Add((term, extra)))
        cand = Rule(cand.binder, cand.atom, tag="mut")
        if cand.signature() == rule.signature():
            continue
        ok, _ = is_admissible(cand, G)
        if ok:
            return cand
    return None
