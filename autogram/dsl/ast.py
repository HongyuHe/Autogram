"""Typed, total DSL for soft data invariants.

The genotype the discovery search manipulates is a :class:`Rule`: a *quantified atom*
``forall b in Binder: left(b) <op> right(b)``.  Terms are a small, total, side-effect-free
algebra (field reference, constant, scalar multiply, n-ary add, family aggregation).
Everything is plain data -- no embedded Python code -- so a rule is serializable, statically
checkable, and trivially terminating.

The AST carries **no** dataset-specific vocabulary.  Binders, single-column *roles* and
family *roles* are not enumerated here; they are supplied by an *induced* schema
(:class:`autogram.schema.spec.SchemaSpec` -> :class:`autogram.dsl.grammar.Grammar`).  A
``Ref``/``Agg`` simply names a role string; the induced schema decides which role strings are
legal for which binder and how to ground them.  This is what lets the same AST describe
invariants on a dataset it was never tuned for.

Surface syntax (ASCII):

    ~=   approximate-equality within a fitted band
    ==   exact equality
    <=, >=, !=   ordering / disequality
    *    scalar multiply
    SUM/MIN/MAX/AVG(role)   family aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

# Operator and aggregation vocabularies are intrinsic to the DSL (not dataset-specific).
OPS = ("~=", "==", "<=", ">=", "!=")
AGG_KINDS = ("SUM", "MIN", "MAX", "AVG")


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """A single column selected from the current binding by ``role``."""
    role: str

    def complexity(self) -> int:
        return 1

    def unparse(self) -> str:
        return self.role


@dataclass(frozen=True)
class Const:
    value: float

    def complexity(self) -> int:
        return 1

    def unparse(self) -> str:
        v = self.value
        return str(int(v)) if float(v).is_integer() else f"{v:g}"


@dataclass(frozen=True)
class Scale:
    """Scalar multiply ``coeff * term`` (the only multiplication; keeps terms linear)."""
    coeff: float
    term: "Term"

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def unparse(self) -> str:
        c = self.coeff
        cs = str(int(c)) if float(c).is_integer() else f"{c:g}"
        return f"{cs}*{self.term.unparse()}"


@dataclass(frozen=True)
class Add:
    """N-ary sum of terms (the additivity shape)."""
    terms: Tuple["Term", ...]

    def complexity(self) -> int:
        return 1 + sum(t.complexity() for t in self.terms)

    def unparse(self) -> str:
        return " + ".join(t.unparse() for t in self.terms)


@dataclass(frozen=True)
class Agg:
    """Family aggregation ``KIND(role)`` over the columns a family role resolves to."""
    kind: str          # SUM | MIN | MAX | AVG
    family_role: str

    def complexity(self) -> int:
        return 2

    def unparse(self) -> str:
        return f"{self.kind}({self.family_role})"


Term = Union[Ref, Const, Scale, Add, Agg]


# ---------------------------------------------------------------------------
# Atom and Rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Compare:
    left: Term
    op: str
    right: Term

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def unparse(self) -> str:
        return f"{self.left.unparse()} {self.op} {self.right.unparse()}"


@dataclass(frozen=True)
class Rule:
    """A quantified atom: ``forall b in <binder>: <atom>``.

    ``tag`` is an optional human label used only in reports; it carries no information that
    affects evaluation.
    """
    binder: str
    atom: Compare
    tag: str = ""

    def complexity(self) -> int:
        return 1 + self.atom.complexity()

    def length(self) -> int:
        """Token length used as the parsimony axis of the Pareto archive."""
        return self.atom.complexity()

    def unparse(self) -> str:
        return f"[forall {self.binder}] {self.atom.unparse()}"

    def signature(self) -> str:
        """Structural identity ignoring the tag (used for dedup / archive keys)."""
        return f"{self.binder}::{self.atom.unparse()}"
