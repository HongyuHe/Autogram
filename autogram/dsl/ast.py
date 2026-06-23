"""Typed, total DSL for soft network invariants (design Sec. 6.6).

The genotype the evolutionary search manipulates is a :class:`Rule`: a *quantified
atom* ``forall b in Binder: left(b) <op> right(b)``.  Terms are a small, total,
side-effect-free algebra (field reference, constant, scalar multiply, n-ary add,
locality-family aggregation).  Everything is plain data -- no embedded Python code --
so a rule is serializable, statically checkable, and trivially terminating (Sec. 6.5).

Surface syntax (ASCII rendering of the Unicode grammar in the doc):

    ~=   approximate-equality within a fitted band   (doc: the wavy approx symbol)
    ==   exact equality
    <=, >=, !=   ordering / disequality
    *    scalar multiply                              (doc: middle dot)
    SUM/MIN/MAX/AVG(Fam(token,type,dir))             locality-grouped aggregation

A *binder* quantifies the atom over a locality family of bindings (cells, directed
links, nodes, or the whole network); grounding (in ``evaluate.py``) turns each binding
into concrete column references via *role* strings, keeping the AST independent of any
particular dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union


# ---------------------------------------------------------------------------
# Binders: how an atom is quantified over locality
# ---------------------------------------------------------------------------

#: binder kinds and the role vocabulary each one exposes to ``Ref``/``Agg``.
BINDERS = ("cell", "link", "node", "network")

#: single-column roles, per binder kind
REF_ROLES = {
    "cell": ("self",),
    "link": ("egress", "ingress_rev", "egress_rev", "ingress",
             "demand", "demand_rev"),
    "node": ("origination", "termination", "demand_self"),
    "network": (),
}

#: family (aggregation) roles, per binder kind
FAM_ROLES = {
    "cell": (),
    "link": (),
    "node": ("demand_row", "demand_col", "ingress_fam", "egress_fam"),
    "network": ("all_orig", "all_term", "all_demand"),
}

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
        return _ROLE_GLYPH.get(self.role, self.role)


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
    """N-ary sum of terms (doc Sec. 4.1: the missing n-ary additivity shape)."""
    terms: Tuple["Term", ...]

    def complexity(self) -> int:
        return 1 + sum(t.complexity() for t in self.terms)

    def unparse(self) -> str:
        return " + ".join(t.unparse() for t in self.terms)


@dataclass(frozen=True)
class Agg:
    """Locality-grouped aggregation ``KIND(Fam(role))`` (doc Sec. 4.2)."""
    kind: str          # SUM | MIN | MAX | AVG
    family_role: str

    def complexity(self) -> int:
        return 2

    def unparse(self) -> str:
        return f"{self.kind}({_FAM_GLYPH.get(self.family_role, self.family_role)})"


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

    ``tag`` is an optional human label (e.g. ``"I5-like"``) used only in reports;
    it carries no oracle information and does not affect evaluation.
    """
    binder: str
    atom: Compare
    tag: str = ""

    def complexity(self) -> int:
        return 1 + self.atom.complexity()

    def unparse(self) -> str:
        return f"[forall {self.binder}] {self.atom.unparse()}"

    def signature(self) -> str:
        """Structural identity ignoring the tag (used for dedup / archive keys)."""
        return f"{self.binder}::{self.atom.unparse()}"


# ---------------------------------------------------------------------------
# Pretty-print glyphs (ASCII, console-safe)
# ---------------------------------------------------------------------------

_ROLE_GLYPH = {
    "self": "v",
    "egress": "e[X->Y]",
    "ingress_rev": "i[Y<-X]",
    "egress_rev": "e[Y->X]",
    "ingress": "i[X<-Y]",
    "demand": "H[X,Y]",
    "demand_rev": "H[Y,X]",
    "origination": "o(X)",
    "termination": "t(X)",
    "demand_self": "H[X,X]",
}

_FAM_GLYPH = {
    "demand_row": "H[X,*]",
    "demand_col": "H[*,X]",
    "ingress_fam": "i[X<-*]",
    "egress_fam": "e[X->*]",
    "all_orig": "o(*)",
    "all_term": "t(*)",
    "all_demand": "H[*,*]",
}
