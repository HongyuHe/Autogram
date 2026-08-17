"""Typed, total DSL for soft data invariants.

The genotype the discovery search manipulates is a :class:`Rule`: a *quantified atom*
``forall b in Binder: atom(b)``. Terms are a small, total, side-effect-free algebra over field references, constants, bounded arithmetic, family/related aggregation, and grouped temporal operators; atoms also include bounded Boolean and categorical definitions.
Everything is plain data -- no embedded Python code -- so a rule is serializable, statically
checkable, and trivially terminating.

The AST carries **no** dataset-specific vocabulary.  Binders, single-column *roles* and
family *roles* are not enumerated here; they are supplied by an *induced* schema
(:class:`autogram.schema.spec.GrammarSpec` -> :class:`autogram.dsl.grammar.Grammar`).  A
``Ref``/``Agg`` simply names a role string; the induced schema decides which role strings are
legal for which binder and how to ground them.  This is what lets the same AST describe
invariants on a dataset it was never tuned for.

Surface syntax (ASCII):

    ~=   approximate-equality within a fitted band
    ==   exact equality
    <=, >=, !=   ordering / disequality
    <|>    bidirectional structural presence
    *    scalar multiply
    SUM/MIN/MAX/AVG(role)   family aggregation
    LAG_k(term), DELTA_k(term), ROLL_SUM_k(term)   grouped temporal terms
    RELATED(role)   declared cross-grain aggregation
    target := ALWAYS_k(bound) | conjunction | categorical priority map
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

# Operator and aggregation vocabularies are intrinsic to the DSL (not dataset-specific).
OPS = ("~=", "==", "<=", ">=", "<", ">", "!=", "<|>", "~∝")
AGG_KINDS = ("SUM", "MIN", "MAX", "AVG")


def _scalar_unparse(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        scalar = value
    elif isinstance(value, numbers.Integral):
        scalar = int(value)
    elif isinstance(value, numbers.Real):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError("DSL scalar values must be finite")
    else:
        raise TypeError(
            f"DSL scalar values must be strings, booleans, numbers, or null; got {value!r}"
        )
    return json.dumps(
        scalar,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """A single column selected from the current binding by ``role``."""
    role: str

    def complexity(self) -> int:
        return 1

    def degree(self) -> int:
        return 1

    def unparse(self) -> str:
        return self.role


@dataclass(frozen=True)
class Const:
    value: float

    def complexity(self) -> int:
        return 1

    def degree(self) -> int:
        return 0

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

    def degree(self) -> int:
        return self.term.degree()

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

    def degree(self) -> int:
        return max((t.degree() for t in self.terms), default=0)

    def unparse(self) -> str:
        return " + ".join(t.unparse() for t in self.terms)


@dataclass(frozen=True)
class Agg:
    """Family aggregation ``KIND(role)`` over the columns a family role resolves to."""
    kind: str          # SUM | MIN | MAX | AVG
    family_role: str

    def complexity(self) -> int:
        return 2

    def degree(self) -> int:
        return 1

    def unparse(self) -> str:
        return f"{self.kind}({self.family_role})"


@dataclass(frozen=True)
class Mul:
    """Product of two measured terms ``left * right`` (nonlinear; raises the polynomial degree)."""
    left: "Term"
    right: "Term"

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def degree(self) -> int:
        return self.left.degree() + self.right.degree()

    def unparse(self) -> str:
        return f"({self.left.unparse()} * {self.right.unparse()})"


@dataclass(frozen=True)
class Div:
    """Ratio ``num / den`` (den must be non-zero; nonlinear).

    Points where ``den == 0`` ground to NaN and are dropped from the residual population.
    """
    num: "Term"
    den: "Term"

    def complexity(self) -> int:
        return 1 + self.num.complexity() + self.den.complexity()

    def degree(self) -> int:
        return self.num.degree() + self.den.degree()

    def unparse(self) -> str:
        return f"({self.num.unparse()} / {self.den.unparse()})"


@dataclass(frozen=True)
class Lag:
    """Backshift ``B^steps term`` within each ordered group."""

    term: "Term"
    steps: int = 1

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def degree(self) -> int:
        return self.term.degree()

    def unparse(self) -> str:
        return f"LAG_{self.steps}({self.term.unparse()})"


@dataclass(frozen=True)
class Diff:
    """Finite difference ``term[t] - term[t-steps]`` within an ordered group."""

    term: "Term"
    steps: int = 1

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def degree(self) -> int:
        return self.term.degree()

    def unparse(self) -> str:
        return f"DELTA_{self.steps}({self.term.unparse()})"


@dataclass(frozen=True)
class Rolling:
    """Trailing full-window aggregation within each ordered group."""

    term: "Term"
    window: int
    kind: str = "SUM"

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def degree(self) -> int:
        return self.term.degree()

    def unparse(self) -> str:
        return f"ROLL_{self.kind}_{self.window}({self.term.unparse()})"


@dataclass(frozen=True)
class RelatedAgg:
    """A declared aggregation over a related finer-grain frame."""

    role: str

    def complexity(self) -> int:
        return 2

    def degree(self) -> int:
        return 1

    def unparse(self) -> str:
        return f"RELATED({self.role})"


Term = Union[Ref, Const, Scale, Add, Agg, Mul, Div, Lag, Diff, Rolling, RelatedAgg]


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
class Bound:
    """A numeric bound used inside a Boolean definition."""

    term: Term
    op: str
    threshold: Optional[float] = None

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def unparse(self) -> str:
        threshold = "?" if self.threshold is None else Const(float(self.threshold)).unparse()
        return f"{self.term.unparse()} {self.op} {threshold}"


@dataclass(frozen=True)
class Sustained:
    """A predicate that must hold for a full trailing window."""

    predicate: Bound
    window: int

    def complexity(self) -> int:
        return 1 + self.predicate.complexity()

    def unparse(self) -> str:
        return f"ALWAYS_{self.window}({self.predicate.unparse()})"


@dataclass(frozen=True)
class Conjunction:
    """A bounded conjunction of Boolean predicates."""

    predicates: Tuple[Union[Bound, Sustained], ...]

    def complexity(self) -> int:
        return 1 + sum(predicate.complexity() for predicate in self.predicates)

    def unparse(self) -> str:
        return " AND ".join(f"({predicate.unparse()})" for predicate in self.predicates)


Predicate = Union[Bound, Sustained, Conjunction]


@dataclass(frozen=True)
class BooleanDefinition:
    """A Boolean measured target defined by a total predicate."""

    target: Term
    predicate: Predicate

    def complexity(self) -> int:
        return 1 + self.target.complexity() + self.predicate.complexity()

    def unparse(self) -> str:
        return f"{self.target.unparse()} := {self.predicate.unparse()}"


@dataclass(frozen=True)
class CategoryDefinition:
    """A priority map from Boolean context columns to one categorical target."""

    target_column: str
    cases: Tuple[Tuple[str, object], ...]
    default: object

    def complexity(self) -> int:
        return 2 + len(self.cases)

    def unparse(self) -> str:
        cases = ", ".join(
            f"{column}->{_scalar_unparse(value)}"
            for column, value in self.cases
        )
        return (
            f"{self.target_column} := PRIORITY("
            f"{cases}; default={_scalar_unparse(self.default)})"
        )


@dataclass(frozen=True)
class BandDefinition:
    """A measured term concentrated around a learned or declared center."""

    term: Term
    center: Optional[float] = None

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def unparse(self) -> str:
        center = "?" if self.center is None else Const(float(self.center)).unparse()
        return f"{self.term.unparse()} ~band {center}"


@dataclass(frozen=True)
class Rule:
    """A quantified atom: ``forall b in <binder>: <atom>``.

    ``tag`` is an optional human label used only in reports; it carries no information that
    affects evaluation.
    """
    binder: str
    atom: Union[Compare, BooleanDefinition, CategoryDefinition, BandDefinition]
    tag: str = ""
    condition: Optional["Condition"] = None

    def complexity(self) -> int:
        return 1 + self.atom.complexity() + (1 if self.condition is not None else 0)

    def length(self) -> int:
        """Token length used as the parsimony axis of the Pareto archive."""
        return self.atom.complexity()

    def unparse(self) -> str:
        suffix = f" where {self.condition.unparse()}" if self.condition is not None else ""
        return f"[forall {self.binder}] {self.atom.unparse()}{suffix}"

    def signature(self) -> str:
        """Structural identity ignoring the tag (used for dedup / archive keys)."""
        return repr((
            self.binder,
            self.atom.unparse(),
            self.condition,
        ))


@dataclass(frozen=True)
class Condition:
    """A bounded row filter over one categorical or Boolean context column."""

    column: str
    op: str
    values: Tuple[object, ...]

    def unparse(self) -> str:
        if self.op == "all":
            return "ALL(" + ", ".join(
                value.unparse() for value in self.values
                if isinstance(value, Condition)
            ) + ")"
        if self.op == "in":
            return (
                f"{self.column} in ("
                f"{', '.join(_scalar_unparse(value) for value in self.values)})"
            )
        value = self.values[0] if self.values else ""
        return f"{self.column} {self.op} {_scalar_unparse(value)}"
