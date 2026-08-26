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
import re
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Tuple, Union

from .scalar_codec import scalar_to_text

# Operator and aggregation vocabularies are intrinsic to the DSL (not dataset-specific).
OPS = ("~=", "==", "<=", ">=", "<", ">", "!=", "<|>", "~∝")
AGG_KINDS = ("SUM", "MIN", "MAX", "AVG")
_SIMPLE_IDENTIFIER = re.compile(r"\w+\Z")


def _scalar_unparse(value: object) -> str:
    return scalar_to_text(value, "DSL scalar value")


def _identifier_unparse(identifier: str) -> str:
    if not isinstance(identifier, str):
        raise TypeError(f"DSL identifiers must be strings; got {identifier!r}")
    if _SIMPLE_IDENTIFIER.fullmatch(identifier):
        return identifier
    return json.dumps(identifier, ensure_ascii=False, separators=(",", ":"))


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
        return _term_unparse(self)


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
        return _term_unparse(self)


@dataclass(frozen=True)
class Add:
    """N-ary sum of terms (the additivity shape)."""
    terms: Tuple["Term", ...]

    def complexity(self) -> int:
        return 1 + sum(t.complexity() for t in self.terms)

    def degree(self) -> int:
        return max((t.degree() for t in self.terms), default=0)

    def unparse(self) -> str:
        return _term_unparse(self)


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
        return _term_unparse(self)


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
        return _term_unparse(self)


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
        return _term_unparse(self)


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
        return _term_unparse(self)


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
        return _term_unparse(self)


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
        return _term_unparse(self)


@dataclass(frozen=True)
class RelatedAgg:
    """A declared aggregation over a related finer-grain frame."""

    role: str

    def complexity(self) -> int:
        return 2

    def degree(self) -> int:
        return 1

    def unparse(self) -> str:
        return _term_unparse(self)


Term = Union[Ref, Const, Scale, Add, Agg, Mul, Div, Lag, Diff, Rolling, RelatedAgg]


def _number_unparse(value: object) -> str:
    """Shortest decimal spelling that reconstructs the exact finite binary float."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("DSL numeric values must be finite")
    if number == 0.0 and math.copysign(1.0, number) < 0:
        return "-0.0"
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _group_add_child(term: Term, rendered: str) -> str:
    return f"({rendered})" if isinstance(term, Add) else rendered


def _term_unparse(term: Term) -> str:
    """Canonical precedence-aware surface rendering for one term."""
    if isinstance(term, Ref):
        return term.role
    if isinstance(term, Const):
        return _number_unparse(term.value)
    if isinstance(term, Scale):
        child = _group_add_child(term.term, _term_unparse(term.term))
        return f"{_number_unparse(term.coeff)}*{child}"
    if isinstance(term, Add):
        if len(term.terms) <= 1:
            children = ", ".join(_term_unparse(child) for child in term.terms)
            return f"ADD({children})"
        rendered = []
        for child in term.terms:
            text = _term_unparse(child)
            rendered.append(_group_add_child(child, text))
        return " + ".join(rendered)
    if isinstance(term, Agg):
        return f"{term.kind}({term.family_role})"
    if isinstance(term, Mul):
        left = _group_add_child(term.left, _term_unparse(term.left))
        right = _group_add_child(term.right, _term_unparse(term.right))
        return f"({left} * {right})"
    if isinstance(term, Div):
        num = _group_add_child(term.num, _term_unparse(term.num))
        den = _group_add_child(term.den, _term_unparse(term.den))
        return f"({num} / {den})"
    if isinstance(term, Lag):
        return f"LAG_{term.steps}({_term_unparse(term.term)})"
    if isinstance(term, Diff):
        return f"DELTA_{term.steps}({_term_unparse(term.term)})"
    if isinstance(term, Rolling):
        return (
            f"ROLL_{term.kind}_{term.window}"
            f"({_term_unparse(term.term)})"
        )
    if isinstance(term, RelatedAgg):
        return f"RELATED({term.role})"
    raise TypeError(f"unknown term {term!r}")


def _float_identity(value: object) -> Tuple[str, str]:
    """Exact, orderable identity for one DSL floating-point field."""
    return ("float", float(value).hex())


def term_identity(term: Term) -> Tuple:
    """Exact typed structural identity for a term.

    Machine-facing deduplication and opaque solver symbols use this tuple rather than reparsing
    surface text.
    """
    if isinstance(term, Ref):
        return ("ref", term.role)
    if isinstance(term, Const):
        return ("const", _float_identity(term.value))
    if isinstance(term, Scale):
        return (
            "scale",
            _float_identity(term.coeff),
            term_identity(term.term),
        )
    if isinstance(term, Add):
        return (
            "add",
            tuple(sorted(
                term_identity(child)
                for child in term.terms
            )),
        )
    if isinstance(term, Agg):
        return ("agg", term.kind, term.family_role)
    if isinstance(term, Mul):
        operands = tuple(sorted((
            term_identity(term.left),
            term_identity(term.right),
        )))
        return (
            "mul",
            operands,
        )
    if isinstance(term, Div):
        return (
            "div",
            term_identity(term.num),
            term_identity(term.den),
        )
    if isinstance(term, Lag):
        return ("lag", int(term.steps), term_identity(term.term))
    if isinstance(term, Diff):
        return ("diff", int(term.steps), term_identity(term.term))
    if isinstance(term, Rolling):
        return (
            "rolling",
            term.kind,
            int(term.window),
            term_identity(term.term),
        )
    if isinstance(term, RelatedAgg):
        return ("related", term.role)
    raise TypeError(f"unknown term {term!r}")


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
        threshold = "?" if self.threshold is None else _number_unparse(self.threshold)
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


def predicate_identity(predicate: Predicate) -> Tuple:
    """Exact typed structural identity for a Boolean-definition predicate."""
    if isinstance(predicate, Bound):
        threshold = (
            None
            if predicate.threshold is None
            else _float_identity(predicate.threshold)
        )
        return (
            "bound",
            term_identity(predicate.term),
            predicate.op,
            threshold,
        )
    if isinstance(predicate, Sustained):
        return (
            "sustained",
            int(predicate.window),
            predicate_identity(predicate.predicate),
        )
    if isinstance(predicate, Conjunction):
        return (
            "conjunction",
            tuple(
                predicate_identity(child)
                for child in predicate.predicates
            ),
        )
    raise TypeError(f"unknown predicate {predicate!r}")


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
            f"{_identifier_unparse(column)}->{_scalar_unparse(value)}"
            for column, value in self.cases
        )
        return (
            f"{_identifier_unparse(self.target_column)} := PRIORITY("
            f"{cases}; default={_scalar_unparse(self.default)})"
        )


class CategoryLabelBlock(NamedTuple):
    """One maximal same-label run in the semantics of a fixed priority rule.

    Cases inside the run can be permuted without changing that rule's total function.  This
    canonical equivalence does not by itself establish that every case has observationally fixed
    precedence relative to cases carrying other labels; discovery checks that at case level.
    """

    value_key: Tuple
    cases: Tuple[Tuple[object, object], ...]

    @property
    def value(self) -> object:
        return self.cases[0][1]


def category_definition_label_blocks(
    cases: Tuple[Tuple[object, object], ...],
    typed_value_key: Callable[[object], Tuple],
) -> Tuple[CategoryLabelBlock, ...]:
    """Ordered canonical blocks for the total function of a categorical priority map.

    Adjacent cases with the same typed output are functionally one unordered OR block. Repeated
    labels separated by another output remain separate blocks because their positions carry
    different precedence. Observational identifiability must still be established for every
    differently labelled *case* before discovery may accept the canonical block representation.
    """
    blocks: list[tuple[Tuple, list[tuple[object, object]]]] = []
    for case in cases:
        value_key = typed_value_key(case[1])
        if blocks and blocks[-1][0] == value_key:
            blocks[-1][1].append(case)
        else:
            blocks.append((value_key, [case]))
    return tuple(
        CategoryLabelBlock(value_key, tuple(block_cases))
        for value_key, block_cases in blocks
    )


def category_definition_cross_label_pairs(
    cases: Tuple[Tuple[object, object], ...],
    typed_value_key: Callable[[object], Tuple],
) -> Tuple[Tuple[int, int], ...]:
    """Case-index pairs whose relative priority can change the rule on an unseen input."""
    value_keys = tuple(
        typed_value_key(value)
        for _column, value in cases
    )
    return tuple(
        (left, right)
        for left in range(len(cases))
        for right in range(left + 1, len(cases))
        if value_keys[left] != value_keys[right]
    )


def category_definition_canonical_cases(
    cases: Tuple[Tuple[object, object], ...],
    typed_value_key: Callable[[object], Tuple],
) -> Tuple[Tuple[object, object], ...]:
    """Canonical case order that sorts only within same-label OR blocks."""
    return tuple(
        case
        for _value_key, block_cases in category_definition_label_blocks(
            cases,
            typed_value_key,
        )
        for case in sorted(
            block_cases,
            key=lambda item: repr(item[0]),
        )
    )


def category_definition_semantic_signature(
    target_column: object,
    cases: Tuple[Tuple[object, object], ...],
    default: object,
    typed_value_key: Callable[[object], Tuple],
) -> Tuple:
    """Typed recovery/solver identity for a categorical priority map.

    Consecutive cases with the same typed output form one OR block, so their Boolean columns are
    unordered. Block order is retained because precedence between different output labels remains
    semantically significant. The flattened result keeps the long-standing recovery-signature
    shape while putting every same-label block in one deterministic order.
    """
    return (
        "categorical_definition",
        (
            target_column,
            tuple(
                (column, typed_value_key(value))
                for column, value in category_definition_canonical_cases(
                    cases,
                    typed_value_key,
                )
            ),
            typed_value_key(default),
        ),
    )


def category_definition_semantic_key(
    atom: CategoryDefinition,
    typed_value_key: Callable[[object], Tuple],
) -> Tuple:
    return category_definition_semantic_signature(
        atom.target_column,
        atom.cases,
        atom.default,
        typed_value_key,
    )


@dataclass(frozen=True)
class BandDefinition:
    """A measured term concentrated around a learned or declared center."""

    term: Term
    center: Optional[float] = None

    def complexity(self) -> int:
        return 1 + self.term.complexity()

    def unparse(self) -> str:
        center = "?" if self.center is None else _number_unparse(self.center)
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
            self.atom,
            self.condition,
        ))


@dataclass(frozen=True)
class Condition:
    """A bounded row filter over one categorical or Boolean context column.

    Surface syntax keeps word-only column names bare and JSON-quotes all other identifiers.
    """

    column: str
    op: str
    values: Tuple[object, ...]

    def unparse(self) -> str:
        if self.op == "all":
            return "ALL(" + ", ".join(
                value.unparse() for value in self.values
                if isinstance(value, Condition)
            ) + ")"
        column = _identifier_unparse(self.column)
        if self.op == "in":
            return (
                f"{column} in ("
                f"{', '.join(_scalar_unparse(value) for value in self.values)})"
            )
        value = self.values[0] if self.values else ""
        return f"{column} {self.op} {_scalar_unparse(value)}"
