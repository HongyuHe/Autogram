"""Declarative schema-induction layer (design Sec. 3.1, 6.4; poc-eval rec. 9 / E6).

The base PoC is specialised to the CrossCheck naming convention through four hardcoded
seams: the name parser (``loader/names.py``), the role ontology (``dsl/ast.py``), the
grounding templates (``dsl/binders.py``) and the cell codec (``loader/loader.py``).  A
proposer that only emits invariant *forms* (role strings) can never introduce a new column
convention, role or grounding rule, so the system cannot adapt to an arbitrary telemetry
schema.

This subpackage closes that gap *additively*.  A :class:`~autogram.schema.spec.SchemaSpec`
is a small, JSON-serialisable, bounded **data** object that describes a schema -- its column
patterns, role ontology, grounding templates, family selectors and cell codec.  A *trusted*
compiler (:func:`~autogram.schema.compiler.compile_spec`, no ``eval``/code execution) turns a
spec into a :class:`~autogram.schema.adapter.SchemaAdapter` that the existing engine consults
through three narrow, default-off seams (an optional ``NameModel.adapter`` and an optional
``Grammar.adapter``).  When no adapter is present the engine behaves exactly as before, so the
CrossCheck path is byte-for-byte unchanged.

``crosscheck_spec`` reproduces the hardcoded behaviour exactly (the regression/faithfulness
gate), and ``schema/benchmark2.py`` exercises the whole path on a structurally different
synthetic schema to show the parser, grounding templates and cell codec are *induced from
data*, not hardcoded.
"""

from .spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    RefTemplate,
    RoleOntology,
    SchemaSpec,
)
from .adapter import SchemaAdapter
from .compiler import CompileError, compile_spec
from .crosscheck import crosscheck_spec

__all__ = [
    "CellCodec",
    "ColumnPattern",
    "FamilySelector",
    "RefTemplate",
    "RoleOntology",
    "SchemaSpec",
    "SchemaAdapter",
    "CompileError",
    "compile_spec",
    "crosscheck_spec",
]
