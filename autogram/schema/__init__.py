"""Declarative schema layer: the data an inducer emits and the trusted compiler that runs it.

A :class:`~autogram.schema.spec.SchemaSpec` is a small, JSON-serialisable, bounded **data**
object that describes a dataset schema -- its column patterns, role ontology, grounding
templates, family selectors and cell codec.  A *trusted* compiler
(:func:`~autogram.schema.compiler.compile_spec`, no ``eval``/code execution) turns a spec into
a :class:`~autogram.schema.adapter.SchemaAdapter` that the engine consults for name parsing,
binding enumeration, grounding and cell decoding.

The spec is the seam the discovery pipeline fills *by induction* from column names (see
:mod:`autogram.discovery.induce`).  It is never hand-written for a particular dataset.
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
]
