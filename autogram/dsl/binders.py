"""Binder enumeration and role resolution (grounding support for the DSL).

A *binder* yields a list of *bindings*; each binding resolves the AST's ``role`` strings to
concrete column names (for :class:`~autogram.dsl.ast.Ref`) or to a tuple of columns (for
:class:`~autogram.dsl.ast.Agg`).  All resolution is delegated to the *induced* schema adapter
attached to the :class:`~autogram.loader.names.NameModel`; there is no hardcoded, dataset
specific fallback.  Grounding therefore leaks no oracle information -- it is driven entirely
by a schema that was itself induced from column names.

A binding is represented as a small dict so it is serializable and inspectable.
"""

from __future__ import annotations

from typing import List, Optional

from ..loader.names import NameModel


def _require_adapter(nm: NameModel):
    adapter = getattr(nm, "adapter", None)
    if adapter is None:
        raise ValueError(
            "grounding requires an induced schema adapter; build the NameModel via "
            "NameModel.from_columns_with_adapter")
    return adapter


def enumerate_bindings(binder: str, nm: NameModel) -> List[dict]:
    """Return the list of bindings for ``binder`` over the dataset's locality."""
    return _require_adapter(nm).enumerate_bindings(binder, nm)


def resolve_ref(role: str, binder: str, binding: dict, nm: NameModel) -> Optional[str]:
    """Resolve a single-column ``role`` to a concrete column name (or ``None``)."""
    return _require_adapter(nm).resolve_ref(role, binder, binding, nm)


def resolve_family(family_role: str, binder: str, binding: dict, nm: NameModel) -> tuple:
    """Resolve a family ``role`` to a tuple of concrete column names."""
    return _require_adapter(nm).resolve_family(family_role, binder, binding, nm)
