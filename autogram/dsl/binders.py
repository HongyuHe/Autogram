"""Binder enumeration and role resolution (grounding support for the DSL).

A *binder* yields a list of *bindings*; each binding resolves the AST's ``role``
strings to concrete column names (for :class:`~autogram.dsl.ast.Ref`) or to a tuple
of columns (for :class:`~autogram.dsl.ast.Agg`).  All resolution is driven by the
:class:`~autogram.loader.names.NameModel`, which itself was built only from column
names -- so grounding leaks no oracle information.

A binding is represented as a small dict so it is serializable and inspectable.
"""

from __future__ import annotations

from typing import List, Optional

from ..loader.names import NameModel


def enumerate_bindings(binder: str, nm: NameModel) -> List[dict]:
    """Return the list of bindings for ``binder`` over the dataset's locality."""
    if getattr(nm, "adapter", None) is not None:
        return nm.adapter.enumerate_bindings(binder, nm)
    if binder == "cell":
        return [{"col": c} for c in (list(nm.low_cols) + list(nm.high_cols))]
    if binder == "node":
        return [{"X": x} for x in nm.node_list()]
    if binder == "link":
        out = []
        for x in nm.node_list():
            # a directed link X->Y exists iff an egress counter is present
            for c, sem in nm.by_name.items():
                if sem.kind == "low" and sem.direction == "egress" and sem.src == x:
                    out.append({"X": x, "Y": sem.peer})
        return out
    if binder == "network":
        return [{}]
    raise ValueError(f"unknown binder {binder!r}")


def resolve_ref(role: str, binder: str, binding: dict, nm: NameModel) -> Optional[str]:
    """Resolve a single-column ``role`` to a concrete column name (or ``None``)."""
    if getattr(nm, "adapter", None) is not None:
        return nm.adapter.resolve_ref(role, binder, binding, nm)
    if binder == "cell":
        return binding["col"] if role == "self" else None
    if binder == "node":
        x = binding["X"]
        if role == "origination":
            return _first(nm, f"low_{x}_origination")
        if role == "termination":
            return _first(nm, f"low_{x}_termination")
        if role == "demand_self":
            return _first(nm, f"high_{x}_{x}")
        return None
    if binder == "link":
        x, y = binding["X"], binding["Y"]
        table = {
            "egress": f"low_{x}_egress_to_{y}",
            "ingress_rev": f"low_{y}_ingress_from_{x}",
            "egress_rev": f"low_{y}_egress_to_{x}",
            "ingress": f"low_{x}_ingress_from_{y}",
            "demand": f"high_{x}_{y}",
            "demand_rev": f"high_{y}_{x}",
        }
        return _first(nm, table.get(role))
    return None


def resolve_family(family_role: str, binder: str, binding: dict, nm: NameModel) -> tuple:
    """Resolve a family ``role`` to a tuple of concrete column names."""
    if getattr(nm, "adapter", None) is not None:
        return nm.adapter.resolve_family(family_role, binder, binding, nm)
    if binder == "node":
        x = binding["X"]
        if family_role == "demand_row":
            return nm.resolve_family(x, "high", "src")
        if family_role == "demand_col":
            return nm.resolve_family(x, "high", "dst")
        if family_role == "ingress_fam":
            return nm.resolve_family(x, "low", "ingress")
        if family_role == "egress_fam":
            return nm.resolve_family(x, "low", "egress")
    if binder == "network":
        if family_role == "all_orig":
            return nm.resolve_family("*", "low", "origination")
        if family_role == "all_term":
            return nm.resolve_family("*", "low", "termination")
        if family_role == "all_demand":
            return nm.resolve_family("*", "high", "all")
    return ()


def _first(nm: NameModel, name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return name if name in nm.by_name else None
