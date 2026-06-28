"""A :class:`SchemaSpec` that reproduces the hardcoded CrossCheck behaviour exactly.

This spec is the *faithfulness anchor* for the whole generalisation: compiling it must yield an
adapter whose ``parse_column`` / ``infer_tokens`` / ``resolve_ref`` / ``resolve_family`` /
``enumerate_bindings`` / cell codec agree, column-for-column and binding-for-binding, with the
hand-written :mod:`autogram.loader.names`, :mod:`autogram.dsl.binders`, and
:mod:`autogram.loader.loader` on the real Abilene and GEANT data (see
``tests/test_schema_faithfulness.py``).  It deliberately *imports* the ontology constants from
:mod:`autogram.dsl.ast` so the spec can never silently drift from the engine's role tables.

It exists only to prove the generalised path subsumes the special case; the CrossCheck datasets
themselves keep ``adapter=None`` so the hot path stays byte-for-byte unchanged.
"""

from __future__ import annotations

from ..dsl import ast as A
from .spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    RefTemplate,
    RoleOntology,
    SchemaSpec,
)


def crosscheck_spec() -> SchemaSpec:
    """Return the declarative spec equivalent to the hardcoded CrossCheck scaffold."""
    patterns = (
        ColumnPattern(
            name="origination", matcher="regex", kind="low", direction="origination",
            regex=r"^low_(?P<node>.+)_origination$",
            node_groups=("node",), src_group="node", token_groups=("node",)),
        ColumnPattern(
            name="termination", matcher="regex", kind="low", direction="termination",
            regex=r"^low_(?P<node>.+)_termination$",
            node_groups=("node",), dst_group="node", token_groups=("node",)),
        ColumnPattern(
            name="egress", matcher="regex", kind="low", direction="egress",
            regex=r"^low_(?P<node>.+)_egress_to_(?P<peer>.+)$",
            node_groups=("node", "peer"), src_group="node", peer_group="peer",
            token_groups=("node",)),
        ColumnPattern(
            name="ingress", matcher="regex", kind="low", direction="ingress",
            regex=r"^low_(?P<node>.+)_ingress_from_(?P<peer>.+)$",
            node_groups=("node", "peer"), dst_group="node", peer_group="peer",
            token_groups=("node",)),
        ColumnPattern(
            name="demand", matcher="split", kind="high", direction="demand",
            prefix="high_", sep="_", split_slots=("src", "dst")),
    )

    ontology = RoleOntology(
        binders=A.BINDERS,
        ref_roles={b: tuple(rs) for b, rs in A.REF_ROLES.items()},
        fam_roles={b: tuple(rs) for b, rs in A.FAM_ROLES.items()},
        ops=A.OPS,
        agg_kinds=A.AGG_KINDS,
        ref_glyphs=dict(A._ROLE_GLYPH),
        fam_glyphs=dict(A._FAM_GLYPH),
    )

    ref_templates = (
        RefTemplate("cell", "self", "{col}"),
        RefTemplate("node", "origination", "low_{X}_origination"),
        RefTemplate("node", "termination", "low_{X}_termination"),
        RefTemplate("node", "demand_self", "high_{X}_{X}"),
        RefTemplate("link", "egress", "low_{X}_egress_to_{Y}"),
        RefTemplate("link", "ingress_rev", "low_{Y}_ingress_from_{X}"),
        RefTemplate("link", "egress_rev", "low_{Y}_egress_to_{X}"),
        RefTemplate("link", "ingress", "low_{X}_ingress_from_{Y}"),
        RefTemplate("link", "demand", "high_{X}_{Y}"),
        RefTemplate("link", "demand_rev", "high_{Y}_{X}"),
    )

    family_selectors = (
        FamilySelector("node", "demand_row", "high",
                       predicates=(("src", "==", "X"), ("dst", "!=", "X"))),
        FamilySelector("node", "demand_col", "high",
                       predicates=(("dst", "==", "X"), ("src", "!=", "X"))),
        FamilySelector("node", "ingress_fam", "low", match_direction="ingress",
                       predicates=(("local", "==", "X"),)),
        FamilySelector("node", "egress_fam", "low", match_direction="egress",
                       predicates=(("local", "==", "X"),)),
        FamilySelector("network", "all_orig", "low", match_direction="origination"),
        FamilySelector("network", "all_term", "low", match_direction="termination"),
        FamilySelector("network", "all_demand", "high",
                       predicates=(("src", "!=", "@dst"),)),
    )

    binder_enumerate = {
        "cell": "per_measured_col",
        "node": "per_node",
        "link": "per_directed_link",
        "network": "singleton",
    }

    return SchemaSpec(
        name="crosscheck",
        patterns=patterns,
        ontology=ontology,
        ref_templates=ref_templates,
        family_selectors=family_selectors,
        binder_enumerate=binder_enumerate,
        cell_codec=CellCodec(kind="dict_gt_hidden",
                             primary="ground_truth", clean="hidden_ground_truth"),
        noisy_kind="low",
        demand_kind="high",
        link_marker_dir="egress",
        notes="Faithful declarative reproduction of loader.names/dsl.binders/loader.loader.",
    )
