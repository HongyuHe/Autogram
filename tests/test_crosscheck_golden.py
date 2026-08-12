"""CrossCheck golden portfolios remain unchanged when GTIB capabilities are disabled."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery.induce import SchemaInducer
from autogram.discovery.induce import make_inducer
from autogram.discovery.export import portfolio_to_dl
from autogram.discovery.loop import discover_dataframe
from autogram.discovery.known import KnownInvariant, recover_known
from autogram.dsl.render import render_rule
from autogram.schema.spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    GrammarSpec,
    RefTemplate,
    RoleOntology,
)


def _crosscheck_spec(
    *,
    link_demand_context: bool = False,
) -> GrammarSpec:
    return GrammarSpec(
        name="crosscheck-golden",
        patterns=(
            ColumnPattern(
                "origination",
                "regex",
                "low",
                "origination",
                regex=r"^low_(?P<source>.+)_origination$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
            ColumnPattern(
                "termination",
                "regex",
                "low",
                "termination",
                regex=r"^low_(?P<source>.+)_termination$",
                node_groups=("source",),
                source_group="source",
                token_groups=("source",),
            ),
            ColumnPattern(
                "egress",
                "regex",
                "low",
                "egress_to",
                regex=r"^low_(?P<source>.+)_egress_to_(?P<peer>.+)$",
                node_groups=("source", "peer"),
                source_group="source",
                peer_group="peer",
                token_groups=("source", "peer"),
            ),
            ColumnPattern(
                "ingress",
                "regex",
                "low",
                "ingress_from",
                regex=r"^low_(?P<source>.+)_ingress_from_(?P<peer>.+)$",
                node_groups=("source", "peer"),
                source_group="source",
                peer_group="peer",
                token_groups=("source", "peer"),
            ),
            ColumnPattern(
                "demand",
                "split",
                "high",
                "demand",
                prefix="high_",
                split_slots=("source", "destination"),
            ),
        ),
        ontology=RoleOntology(
            binders=("cell", "node", "link", "network"),
            ref_roles={
                "cell": ("self",),
                "node": (
                    "measurement_origination",
                    "measurement_termination",
                    "demand_self",
                ),
                "link": (
                    "o0",
                    "o0_rev",
                    "o1",
                    "o1_rev",
                    "demand",
                    "demand_rev",
                    *(("demand_self",) if link_demand_context else ()),
                ),
                "network": (),
            },
            fam_roles={
                "cell": (),
                "node": (
                    "demand_row",
                    "demand_col",
                    "fam_egress_to",
                    "fam_ingress_from",
                ),
                "link": (
                    ("demand_row", "demand_col")
                    if link_demand_context
                    else ()
                ),
                "network": (
                    "all_demand",
                    "all_measurement_origination",
                    "all_measurement_termination",
                ),
            },
            ops=("~=", "==", "!=", "<=", ">=", "<|>"),
            agg_kinds=("SUM", "AVG", "MIN", "MAX"),
        ),
        ref_templates=(
            RefTemplate("cell", "self", "{col}"),
            RefTemplate("node", "measurement_origination", "low_{X}_origination"),
            RefTemplate("node", "measurement_termination", "low_{X}_termination"),
            RefTemplate("node", "demand_self", "high_{X}_{X}"),
            RefTemplate("link", "o0", "low_{X}_egress_to_{Y}"),
            RefTemplate("link", "o0_rev", "low_{Y}_egress_to_{X}"),
            RefTemplate("link", "o1", "low_{X}_ingress_from_{Y}"),
            RefTemplate("link", "o1_rev", "low_{Y}_ingress_from_{X}"),
            RefTemplate("link", "demand", "high_{X}_{Y}"),
            RefTemplate("link", "demand_rev", "high_{Y}_{X}"),
            *(
                (RefTemplate("link", "demand_self", "high_{X}_{X}"),)
                if link_demand_context
                else ()
            ),
        ),
        family_selectors=(
            FamilySelector(
                "node",
                "demand_row",
                "high",
                "demand",
                (("source", "==", "X"), ("destination", "!=", "X")),
            ),
            FamilySelector(
                "node",
                "demand_col",
                "high",
                "demand",
                (("destination", "==", "X"), ("source", "!=", "X")),
            ),
            *(
                (
                    FamilySelector(
                        "link",
                        "demand_row",
                        "high",
                        "demand",
                        (
                            ("source", "==", "X"),
                            ("destination", "!=", "X"),
                        ),
                    ),
                    FamilySelector(
                        "link",
                        "demand_col",
                        "high",
                        "demand",
                        (
                            ("destination", "==", "X"),
                            ("source", "!=", "X"),
                        ),
                    ),
                )
                if link_demand_context
                else ()
            ),
            FamilySelector(
                "node",
                "fam_egress_to",
                "low",
                "egress_to",
                (("source", "==", "X"),),
            ),
            FamilySelector(
                "node",
                "fam_ingress_from",
                "low",
                "ingress_from",
                (("source", "==", "X"),),
            ),
            FamilySelector(
                "network",
                "all_demand",
                "high",
                "demand",
                (("source", "!=", "@destination"),),
            ),
            FamilySelector(
                "network",
                "all_measurement_origination",
                "low",
                "origination",
            ),
            FamilySelector(
                "network",
                "all_measurement_termination",
                "low",
                "termination",
            ),
        ),
        binder_enumerate={
            "cell": "per_measured_col",
            "node": "per_node",
            "link": "per_directed_link",
            "network": "singleton",
        },
        cell_codec=CellCodec(
            kind="dict_gt_hidden",
            primary="ground_truth",
            clean="hidden_ground_truth",
        ),
        noisy_kind="low",
        demand_kind="high",
        link_marker_direction="egress_to",
    )


class _CrossCheckInducer(SchemaInducer):
    def __init__(self, *, link_demand_context: bool = False):
        self.link_demand_context = bool(link_demand_context)

    def induce(self, columns, sample_rows=None) -> GrammarSpec:
        return _crosscheck_spec(
            link_demand_context=self.link_demand_context,
        )


def _baseline_rules(path: str) -> list[str]:
    output = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        output.append(line.split(" # ", 1)[0].rstrip())
    return output


def _portfolio_records(text: str) -> list[str]:
    return [
        line.rstrip()
        for line in text.splitlines()
        if line and not line.startswith("#")
    ]


def _known_catalog(dataset):
    adapter = dataset.name_model.adapter
    nodes = dataset.name_model.node_list()
    known = [
        KnownInvariant(f"nonnegative_{column}", ">=", column, 0)
        for column in dataset.observed.names
    ]
    for node in nodes:
        known.append(KnownInvariant(
            f"zero_self_{node}",
            "~=",
            f"high_{node}_{node}",
            0,
        ))
        row = [f"high_{node}_{other}" for other in nodes if other != node]
        column = [f"high_{other}_{node}" for other in nodes if other != node]
        ingress = [
            name
            for name, semantics in dataset.name_model.by_name.items()
            if semantics.kind == adapter.noisy_kind
            and semantics.direction == "ingress_from"
            and semantics.source == node
        ]
        egress = [
            name
            for name, semantics in dataset.name_model.by_name.items()
            if semantics.kind == adapter.noisy_kind
            and semantics.direction == "egress_to"
            and semantics.source == node
        ]
        known.extend([
            KnownInvariant(
                f"row_sum_{node}",
                "~=",
                f"low_{node}_origination",
                {"sum": row},
            ),
            KnownInvariant(
                f"column_sum_{node}",
                "~=",
                f"low_{node}_termination",
                {"sum": column},
            ),
            KnownInvariant(
                f"node_balance_{node}",
                "~=",
                {"sum": [f"low_{node}_origination", *ingress]},
                {"sum": [f"low_{node}_termination", *egress]},
            ),
        ])
    for binding in adapter.enumerate_bindings("link", dataset.name_model):
        source, peer = binding["X"], binding["Y"]
        known.extend([
            KnownInvariant(
                f"presence_{source}_{peer}",
                "<|>",
                f"low_{source}_egress_to_{peer}",
                f"low_{peer}_egress_to_{source}",
            ),
            KnownInvariant(
                f"two_end_{source}_{peer}",
                "~=",
                f"low_{source}_egress_to_{peer}",
                f"low_{peer}_ingress_from_{source}",
            ),
            KnownInvariant(
                f"directionality_{source}_{peer}",
                "!=",
                f"low_{source}_egress_to_{peer}",
                f"low_{peer}_egress_to_{source}",
            ),
        ])
    known.extend([
        KnownInvariant(
            "network_origination_termination",
            "~=",
            {"sum": [f"low_{node}_origination" for node in nodes]},
            {"sum": [f"low_{node}_termination" for node in nodes]},
        ),
        KnownInvariant(
            "network_origination_demand",
            "~=",
            {"sum": [f"low_{node}_origination" for node in nodes]},
            {
                "sum": [
                    f"high_{source}_{destination}"
                    for source in nodes
                    for destination in nodes
                    if source != destination
                ],
            },
        ),
    ])
    return known


@pytest.mark.parametrize(
    "dataset_path,baseline_path,link_demand_context",
    [
        (
            "data/crosscheck-samples/abilene_sample_1000.pkl",
            "rules/abilene_sample_1000_20260703T034511Z.dl",
            True,
        ),
        (
            "data/crosscheck-samples/geant_sample_1000.pkl",
            "rules/geant_sample_1000_20260703T035301Z.dl",
            False,
        ),
    ],
)
def test_crosscheck_portfolio_matches_checked_in_golden(
    dataset_path,
    baseline_path,
    link_demand_context,
):
    frame = pd.read_pickle(dataset_path)
    result = discover_dataframe(
        frame,
        inducer=_CrossCheckInducer(
            link_demand_context=link_demand_context,
        ),
        discovery_cfg=DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.62,
            band_mode="adaptive",
            seed=0,
        ),
        search_cfg=SearchConfig(max_complexity=10, max_add_arity=2),
        name=Path(dataset_path).stem,
    )
    dataset = result.dataset
    grammar = result.grammar
    current = [
        render_rule(evaluation.rule, dataset.name_model.adapter)
        for evaluation in result.portfolio
    ]

    assert current == _baseline_rules(baseline_path)
    current_records = _portfolio_records(
        portfolio_to_dl(
            result,
            Path(dataset_path).stem,
            seed=0,
            proposer="enumeration",
            git="ignored",
        )
    )
    assert current_records == _portfolio_records(
        Path(baseline_path).read_text(encoding="utf-8")
    )
    recall = recover_known(result, _known_catalog(dataset))
    assert recall["recall"] == 1.0, [
        invariant
        for invariant in recall["invariants"]
        if not invariant["recovered"]
    ]
    assert grammar.temporal_enabled is False
    assert grammar.conditional_enabled is False
    assert grammar.advanced_enabled is False
    assert not any(grammar.related_for(binder) for binder in grammar.binders)


@pytest.mark.parametrize(
    "dataset_path",
    [
        "data/crosscheck-samples/abilene_sample_1000.pkl",
        "data/crosscheck-samples/geant_sample_1000.pkl",
    ],
)
def test_crosscheck_production_induction_recovers_concrete_catalog(dataset_path):
    frame = pd.read_pickle(dataset_path)
    result = discover_dataframe(
        frame,
        inducer=make_inducer("subagent"),
        discovery_cfg=DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.62,
            band_mode="adaptive",
            seed=0,
        ),
        search_cfg=SearchConfig(
            max_complexity=10,
            max_add_arity=2,
            seed=0,
        ),
        name=Path(dataset_path).stem,
    )

    recall = recover_known(
        result,
        _known_catalog(result.dataset),
    )
    assert recall["recall"] == 1.0, [
        invariant
        for invariant in recall["invariants"]
        if not invariant["recovered"]
    ]
