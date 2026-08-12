"""Learned center bands for healthy operating regimes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.known import KnownInvariant, recover_known
from autogram.discovery.loop import build_dataframe_grammar, run_prepared
from autogram.discovery.propose import EnumerationProposer, normalize_rule
from autogram.dsl import ast as A
from autogram.dsl.typecheck import is_admissible
from autogram.loader.gtib import profile_dataframe
from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="band",
        patterns=(
            ColumnPattern(
                "placeholder",
                "regex",
                "unused",
                "unused",
                regex=r"^does_not_match$",
            ),
        ),
        ontology=RoleOntology(
            binders=("network",),
            ref_roles={"network": ()},
            fam_roles={"network": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"network": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
    )


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    archetype = np.where(np.arange(n) % 4 == 0, "bursty_ml", "steady")
    label = np.where(np.arange(n) % 7 == 0, "true_loss", "normal")
    ratio = rng.normal(0.998, 0.001, size=n)
    ratio[archetype == "bursty_ml"] += rng.normal(0.0, 0.08, size=np.sum(archetype == "bursty_ml"))
    ratio[label == "true_loss"] -= 0.08
    return profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "consumer_id": "consumer",
            "archetype": archetype,
            "label": label,
            "completeness_ratio": ratio,
        }),
        time_index="timestamp",
        group_keys=("consumer_id",),
        condition_columns=("archetype", "label"),
        band_enabled=True,
    )


def _condition():
    return A.Condition(
        "",
        "all",
        (
            A.Condition("archetype", "==", ("steady",)),
            A.Condition("label", "==", ("normal",)),
        ),
    )


def test_learned_band_recovers_steady_normal_healthy_regime():
    dataset, grammar = build_dataframe_grammar(_frame(), _base_spec(), name="healthy")
    rule = A.Rule(
        "record",
        A.BandDefinition(A.Ref("completeness_ratio"), None),
        condition=_condition(),
    )

    assert is_admissible(rule, grammar)[0] is True
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.9,
            band_mode="global",
        ),
    ).evaluate(rule)

    assert result.accepted
    assert abs(result.parameters["center"] - 0.998) < 0.002
    assert result.hold_rate > 0.99
    rendered = {candidate.unparse() for candidate in EnumerationProposer(grammar).propose()}
    assert normalize_rule(rule).unparse() in rendered


def test_archive_preserves_conditioned_band_behavior():
    dataset, grammar = build_dataframe_grammar(
        _frame(),
        _base_spec(),
        name="healthy_archive",
    )
    expected = normalize_rule(A.Rule(
        "record",
        A.BandDefinition(A.Ref("completeness_ratio"), None),
        condition=_condition(),
    ))

    result = run_prepared(
        dataset,
        grammar,
        discovery_cfg=DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.5,
            band_mode="adaptive",
            seed=0,
        ),
        search_cfg=SearchConfig(seed=0),
    )

    assert expected.signature() in {
        evaluation.rule.signature()
        for evaluation in result.portfolio
    }


def test_conditional_band_enforces_and_reports_source_support():
    n_rows = 200
    rare = np.zeros(n_rows, dtype=bool)
    rare[:30] = True
    ratio = np.linspace(0.5, 1.5, n_rows)
    ratio[rare] = 0.998
    frame = profile_dataframe(
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="1min"),
            "consumer_id": "consumer",
            "rare": rare,
            "completeness_ratio": ratio,
        }),
        time_index="timestamp",
        group_keys=("consumer_id",),
        condition_columns=("rare",),
        band_enabled=True,
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="rare_band",
    )
    rule = A.Rule(
        "record",
        A.BandDefinition(A.Ref("completeness_ratio"), None),
        condition=A.Condition("rare", "==", (True,)),
    )

    rejected = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.5,
            band_mode="global",
            min_condition_points=100,
            min_condition_fraction=0.5,
        ),
    ).evaluate(rule)
    accepted = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.5,
            band_mode="global",
            min_condition_points=20,
            min_condition_fraction=0.1,
        ),
    ).evaluate(rule)

    assert not rejected.accepted
    assert "condition support" in rejected.reason
    assert accepted.accepted
    assert abs(accepted.support - 0.15) < 1e-12


def test_learned_band_keeps_and_gates_every_declared_group():
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": np.repeat(["large", "small"], [100, 2]),
            "value": np.concatenate([
                np.ones(100),
                np.full(2, 10.0),
            ]),
        }),
        group_keys=("group_id",),
        band_enabled=True,
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="grouped_band",
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.8,
            band_mode="adaptive",
            seed=1,
        ),
    ).evaluate(
        A.Rule("record", A.BandDefinition(A.Ref("value"), None))
    )

    assert not result.accepted
    assert set(result.parameters["group_hold_rates"]) == {
        "large",
        "small",
    }
    assert result.parameters["group_hold_rate_lows"]["small"] < 0.8


def test_known_band_matching_checks_center():
    dataset, _grammar = build_dataframe_grammar(_frame(), _base_spec(), name="healthy")
    rule = A.Rule(
        "record",
        A.BandDefinition(A.Ref("completeness_ratio"), None),
        condition=_condition(),
    )
    evaluation = SimpleNamespace(
        rule=rule,
        parameters={"center": 0.9981},
    )
    result = SimpleNamespace(dataset=dataset, portfolio=[evaluation])
    correct = KnownInvariant(
        "healthy",
        "~band",
        "completeness_ratio",
        {"center": 0.998},
        where={"all": [{"archetype": "steady"}, {"label": "normal"}]},
    )
    wrong = KnownInvariant(
        "wrong",
        "~band",
        "completeness_ratio",
        {"center": 0.5},
        where={"all": [{"archetype": "steady"}, {"label": "normal"}]},
    )

    assert recover_known(result, [correct])["recall"] == 1.0
    assert recover_known(result, [wrong])["recall"] == 0.0

    unconditional = SimpleNamespace(
        dataset=dataset,
        portfolio=[
            SimpleNamespace(
                rule=A.Rule(
                    "record",
                    A.BandDefinition(A.Ref("completeness_ratio"), None),
                ),
                parameters={"center": 0.9981},
            )
        ],
    )
    assert recover_known(unconditional, [correct])["recall"] == 0.0
