"""Arithmetic overflow is a candidate's own failure, never silently-missing data.

Round-29 / TODO-1.  ``A.Div`` guards a division by *exact* zero, but a finite-but-tiny denominator
overflows ``float64`` to an infinity.  The grounder used to drop every non-finite row through its
finite mask, so an overflowed row vanished: it was neither graded nor counted as a violation, while
the reported support still described the *offered* population.  That is a false discovery presented
with full confidence, and the Wilson bound computed on the survivors is meaningless because ``n`` no
longer counts the rows the rule claims to describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autogram.config import DiscoveryConfig
from autogram.discovery.evaluate import DataOnlyEvaluator
from autogram.discovery.loop import build_dataframe_grammar
from autogram.dsl import ast as A
from autogram.dsl.evaluate import eval_term_overflow, ground, robust_median
from autogram.loader.gtib import profile_dataframe
from autogram.loader.loader import TermCache
from autogram.schema.spec import CellCodec, ColumnPattern, GrammarSpec, RoleOntology


def _base_spec() -> GrammarSpec:
    return GrammarSpec(
        name="flat",
        patterns=(
            ColumnPattern(
                name="placeholder",
                matcher="regex",
                kind="unused",
                direction="unused",
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


def _dataset(df: pd.DataFrame, name: str):
    dataset, _grammar = build_dataframe_grammar(
        profile_dataframe(df),
        _base_spec(),
        name=name,
    )
    return dataset


def _overflowing_ratio_frame() -> pd.DataFrame:
    """20 rows where ``ratio == num / den`` holds, 80 where the division overflows.

    The denominator is subnormal but strictly non-zero on the overflowing rows, so ``A.Div``'s
    exact-zero guard does not fire and the quotient becomes ``+inf``.
    """
    num = np.full(100, 2.0)
    den = np.full(100, 1.0)
    ratio = np.full(100, 2.0)
    num[20:] = 1e300
    den[20:] = 1e-320
    ratio[20:] = 7.0            # the law is FALSE on these rows
    return pd.DataFrame({"num": num, "den": den, "ratio": ratio})


def _ratio_rule() -> A.Rule:
    return A.Rule(
        "record",
        A.Compare(A.Ref("ratio"), "~=", A.Div(A.Ref("num"), A.Ref("den"))),
    )


def test_overflowed_rows_are_counted_not_silently_dropped():
    dataset = _dataset(_overflowing_ratio_frame(), "overflow_ratio")

    g = ground(_ratio_rule(), dataset.observed, dataset.name_model)

    assert g.overflow_points == 80
    assert abs(g.overflow_fraction - 0.8) < 1e-12
    assert g.graded_points == 20
    # Support must describe the rows the rule was scored on, not the rows it was offered.
    assert abs(g.support - 0.2) < 1e-12


def test_overflowing_ratio_law_is_refused_rather_than_scored_on_survivors():
    dataset = _dataset(_overflowing_ratio_frame(), "overflow_ratio_eval")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    )

    result = evaluator.evaluate(_ratio_rule())

    assert not result.accepted
    assert "overflow" in result.reason
    assert result.support < 1.0


def test_division_by_exact_zero_stays_undefined_and_is_not_an_overflow():
    # Missing/undefined is a property of the DATA and is legitimately dropped; only the candidate's
    # own blow-up is an overflow. Conflating the two would refuse every guarded ratio.
    num = np.full(100, 2.0)
    den = np.full(100, 1.0)
    den[:40] = 0.0
    df = pd.DataFrame({"num": num, "den": den, "ratio": np.full(100, 2.0)})
    dataset = _dataset(df, "zero_denominator")

    g = ground(_ratio_rule(), dataset.observed, dataset.name_model)

    assert g.overflow_points == 0
    assert g.graded_points == 60
    assert abs(g.support - 0.6) < 1e-12


def test_multiplication_overflow_is_detected():
    left = np.full(50, 1e200)
    right = np.full(50, 1e200)
    right[:10] = 1.0
    df = pd.DataFrame({"a": left, "b": right, "c": np.full(50, 1e200)})
    dataset = _dataset(df, "overflow_mul")
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("c"), "~=", A.Mul(A.Ref("a"), A.Ref("b"))),
    )

    g = ground(rule, dataset.observed, dataset.name_model)

    assert g.overflow_points == 40
    assert g.graded_points == 10


def test_additive_chain_overflow_is_detected():
    a = np.full(50, 1e308)
    b = np.full(50, 1.0)
    b[:10] = 0.0
    df = pd.DataFrame({"a": a, "b": b, "c": np.full(50, 1e308)})
    dataset = _dataset(df, "overflow_add")
    # 1e308 + 1e308 overflows; 1e308 + 0 does not.
    rule = A.Rule(
        "record",
        A.Compare(
            A.Ref("c"),
            "~=",
            A.Add((A.Ref("a"), A.Mul(A.Ref("b"), A.Ref("a")))),
        ),
    )

    g = ground(rule, dataset.observed, dataset.name_model)

    assert g.overflow_points == 40
    assert g.graded_points == 10


def test_residual_subtraction_overflow_is_detected():
    # Both sides are finite, but their difference exceeds float64.
    df = pd.DataFrame({
        "a": np.full(30, 1.5e308),
        "b": np.full(30, -1.5e308),
    })
    dataset = _dataset(df, "overflow_residual")
    rule = A.Rule("record", A.Compare(A.Ref("a"), "~=", A.Ref("b")))

    g = ground(rule, dataset.observed, dataset.name_model)

    assert g.overflow_points == 30
    assert g.graded_points == 0


def test_overflow_taint_survives_being_mapped_back_to_a_finite_value():
    # ``1 / (a * b)`` maps an overflowed product back to a finite 0.0, so finiteness of the final
    # value cannot be used to detect the blow-up -- the taint has to be tracked through the tree.
    df = pd.DataFrame({
        "a": np.full(30, 1e300),
        "b": np.full(30, 1e300),
        "c": np.zeros(30),
    })
    dataset = _dataset(df, "overflow_recovered")
    term = A.Div(A.Const(1.0), A.Mul(A.Ref("a"), A.Ref("b")))
    binding: dict = {}

    values, overflow = eval_term_overflow(
        term, "record", binding, dataset.observed, dataset.name_model,
    )

    assert np.all(np.isfinite(values))
    assert overflow is not None and bool(np.all(overflow))

    g = ground(
        A.Rule("record", A.Compare(A.Ref("c"), "~=", term)),
        dataset.observed,
        dataset.name_model,
    )
    assert g.overflow_points == 30


def test_scale_floor_median_cannot_overflow_and_flatten_every_residual():
    # Both columns are finite and no arithmetic overflows, but ``np.median`` averages the two
    # central scale values and that intermediate SUM exceeds float64. The resulting infinite floor
    # used to raise EVERY scale to infinity, drive every relative residual to zero, and accept a
    # law whose two sides differ by 6.7% with a perfect hold rate.
    df = pd.DataFrame({
        "a": np.full(40, 1.5e308),
        "b": np.full(40, 1.4e308),
    })
    dataset = _dataset(df, "huge_scale")
    rule = A.Rule("record", A.Compare(A.Ref("a"), "~=", A.Ref("b")))

    g = ground(rule, dataset.observed, dataset.name_model)

    assert g.overflow_points == 0
    assert np.all(np.isfinite(g.scale))
    relative = np.abs(g.rho) / g.scale
    assert np.all(relative > 0.05)

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    ).evaluate(rule)
    assert not result.accepted


def test_robust_median_matches_numpy_on_ordinary_samples():
    rng = np.random.default_rng(0)
    for size in (1, 2, 7, 8, 101, 500):
        sample = rng.normal(size=size) * 1e3
        assert abs(robust_median(sample) - float(np.median(sample))) < 1e-9
    assert robust_median(np.empty(0)) == 1.0
    # Even-length sample at the float64 ceiling: numpy overflows, the robust form does not.
    ceiling = np.full(4, 1.5e308)
    assert not np.isfinite(np.median(ceiling))
    assert robust_median(ceiling) == 1.5e308


def test_term_cache_charges_both_members_of_a_cached_pair():
    cache = TermCache(max_entries=8, max_bytes=1_024)
    for index in range(10):
        cache[index] = (np.ones(100, dtype=float), np.zeros(100, dtype=bool))

    assert cache.total_bytes <= 1_024
    assert len(cache) <= 8
