"""Arithmetic overflow is a candidate's own failure, never silently-missing data.

Round-29 / TODO-1.  ``A.Div`` guards a division by *exact* zero, but a finite-but-tiny denominator
overflows ``float64`` to an infinity.  The grounder used to drop every non-finite row through its
finite mask, so an overflowed row vanished: it was neither graded nor counted as a violation, while
the reported support still described the *offered* population.  That is a false discovery presented
with full confidence, and the Wilson bound computed on the survivors is meaningless because ``n`` no
longer counts the rows the rule claims to describe.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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


def test_overflow_only_group_still_fails_the_per_group_gate():
    n_clean = 95
    n_overflow = 5
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": (
                ["clean"] * n_clean
                + ["overflow"] * n_overflow
            ),
            "a": np.concatenate((
                np.ones(n_clean),
                np.full(n_overflow, 1e308),
            )),
            "b": np.concatenate((
                np.ones(n_clean),
                np.full(n_overflow, 1e308),
            )),
            "target": np.ones(n_clean + n_overflow),
        }),
        group_keys=("group_id",),
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="overflow_only_group",
    )
    rule = A.Rule(
        "record",
        A.Compare(
            A.Ref("target"),
            "~=",
            A.Mul(A.Ref("a"), A.Ref("b")),
        ),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.9,
            band_mode="global",
            max_overflow_fraction=0.1,
        ),
    ).evaluate(rule)

    assert result.support == pytest.approx(0.95)
    assert not result.accepted
    assert result.parameters["group_hold_rates"]["overflow"] == 0.0


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
    # Exact accounting, not a bound: charging only the first member reports zero bytes for every
    # entry, which satisfies any "<= max_bytes" assertion while letting the cache grow unbounded.
    values = np.ones(100, dtype=float)
    overflow = np.zeros(100, dtype=bool)
    entry_bytes = int(values.nbytes) + int(overflow.nbytes)

    cache = TermCache(max_entries=8, max_bytes=10 * entry_bytes)
    cache["only"] = (values.copy(), overflow.copy())
    assert cache.total_bytes == entry_bytes

    for index in range(10):
        cache[index] = (values.copy(), overflow.copy())

    # The byte ceiling, not the entry ceiling, is what evicts here.
    assert cache.total_bytes == len(cache) * entry_bytes
    assert cache.total_bytes <= 10 * entry_bytes
    assert 0 < len(cache) <= 8


def _definition_dataset(name: str):
    """A Boolean target defined by a ratio whose denominator underflows on most rows.

    The predicate is TRUE exactly where the ratio is genuinely large, and the overflowing rows are
    the ones where the definition would otherwise be scored on nothing at all.
    """
    n = 120
    num = np.full(n, 4.0)
    den = np.full(n, 1.0)
    target = np.zeros(n)
    target[:40] = 1.0
    num[40:] = 1e300
    den[40:] = 1e-320          # 1e300 / 1e-320 overflows to +inf
    df = pd.DataFrame({
        "target": target.astype(bool),
        "num": num,
        "den": den,
    })
    return _dataset(df, name)


def _definition_rule() -> A.Rule:
    return A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Bound(A.Div(A.Ref("num"), A.Ref("den")), ">", 2.0),
        ),
    )


def test_boolean_definition_is_refused_when_its_predicate_overflows():
    dataset = _definition_dataset("overflow_definition")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.62, band_mode="global", seed=0),
    )

    result = evaluator.evaluate(_definition_rule())

    assert not result.accepted
    assert "overflow" in result.reason


def test_band_definition_is_refused_when_its_term_overflows():
    df = pd.DataFrame({
        "num": np.concatenate([np.full(40, 4.0), np.full(80, 1e300)]),
        "den": np.concatenate([np.full(40, 2.0), np.full(80, 1e-320)]),
    })
    dataset = _dataset(df, "overflow_band_definition")
    rule = A.Rule(
        "record",
        A.BandDefinition(A.Div(A.Ref("num"), A.Ref("den")), None),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    ).evaluate(rule)

    assert not result.accepted
    assert "overflow" in result.reason


def test_definition_without_overflow_is_still_evaluated_normally():
    n = 200
    rng = np.random.default_rng(3)
    value = rng.uniform(0.0, 10.0, size=n)
    df = pd.DataFrame({"target": value > 6.0, "value": value})
    dataset = _dataset(df, "clean_definition")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(A.Ref("target"), A.Bound(A.Ref("value"), ">", 6.0)),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.62, band_mode="global", seed=0),
    ).evaluate(rule)

    assert "overflow" not in result.reason
    assert result.accepted


def test_proportional_fit_overflow_is_refused_not_scored():
    """Round-29 review: the fitted coefficient introduces arithmetic ``ground()`` never saw.

    ``rho = left - coefficient * right`` and its scale are recomputed AFTER grounding, so a large
    fitted coefficient can overflow the product on rows whose operands were finite. Those rows were
    then scored as ordinary residuals behind a full-confidence support figure.
    """
    n = 60
    right = np.full(n, 1.0)
    left = np.full(n, 1e308)
    right[-1] = 10.0                      # 1e308 * 10 overflows once the coefficient is fitted
    left[-1] = 1e308
    df = pd.DataFrame({"x": right, "y": left})
    dataset = _dataset(df, "proportional_overflow")

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    ).evaluate(A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x"))))

    assert not result.accepted
    assert "overflow" in result.reason


def test_conditioned_fit_overflow_keeps_full_frame_support_denominator():
    n = 1000
    selected = np.arange(n) < 100
    x = np.ones(n)
    x[50] = 10.0
    y = np.ones(n)
    y[selected] = 1e308
    frame = profile_dataframe(
        pd.DataFrame({
            "x": x,
            "y": y,
            "label": np.where(selected, "selected", "other"),
        }),
        condition_columns=("label",),
        proportional=True,
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="conditioned_overflow_support",
    )
    rule = A.Rule(
        "record",
        A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")),
        condition=A.Condition(
            "label",
            "==",
            ("selected",),
        ),
    )

    grounded = ground(
        rule,
        dataset.observed,
        dataset.name_model,
    )
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05,
            hold_rate_threshold=0.62,
            band_mode="global",
            seed=1,
            max_overflow_fraction=0.02,
        ),
    ).evaluate(rule)

    assert grounded.support == pytest.approx(0.1)
    assert result.accepted
    assert result.support == pytest.approx(0.099)


def test_tolerated_band_overflow_group_still_fails_group_gate():
    frame = profile_dataframe(
        pd.DataFrame({
            "group_id": (
                ["clean"] * 380
                + ["overflow"] * 20
            ),
            "x": np.concatenate((
                np.full(380, 1.5e308),
                np.full(20, -1.5e308),
            )),
        }),
        group_keys=("group_id",),
    )
    dataset, _grammar = build_dataframe_grammar(
        frame,
        _base_spec(),
        name="band_overflow_group",
    )
    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.01,
            hold_rate_threshold=0.9,
            band_mode="global",
            max_overflow_fraction=0.1,
        ),
    ).evaluate(A.Rule(
        "record",
        A.BandDefinition(A.Ref("x"), None),
    ))

    assert result.support == pytest.approx(0.95)
    assert not result.accepted
    assert result.parameters["group_hold_rates"]["overflow"] == 0.0


def test_proportional_coefficient_median_survives_ceiling_scale_ratios():
    # A well-determined coefficient must not be discarded (and the estimator silently swapped for
    # least squares) merely because the median's intermediate sum overflowed.
    from autogram.discovery.evaluate import _fit_proportional

    n = 80
    x = np.full(n, 1.0)
    y = np.full(n, 1.5e308)
    df = pd.DataFrame({"x": x, "y": y})
    dataset = _dataset(df, "proportional_ceiling")
    g = ground(
        A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x"))),
        dataset.observed,
        dataset.name_model,
    )
    fitted = _fit_proportional(g, dataset.observed, dataset.name_model, DiscoveryConfig(seed=0))

    assert fitted is not None
    coefficients = fitted[0]
    assert all(abs(value - 1.5e308) < 1e295 for value in coefficients.values())


def test_band_definition_support_reports_the_graded_population():
    """Round-29 review: an accepted band reported full support while grading a tenth of the rows."""
    values = np.full(100, np.nan)
    values[:10] = 5.0
    df = pd.DataFrame({"m": values})
    dataset = _dataset(df, "band_graded_support")
    rule = A.Rule("record", A.BandDefinition(A.Ref("m"), 5.0))

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    ).evaluate(rule)

    assert result.n_points == 10
    assert abs(result.support - 0.1) < 1e-9


def test_null_control_magnitudes_stay_finite_at_ceiling_scale():
    """Round-29 review: a finite median was not enough to keep the null control finite.

    The lognormal multiplier is unbounded above, so a ceiling-scale column still generated
    infinities. Those become missing data, the null candidates built on them are refused for
    overflowing rather than judged on their merits, and the false-discovery control silently goes
    vacuous -- which is the one thing the calibration protocol cannot tolerate.
    """
    from autogram.discovery.validate import _balanced_null_numeric

    for seed in range(5):
        generated = _balanced_null_numeric(
            np.full(100, 1.5e308), np.random.default_rng(seed), binary=False,
        )
        assert np.all(np.isfinite(generated)), seed
        assert np.any(generated > 0.0) and np.any(generated < 0.0)

    # Finite leaves are not enough: the null control is run through the SAME candidate grammar as
    # the data, so its sums, differences and degree-2 products must stay finite too. Otherwise every
    # null candidate is refused for overflowing rather than judged on its merits, and the
    # false-discovery control silently goes vacuous.
    left = _balanced_null_numeric(
        np.full(500, 1.5e308), np.random.default_rng(11), binary=False,
    )
    right = _balanced_null_numeric(
        np.full(500, 1.5e308), np.random.default_rng(12), binary=False,
    )
    for combined in (left - right, left + right, left * right):
        assert np.all(np.isfinite(combined))

    # Ordinary data must be untouched by the safeguard.
    ordinary = _balanced_null_numeric(
        np.linspace(1.0, 100.0, 200), np.random.default_rng(1), binary=False,
    )
    assert np.all(np.isfinite(ordinary))
    assert 10.0 < float(np.median(np.abs(ordinary))) < 200.0


def test_proportional_overflow_is_caught_even_when_the_subsample_misses_it():
    """Round-30 review: a coreset must never decide a universal claim.

    With ``subsample`` active, the single overflowing row was dropped from the scoring population
    and the rule was accepted with coefficient 1e308 and support 1.0, although its own expression
    overflowed. The guard now reads the full graded population.
    """
    n = 400
    x = np.full(n, 1.0)
    y = np.full(n, 1e308)
    x[123] = 10.0                      # 1e308 * 10 overflows once the coefficient is fitted
    y[123] = 1e308
    dataset = _dataset(pd.DataFrame({"x": x, "y": y}), "proportional_subsampled")

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05, hold_rate_threshold=0.62, band_mode="global",
            seed=0, subsample=50,
        ),
    ).evaluate(A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x"))))

    assert not result.accepted
    assert "overflow" in result.reason


def test_base_and_post_fit_overflow_share_one_cap():
    """Round-30 review: two disjoint sub-cap overflows must not pass a single cap between them."""
    n = 100
    y = np.full(n, 1e308)
    y2 = np.zeros(n)
    x = np.full(n, 1.0)
    y2[:8] = 1e308        # `y + y2` overflows on 8 rows, caught while grounding
    x[8:16] = 10.0        # `coefficient * x` overflows on 8 more, only after the fit
    dataset = _dataset(pd.DataFrame({"x": x, "y": y, "y2": y2}), "combined_overflow")
    rule = A.Rule(
        "record",
        A.Compare(A.Add((A.Ref("y"), A.Ref("y2"))), "~\u221d", A.Ref("x")),
    )

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(
            tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0,
            max_overflow_fraction=0.10,
        ),
    ).evaluate(rule)

    assert not result.accepted
    # Each source of overflow is 8% on its own -- under the cap -- but they compose to 16%.
    assert "16 of 100" in result.reason


def test_band_support_accounts_for_ungrounded_bindings():
    """Round-30 review: support must fall when only some bindings ground.

    ``Grounded.support`` multiplies the graded-row fraction by the fraction of bindings that
    grounded. The band path counted only the rows, so a term resolvable for one binding out of two
    still reported support 1.0 -- and hard-coded ``n_bindings=1`` besides.
    """
    import autogram.discovery.evaluate as E

    df = pd.DataFrame({"m": np.full(60, 5.0)})
    dataset = _dataset(df, "band_bindings")
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    )
    rule = A.Rule("record", A.BandDefinition(A.Ref("m"), 5.0))

    grounded = evaluator.evaluate(rule)
    assert grounded.accepted
    assert abs(grounded.support - 1.0) < 1e-9
    assert grounded.n_bindings == 1

    # Offer a second binding whose term does not ground: half the attempted bindings now ground.
    original_bindings = E.enumerate_bindings
    original_eval = E.eval_term
    calls = {"n": 0}

    def _two_bindings(binder, nm):
        return [{}, {"unused": "second"}]

    def _second_is_out_of_scope(term, binder, binding, frame, nm):
        calls["n"] += 1
        return None if calls["n"] == 2 else original_eval(term, binder, binding, frame, nm)

    E.enumerate_bindings = _two_bindings
    E.eval_term = _second_is_out_of_scope
    try:
        half = evaluator.evaluate(rule)
    finally:
        E.enumerate_bindings = original_bindings
        E.eval_term = original_eval

    assert half.n_bindings == 1
    assert abs(half.support - 0.5) < 1e-9


def test_sustained_definition_overflow_counts_every_tainted_window():
    """Round-31 review: one blown-up row taints every SUSTAINED window that contains it.

    Counting the source row alone undercounts the population the definition fails to describe, so a
    single bad row in two hundred passed a 1% cap while ten windows were actually unevaluable.
    """
    n = 200
    window = 10
    value = np.full(n, 1.0)
    num = np.full(n, 2.0)
    den = np.full(n, 1.0)
    num[50] = 1e300
    den[50] = 1e-320                        # one overflowing row
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "consumer_id": ["c0"] * n,
        "target": (value > 0.5),
        "num": num,
        "den": den,
    })
    frame = profile_dataframe(
        df, time_index="timestamp", group_keys=("consumer_id",),
        temporal_windows=(window,), advanced=True,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="sustained_overflow")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Sustained(A.Bound(A.Div(A.Ref("num"), A.Ref("den")), ">", 1.0), window),
        ),
    )

    cfg = DiscoveryConfig(hold_rate_threshold=0.62, band_mode="global", seed=0,
                          max_overflow_fraction=0.01)
    result = DataOnlyEvaluator(dataset, cfg).evaluate(rule)

    assert not result.accepted
    assert "overflow" in result.reason
    # 1/200 is under the 1% cap; the ten tainted windows are not.
    assert " 10 of " in result.reason


def test_proportional_group_keys_do_not_collide_when_stringified():
    """Round-31 review: distinct group labels that stringify identically must not share a coefficient.

    Keying by ``str(label)`` let group ``1`` and group ``"1"`` collide, so one group's fitted
    coefficient silently replaced the other's -- and the finite-arithmetic guard then checked the
    wrong coefficient.
    """
    from autogram.discovery.evaluate import _fit_proportional
    from autogram.dsl.evaluate import ground

    n = 120
    labels = np.array([1 if index % 2 else "1" for index in range(n)], dtype=object)
    x = np.full(n, 2.0)
    y = np.where(labels == 1, 6.0, 20.0)
    df = pd.DataFrame({"group_id": labels, "x": x, "y": y})
    frame = profile_dataframe(df, group_keys=("group_id",))
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="colliding_groups")
    rule = A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")))

    g = ground(rule, dataset.observed, dataset.name_model)
    fitted = _fit_proportional(g, dataset.observed, dataset.name_model, DiscoveryConfig(seed=0))

    assert fitted is not None
    coefficients = fitted[0]
    assert len(coefficients) == 2, coefficients
    assert sorted(round(value, 6) for value in coefficients.values()) == [3.0, 10.0]


def test_tolerated_post_fit_overflow_still_shrinks_reported_support():
    """Round-32 review: an overflow inside the cap is tolerated, but it is not evidence.

    With a non-zero `max_overflow_fraction` the rule is allowed through, yet the blown-up rows were
    still scored and the reported support still described the whole population.
    """
    n = 1000
    x = np.full(n, 1.0)
    y = np.full(n, 1e308)
    x[500] = 10.0                       # one post-fit overflow, 0.1% of the rows
    dataset = _dataset(pd.DataFrame({"x": x, "y": y}), "tolerated_overflow")
    cfg = DiscoveryConfig(
        tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0,
        max_overflow_fraction=0.01,
    )
    rule = A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x")))

    baseline = DataOnlyEvaluator(dataset, cfg).evaluate(rule)

    assert baseline.support < 1.0
    assert abs(baseline.support - 0.999) < 1e-9

    # The row must also leave the SCORED population, not merely the support figure. Find a seed
    # whose evaluation split contains the overflowing row, so the narrowing is what removes it.
    from autogram.discovery.evaluate import _fit_proportional
    from autogram.dsl.evaluate import ground

    for seed in range(12):
        seeded = replace(cfg, seed=seed)
        g = ground(rule, dataset.observed, dataset.name_model, seed=seed)
        fitted = _fit_proportional(g, dataset.observed, dataset.name_model, seeded)
        assert fitted is not None
        evaluation_mask = fitted[2]
        if not bool(evaluation_mask[500]):
            continue
        scored = DataOnlyEvaluator(dataset, seeded).evaluate(rule)
        assert scored.n_points == int(np.count_nonzero(evaluation_mask)) - 1
        break
    else:                                                   # pragma: no cover - fixture guard
        raise AssertionError("no seed placed the overflowing row in the evaluation split")


def test_reported_group_keys_survive_labels_that_stringify_alike():
    """Round-32 review: reporting re-introduced the collision that fitting had just removed.

    The persisted parameters are the audit trail for an accepted per-group law, so two distinct
    labels must not collapse into one key and lose a coefficient.
    """
    n = 120
    labels = np.array([1 if index % 2 else "1" for index in range(n)], dtype=object)
    x = np.full(n, 2.0)
    y = np.where(labels == 1, 6.0, 20.0)
    frame = profile_dataframe(
        pd.DataFrame({"group_id": labels, "x": x, "y": y}), group_keys=("group_id",),
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="reported_groups")

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.02, hold_rate_threshold=0.85, band_mode="global", seed=0),
    ).evaluate(A.Rule("record", A.Compare(A.Ref("y"), "~\u221d", A.Ref("x"))))

    reported = result.parameters["coefficients"]
    assert len(reported) == 2, reported
    assert sorted(round(value, 6) for value in reported.values()) == [3.0, 10.0]
    assert len(result.parameters["group_hold_rates"]) == 2


def _tainted_definition_dataset(name: str, n: int = 100, window: int = 10, blown: int = 2):
    value = np.linspace(0.0, 10.0, n)
    num = np.full(n, 2.0)
    den = np.full(n, 1.0)
    for index in range(blown):
        num[20 + index] = 1e300
        den[20 + index] = 1e-320
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "consumer_id": ["c0"] * n,
        "target": value > 5.0,
        "num": num,
        "den": den,
    })
    frame = profile_dataframe(
        df, time_index="timestamp", group_keys=("consumer_id",),
        temporal_windows=(window,), advanced=True,
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name=name)
    return dataset, window


def test_tolerated_definition_overflow_is_not_scored_as_evidence():
    """Round-33 review: a tolerated overflow must still leave the definition's population.

    A SUSTAINED predicate turns a tainted window into a confident ``False``, which then scores as a
    correct prediction -- so leaving the rows in inflates both the agreement and the support of a
    definition the data cannot actually witness there.
    """
    dataset, window = _tainted_definition_dataset("tainted_definition")
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Sustained(A.Bound(A.Div(A.Ref("num"), A.Ref("den")), ">", 1.0), window),
        ),
    )
    cfg = DiscoveryConfig(hold_rate_threshold=0.62, band_mode="global", seed=0,
                          max_overflow_fraction=0.5)

    result = DataOnlyEvaluator(dataset, cfg).evaluate(rule)

    # Tolerated (2 blown rows taint 11 windows, well under the 50% cap), but excluded from scoring.
    assert "overflow" not in result.reason
    assert 0 < result.n_points <= 100 - 11


def test_display_keys_are_injective_for_labels_that_render_alike():
    """Round-33 review: a per-label fallback re-collides with a label that was never ambiguous.

    ``1``, ``"1"`` and ``"'1'"`` render as ``1``, ``'1'`` and ``'1'`` -- so disambiguating only the
    colliding pair produces two identical keys anyway, and a third group's coefficient vanishes from
    the persisted parameters.
    """
    from autogram.discovery.evaluate import _display_keys, _typed_label

    for labels in (
        [1, "1", "'1'"],
        ["a", "b", "c"],
        [1, 2, 3],
        [(1, "a"), (1, "b")],
        [True, 1, "1"],
        [(True, "x"), (1, "x")],
    ):
        typed = [_typed_label(label) for label in labels]
        assert len(set(typed)) == len(labels), labels        # typed identity keeps them apart
        keys = _display_keys(typed)
        assert len(set(keys.values())) == len(keys), (labels, keys)


def test_reduction_that_cancels_to_nan_is_an_overflow_not_missing_data():
    """Round-34 review: a reduction can return NaN from finite members.

    Pairwise summation overflows a partial sum and then cancels, so the result is ``NaN`` rather
    than an infinity -- and a guard that only looks for infinities lets those rows be dropped as
    ordinary missing data, shrinking the population behind a full-confidence support figure.
    """
    import autogram.dsl.binders as B
    import autogram.dsl.evaluate as E

    # Sixteen members with alternating signs: numpy's unrolled accumulators reach +inf and -inf
    # separately and their combination is NaN, even though every member is finite and the exact
    # mathematical sum is zero.
    n = 40
    columns = [f"m{index}" for index in range(16)]
    values = {
        name: np.full(n, 1.5e308 if index % 2 == 0 else -1.5e308)
        for index, name in enumerate(columns)
    }
    for name in columns:                       # the first ten rows sum cleanly to zero
        values[name][:10] = np.sign(values[name][:10])
    df = pd.DataFrame({**values, "target": np.zeros(n)})
    dataset = _dataset(df, "nan_reduction")
    with np.errstate(over="ignore", invalid="ignore"):
        last = np.stack([values[c] for c in columns], axis=1)[-1].sum()
    assert np.isnan(last), last

    original = B.resolve_family
    B.resolve_family = lambda role, binder, binding, nm: tuple(columns)
    try:
        _value, overflow = E.eval_term_overflow(
            A.Agg("SUM", "fam"), "record", {}, dataset.observed, dataset.name_model,
        )
    finally:
        B.resolve_family = original

    assert overflow is not None
    assert int(np.count_nonzero(overflow)) == 30      # every row whose partial sums cancelled


def test_rolling_reduction_that_cancels_to_nan_is_an_overflow():
    n = 60
    window = 16
    series = np.resize(np.array([1.5e308, -1.5e308]), n)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "consumer_id": ["c0"] * n,
        "m": series,
        "target": np.zeros(n),
    })
    frame = profile_dataframe(
        df, time_index="timestamp", group_keys=("consumer_id",), temporal_windows=(window,),
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="nan_rolling")

    _value, overflow = eval_term_overflow(
        A.Rolling(A.Ref("m"), window, "SUM"), "record", {},
        dataset.observed, dataset.name_model,
    )

    assert overflow is not None and bool(np.any(overflow))


def test_band_centre_overflow_is_refused():
    """Round-34 review: `value - centre` is post-fit arithmetic the grounding pass never saw."""
    values = np.full(60, 1.5e308)
    values[:20] = -1.5e308
    dataset = _dataset(pd.DataFrame({"m": values}), "band_centre_overflow")
    rule = A.Rule("record", A.BandDefinition(A.Ref("m"), -1.5e308))

    result = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(tolerance=0.05, hold_rate_threshold=0.62, band_mode="global", seed=0),
    ).evaluate(rule)

    assert not result.accepted
    assert "overflow" in result.reason


def test_definition_taint_is_confined_to_the_binding_that_blew_up():
    """Round-34 review: one binding's blow-up must not excuse another binding's failures.

    A frame-row mask applied to every binding removes rows the other bindings never blew up on --
    and those rows are exactly where a wrong definition fails.
    """
    from autogram.discovery.evaluate import _binding_key

    dataset, window = _tainted_definition_dataset("binding_taint", blown=2)
    rule = A.Rule(
        "record",
        A.BooleanDefinition(
            A.Ref("target"),
            A.Sustained(A.Bound(A.Div(A.Ref("num"), A.Ref("den")), ">", 1.0), window),
        ),
    )
    evaluator = DataOnlyEvaluator(
        dataset,
        DiscoveryConfig(hold_rate_threshold=0.62, band_mode="global", seed=0,
                        max_overflow_fraction=0.5),
    )

    _rejection, tainted = evaluator._definition_overflow_rejection(rule)

    assert isinstance(tainted, dict)
    assert list(tainted) == [_binding_key({})]
    assert int(np.count_nonzero(next(iter(tainted.values())))) == 11


def test_typed_group_identity_survives_stratified_subsampling():
    """Round-35 review: a subsample must not drop a group by merging it with another.

    ``True == 1`` and they hash alike, so bucketing by the raw label merged two groups; the merged
    bucket then contributed one sample and the failing group vanished from the per-group gate.
    """
    from autogram.dsl.evaluate import _row_group_keys, _stratified_subsample, typed_group_key

    # A rare typed-distinct group: merged bucketing draws from one pool and misses it, while
    # typed bucketing is obliged to represent every group.
    n = 120
    labels = np.array([True if index < 2 else 1 for index in range(n)], dtype=object)
    df = pd.DataFrame({
        "group_id": labels,
        "x": np.full(n, 2.0),
        "y": np.where(labels == True, 4.0, 6.0),          # noqa: E712 - typed comparison intended
    })
    frame = profile_dataframe(df, group_keys=("group_id",))
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="typed_subsample")

    rows = np.arange(n)
    observed = _row_group_keys(dataset.observed, dataset.name_model, rows)
    assert observed is not None
    assert len({typed_group_key(item) for item in observed.tolist()}) == 2

    keep = _stratified_subsample(rows, dataset.observed, dataset.name_model, 4, 0)
    assert keep is not None
    kept_groups = {typed_group_key(observed[index]) for index in keep}
    assert len(kept_groups) == 2, "a typed-distinct group was dropped from the subsample"


def test_composite_group_keys_do_not_collapse_across_types():
    from autogram.discovery.evaluate import _typed_label

    assert _typed_label((True, "x")) != _typed_label((1, "x"))
    assert _typed_label((1, "x")) == _typed_label((1, "x"))
    assert _typed_label(((1, True), "y")) != _typed_label(((1, 1), "y"))


def test_global_group_sentinel_cannot_collide_with_a_real_label():
    """A string sentinel reports one group's coefficient under another's name."""
    from autogram.discovery.evaluate import GLOBAL_GROUP, _reported_coefficients, _typed_label

    coefficients = {
        _typed_label(GLOBAL_GROUP): 1.0,
        _typed_label("global"): 2.0,
        _typed_label("__global__"): 3.0,
    }

    reported = _reported_coefficients(coefficients)

    assert len(reported) == 3
    assert reported["global"] == 1.0


def test_span_window_does_not_wrap_at_the_timestamp_ceiling():
    """Round-35 review: int64 timestamps wrap, so a window could end before it starts."""
    import pandas as pd_local

    from autogram.dsl.evaluate import _saturating_add_ns

    ceiling = pd_local.Timestamp.max.value
    starts = np.array([ceiling - 1000, ceiling], dtype=np.int64)

    ends, saturated = _saturating_add_ns(starts, 10 ** 12)
    assert bool(np.all(saturated))

    assert np.all(ends >= starts), "the window ended before it started"
    assert np.all(np.isfinite(ends.astype(float)))
    # Ordinary timestamps are untouched.
    ordinary, ordinary_saturated = _saturating_add_ns(np.array([0], dtype=np.int64), 60 * 10 ** 9)
    assert int(ordinary[0]) == 60 * 10 ** 9 and not bool(np.any(ordinary_saturated))


def test_typed_groups_all_reach_the_evaluation_split():
    """Round-36 review: a grouped holdout split must represent every typed-distinct group.

    Raw bucketing merged ``True`` with ``1``, and the merged split left the failing rows entirely
    out of the evaluation half -- where they could no longer fail the per-group gate, so a false
    law was accepted at hold rate 1.0.
    """
    from autogram.discovery.evaluate import _parameter_masks
    from autogram.evaluator.band import _grouped_split

    n = 120
    labels = np.array([True if index < 2 else 1 for index in range(n)], dtype=object)

    _calibration, evaluation = _grouped_split(labels, 0.3, 0)
    evaluated = {bool(labels[index]) is True and isinstance(labels[index], bool)
                 for index in evaluation}
    assert any(isinstance(labels[index], bool) for index in evaluation), (
        "the rare typed group never reached the band's evaluation split"
    )
    assert evaluated  # both renderings present

    valid = np.ones(n, dtype=bool)
    cfg = DiscoveryConfig(parameter_holdout_frac=0.3, seed=0)
    _fit_mask, eval_mask = _parameter_masks(valid, cfg, split=True, groups=labels)
    assert any(
        isinstance(labels[index], bool)
        for index in np.flatnonzero(eval_mask)
    ), "the rare typed group never reached the definition's evaluation split"


def test_related_join_keeps_typed_distinct_keys_apart():
    """Round-36 review: pandas groupby merges ``True`` with ``1`` before the key is ever seen."""
    from autogram.dsl.evaluate import _related_aggregate
    from autogram.schema.spec import RelatedTemplate

    timestamps = pd.date_range("2026-01-01", periods=6, freq="10s")
    n = len(timestamps)
    # Built as an object column element-wise: `pd.concat` would coerce `True` to `1` before the
    # engine ever sees the two shards as distinct.
    shard_ids = np.empty(2 * n, dtype=object)
    shard_ids[:n] = True
    shard_ids[n:] = 1
    raw = pd.DataFrame({
        "timestamp": list(timestamps) * 2,
        "shard_id": shard_ids,
        "level": np.concatenate([np.full(n, 10.0), np.full(n, 20.0)]),
    })
    assert {type(value).__name__ for value in raw["shard_id"]} == {"bool", "int"}

    parent = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 00:00:00", periods=2, freq="1min"),
        "consumer_id": ["c0", "c0"],
        "total": [1.0, 1.0],
    })
    frame = profile_dataframe(
        parent, time_index="timestamp", group_keys=("consumer_id",),
        related_frames={"raw": raw},
    )
    dataset, _grammar = build_dataframe_grammar(frame, _base_spec(), name="typed_related")
    template = RelatedTemplate(
        binder="record", role="raw_level", relation="raw", column="level",
        mode="sum_last", parent_keys=(), child_keys=(), partition_keys=("shard_id",),
        parent_time="timestamp", child_time="timestamp", window_seconds=60,
        reset_column="", validity_columns=(),
    )

    values, _overflow = _related_aggregate(template, dataset.observed)

    # Two typed-distinct shards contribute 10 and 20; merging them would report 30 for one shard.
    finite = values[np.isfinite(values)]
    assert finite.size
    assert np.allclose(finite, 30.0), finite   # the SUM over both shards, computed once each


def test_forward_fill_treats_an_infinite_reading_as_missing():
    """Round-36 review: ffill only carries NaN, so an infinity survives as a real reading."""
    from autogram.dsl.evaluate import _partition_values

    child = pd.DataFrame({"counter": [10.0, np.inf, 20.0]})
    partition = {"positions": np.arange(3), "values": {}, "reset_prefix": {}}

    filled = _partition_values(partition, child, "counter", True)

    assert np.allclose(filled, [10.0, 10.0, 20.0])


def test_saturating_add_is_correct_for_pre_epoch_timestamps():
    """Round-36 review: ``limit - times`` itself overflows for a negative timestamp."""
    import pandas as pd_local

    from autogram.dsl.evaluate import _saturating_add_ns

    start = pd_local.Timestamp("1969-12-31 23:59").value
    shifted, saturated = _saturating_add_ns(np.array([start], dtype=np.int64), 60 * 10 ** 9)
    assert not bool(np.any(saturated)), "a pre-epoch timestamp must not report saturation"
    end = int(shifted[0])

    assert pd_local.Timestamp(end) == pd_local.Timestamp("1970-01-01 00:00:00")


def test_missing_group_labels_do_not_fragment_into_singletons():
    """Round-37 review: ``NaN != NaN``, so a typed identity that keeps it makes every row its own group.

    A group of one is placed entirely in the evaluation half, so it can neither be fitted nor
    meaningfully gated -- the typed identity that was introduced to stop groups being *merged* would
    instead shatter them.
    """
    from autogram.discovery.evaluate import _typed_label
    from autogram.dsl.evaluate import typed_group_key

    missing = [float("nan"), float("nan"), None]
    for identity in (typed_group_key, _typed_label):
        keys = {identity(value) for value in missing}
        assert len(keys) == 1, keys
        assert identity(True) != identity(1)          # the real distinction still holds
        assert identity((float("nan"), "x")) == identity((None, "x"))


def test_partition_order_follows_natural_order_not_string_order():
    """Round-37 review: partition visit order decides accumulation order, and so the total.

    Sorting typed keys by their rendered form puts ``"10"`` before ``"9"``, which reorders the
    floating-point accumulation and changes the sum. The order must follow the natural ordering of
    the values, as the previous ``groupby(sort=True)`` did.
    """
    from autogram.dsl.evaluate import typed_group_key, typed_sort_key

    labels = [9, 10, 2]
    ordered = sorted(labels, key=lambda value: typed_sort_key(typed_group_key(value)))
    assert ordered == [2, 9, 10]

    mixed = ["b", "a"]
    assert sorted(mixed, key=lambda v: typed_sort_key(typed_group_key(v))) == ["a", "b"]

    # Types stay grouped, so a mixed column is still deterministically ordered.
    both = [1, "1", True]
    keys = [typed_sort_key(typed_group_key(value)) for value in both]
    assert len(set(keys)) == 3
