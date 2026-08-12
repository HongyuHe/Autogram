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
