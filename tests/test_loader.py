"""Focused loader regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autogram.dsl import ast as A
from autogram.dsl.evaluate import eval_term
from autogram.loader.loader import build_dataset
from autogram.schema.compiler import compile_spec
from autogram.schema.spec import (
    CellCodec,
    ColumnPattern,
    GrammarSpec,
    RefTemplate,
    RoleOntology,
)


def _matrix_adapter(*, time_index=""):
    return compile_spec(GrammarSpec(
        name="matrix_identity",
        patterns=(
            ColumnPattern(
                name="metric",
                matcher="regex",
                kind="measurement",
                direction="value",
                regex=r"^(?:event_time|group_id|metric)$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ()},
            fam_roles={"record": ()},
        ),
        ref_templates=(),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
        noisy_kind="measurement",
        time_index=time_index,
        group_keys=("group_id",),
    ))


def _boolean_context_adapter(
    *,
    boolean_role=True,
    condition_values=(False, True),
):
    return compile_spec(GrammarSpec(
        name="boolean_context",
        patterns=(
            ColumnPattern(
                name="measurement",
                matcher="regex",
                kind="measurement",
                direction="value",
                regex=r"^(?:flag|metric)$",
            ),
        ),
        ontology=RoleOntology(
            binders=("record",),
            ref_roles={"record": ("flag", "metric")},
            fam_roles={"record": ()},
        ),
        ref_templates=(
            RefTemplate("record", "flag", "flag"),
            RefTemplate("record", "metric", "metric"),
        ),
        family_selectors=(),
        binder_enumerate={"record": "singleton"},
        cell_codec=CellCodec(kind="scalar"),
        noisy_kind="measurement",
        condition_columns={"flag": condition_values},
        boolean_roles={"record": ("flag",)} if boolean_role else {},
        advanced_enabled=True,
    ))


@pytest.mark.parametrize(
    "matrix,expected_dtype",
    [
        (
            np.array([
                [2**53, 1],
                [2**53 + 1, 2],
            ], dtype=np.int64),
            np.dtype(np.int64),
        ),
        (
            [
                [2**53, 1.0],
                [2**53 + 1, 2.0],
            ],
            np.dtype(object),
        ),
        (
            pd.DataFrame({
                "group_id": pd.Series(
                    [2**53, 2**53 + 1],
                    dtype=np.int64,
                ),
                "metric": [1.0, 2.0],
            }),
            np.dtype(np.int64),
        ),
    ],
    ids=("integer-array", "row-sequence", "dataframe"),
)
def test_matrix_loader_preserves_large_integer_group_identity(
    matrix,
    expected_dtype,
):
    dataset = build_dataset(
        ("group_id", "metric"),
        matrix,
        _matrix_adapter(),
        "large_groups",
    )

    identities = dataset.row_context["group_id"]
    assert identities.dtype == expected_dtype
    assert identities.tolist() == [2**53, 2**53 + 1]
    assert dataset.observed.matrix.dtype == np.float64
    assert dataset.observed.col("metric").tolist() == [1.0, 2.0]


def test_matrix_loader_casts_measurements_without_casting_group_keys():
    matrix = np.array([
        ["consumer-a", "1.5"],
        ["consumer-b", "2.5"],
    ], dtype=object)

    dataset = build_dataset(
        ("group_id", "metric"),
        matrix,
        _matrix_adapter(),
        "string_groups",
    )

    assert dataset.row_context["group_id"].tolist() == [
        "consumer-a",
        "consumer-b",
    ]
    assert dataset.observed.col("metric").tolist() == [1.5, 2.5]

    matrix[0, 1] = "not-a-number"
    with pytest.raises(
        ValueError,
        match="measurement column 'metric' cannot be converted to float",
    ):
        build_dataset(
            ("group_id", "metric"),
            matrix,
            _matrix_adapter(),
            "invalid_measurement",
        )


def test_matrix_loader_preserves_declared_time_column_when_not_overridden():
    matrix = [
        [2**53, "consumer-a", 1.0],
        [2**53 + 1, "consumer-b", 2.0],
    ]

    dataset = build_dataset(
        ("event_time", "group_id", "metric"),
        matrix,
        _matrix_adapter(time_index="event_time"),
        "source_time",
    )

    assert dataset.timestamps.tolist() == [2**53, 2**53 + 1]
    assert dataset.row_context["event_time"].tolist() == [
        2**53,
        2**53 + 1,
    ]
    assert dataset.observed.names == ["metric"]


def test_matrix_loader_keeps_row_sequence_boolean_context_ref():
    dataset = build_dataset(
        ("flag", "metric"),
        [[False, 1.0], [True, 2.0]],
        _boolean_context_adapter(),
        "row_booleans",
    )

    assert dataset.observed.names == ["flag", "metric"]
    assert [type(value) for value in dataset.row_context["flag"]] == [
        bool,
        bool,
    ]
    np.testing.assert_array_equal(
        eval_term(
            A.Ref("flag"),
            "record",
            {},
            dataset.observed,
            dataset.name_model,
        ),
        [0.0, 1.0],
    )


def test_matrix_loader_keeps_nullable_boolean_context_ref():
    frame = pd.DataFrame({
        "flag": pd.Series([False, pd.NA, True], dtype="boolean"),
        "metric": [1.0, 2.0, 3.0],
    })

    dataset = build_dataset(
        tuple(frame.columns),
        frame,
        _boolean_context_adapter(),
        "nullable_booleans",
    )

    assert dataset.observed.names == ["flag", "metric"]
    context = dataset.row_context["flag"]
    assert type(context[0]) is bool
    assert context[1] is pd.NA
    assert type(context[2]) is bool
    np.testing.assert_allclose(
        dataset.observed.col("flag"),
        [0.0, np.nan, 1.0],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "boolean_role,condition_values",
    [
        (False, (False, True)),
        (True, ()),
    ],
    ids=("condition", "ref-role"),
)
def test_matrix_loader_keeps_all_missing_declared_boolean_context(
    boolean_role,
    condition_values,
):
    frame = pd.DataFrame({
        "flag": pd.Series([pd.NA, pd.NA], dtype="boolean"),
        "metric": [1.0, 2.0],
    })

    dataset = build_dataset(
        tuple(frame.columns),
        frame,
        _boolean_context_adapter(
            boolean_role=boolean_role,
            condition_values=condition_values,
        ),
        "missing_booleans",
    )

    assert dataset.observed.names == ["flag", "metric"]
    assert all(value is pd.NA for value in dataset.row_context["flag"])
    assert np.isnan(dataset.observed.col("flag")).all()


@pytest.mark.parametrize(
    "flags",
    [
        [0, 1],
        ["False", "True"],
    ],
    ids=("numeric", "string"),
)
def test_matrix_loader_does_not_infer_non_boolean_context(flags):
    dataset = build_dataset(
        ("flag", "metric"),
        [[flags[0], 1.0], [flags[1], 2.0]],
        _boolean_context_adapter(),
        "non_boolean_context",
    )

    assert dataset.observed.names == ["metric"]
    assert dataset.row_context["flag"].tolist() == flags
