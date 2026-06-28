"""Faithfulness gate: the compiled CrossCheck adapter == the hardcoded scaffold.

The whole generalisation rests on one claim: a declarative :class:`SchemaSpec`, run through the
trusted compiler, reproduces the four hardcoded CrossCheck seams *exactly*.  These tests prove
that claim column-for-column and binding-for-binding on the real Abilene and GEANT samples, so
``adapter=None`` (CrossCheck) and a compiled ``crosscheck_spec()`` are interchangeable.  They
touch no engine module, so they are pure additions to the regression gate.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from autogram.dsl import ast as A
from autogram.dsl import binders as B
from autogram.loader import names as N
from autogram.loader.loader import load_dataset
from autogram.schema import CompileError, compile_spec, crosscheck_spec
from autogram.schema.spec import CellCodec, ColumnPattern, RoleOntology, SchemaSpec

_PATHS = {
    "abilene": "data/crosscheck-samples/abilene_sample_1000.pkl",
    "geant": "data/crosscheck-samples/geant_sample_1000.pkl",
}


@pytest.fixture(scope="module")
def adapter():
    return compile_spec(crosscheck_spec())


def _load(name):
    p = _PATHS[name]
    if not os.path.exists(p):
        pytest.skip(f"sample data not found: {p}")
    return p, pd.read_pickle(p)


@pytest.mark.parametrize("name", ["abilene", "geant"])
def test_parse_and_tokens_match(adapter, name):
    _, df = _load(name)
    cols = list(df.columns)
    assert adapter.infer_tokens(cols) == N.infer_nodes(cols)
    nodes = frozenset(N.infer_nodes(cols))
    for c in cols:
        assert adapter.parse_column(c, nodes) == N.parse_column(c, nodes), c


@pytest.mark.parametrize("name", ["abilene", "geant"])
def test_grounding_matches(adapter, name):
    _, df = _load(name)
    nm = N.NameModel.from_columns(list(df.columns))
    for binder in A.BINDERS:
        bh = B.enumerate_bindings(binder, nm)
        assert adapter.enumerate_bindings(binder, nm) == bh, binder
        for role in A.REF_ROLES[binder]:
            for bd in bh:
                assert adapter.resolve_ref(role, binder, bd, nm) == \
                    B.resolve_ref(role, binder, bd, nm), (binder, role, bd)
        for fam in A.FAM_ROLES[binder]:
            for bd in bh:
                assert adapter.resolve_family(fam, binder, bd, nm) == \
                    B.resolve_family(fam, binder, bd, nm), (binder, fam, bd)


@pytest.mark.parametrize("name", ["abilene", "geant"])
def test_cell_codec_matches(adapter, name):
    p, df = _load(name)
    ds = load_dataset(p, name=name)
    nm = ds.name_model
    ordered = list(nm.low_cols) + list(nm.high_cols)
    n = len(df)
    obs = np.empty((n, len(ordered)))
    cln = np.empty((n, len(ordered)))
    for j, c in enumerate(ordered):
        kind = nm.by_name[c].kind
        vals = df[c].values
        for i in range(n):
            obs[i, j] = adapter.decode_observed(vals[i])
            cln[i, j] = adapter.decode_clean(vals[i], kind)
    assert np.allclose(obs, ds.observed.matrix, equal_nan=True)
    assert np.allclose(cln, ds.clean.matrix, equal_nan=True)


def test_compiler_rejects_bad_strategy():
    spec = crosscheck_spec()
    bad = SchemaSpec(
        name="bad", patterns=spec.patterns, ontology=spec.ontology,
        ref_templates=spec.ref_templates, family_selectors=spec.family_selectors,
        binder_enumerate={**spec.binder_enumerate, "cell": "no_such_strategy"},
        cell_codec=spec.cell_codec)
    with pytest.raises(CompileError):
        compile_spec(bad)


def test_compiler_rejects_bad_codec():
    spec = crosscheck_spec()
    bad = SchemaSpec(
        name="bad", patterns=spec.patterns, ontology=spec.ontology,
        ref_templates=spec.ref_templates, family_selectors=spec.family_selectors,
        binder_enumerate=spec.binder_enumerate,
        cell_codec=CellCodec(kind="not_a_codec"))
    with pytest.raises(CompileError):
        compile_spec(bad)


def test_compiler_rejects_bad_regex():
    onto = RoleOntology(binders=("cell",), ref_roles={"cell": ("self",)},
                        fam_roles={"cell": ()})
    bad = SchemaSpec(
        name="bad",
        patterns=(ColumnPattern(name="x", matcher="regex", kind="low",
                                direction="origination", regex=r"(?P<n>.+"),),
        ontology=onto, ref_templates=(), family_selectors=(),
        binder_enumerate={"cell": "per_measured_col"},
        cell_codec=CellCodec(kind="scalar"))
    with pytest.raises(CompileError):
        compile_spec(bad)
