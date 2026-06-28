"""Shared fixtures for the discovery test-suite.

Everything is synthetic and offline: a small dataset with structured names + planted invariants,
its induced schema adapter, the derived grammar, and a built :class:`Dataset`.  No CrossCheck
data, no oracle, no network.
"""

from __future__ import annotations

import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.induce import HeuristicInducer, induce_adapter
from autogram.dsl.grammar import grammar_from_adapter
from autogram.loader.loader import build_dataset


@pytest.fixture(scope="session")
def data():
    return synth.make_synthetic(n_entities=4, n_snapshots=160, noise=0.02, seed=0)


@pytest.fixture(scope="session")
def adapter(data):
    return induce_adapter(data.columns)


@pytest.fixture(scope="session")
def grammar(adapter):
    return grammar_from_adapter(adapter, max_complexity=12, max_add_arity=3)


@pytest.fixture(scope="session")
def dataset(data, adapter):
    return build_dataset(data.columns, data.matrix, adapter, "synthetic", data.timestamps)


@pytest.fixture
def small_cfgs():
    return (DiscoveryConfig(n_perm=8, seed=0),
            SearchConfig(rounds=4, proposals_per_round=80, seed=0))
