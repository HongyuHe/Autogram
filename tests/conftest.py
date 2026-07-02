"""Shared fixtures for the discovery test-suite.

The data is synthetic, but schema induction still goes through the real subagent backend.  No
CrossCheck data, clean oracle, or target-rule catalogue reaches the engine.
"""

from __future__ import annotations

import pytest

from autogram.config import DiscoveryConfig, SearchConfig
from autogram.discovery import synth
from autogram.discovery.induce import induce_adapter
from autogram.dsl.grammar import grammar_from_adapter
from autogram.loader.loader import build_dataset


@pytest.fixture(scope="session", autouse=True)
def _cache_real_subagent_replies():
    """Keep tests tractable while each unique schema prompt is still parsed by the real subagent."""
    import os

    old = os.environ.get("AUTOGRAM_SUBAGENT_CACHE")
    os.environ["AUTOGRAM_SUBAGENT_CACHE"] = "1"
    yield
    if old is None:
        os.environ.pop("AUTOGRAM_SUBAGENT_CACHE", None)
    else:
        os.environ["AUTOGRAM_SUBAGENT_CACHE"] = old


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
    return (DiscoveryConfig(seed=0, hold_rate_threshold=0.9),
            SearchConfig(seed=0, max_complexity=8))
