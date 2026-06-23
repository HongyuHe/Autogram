"""Shared pytest fixtures.

A single Abilene load is shared across the data-backed tests (session-scoped) so the suite
stays fast.  Tests that do not touch data avoid the fixture entirely.
"""

from __future__ import annotations

import os

import pytest

from autogram.config import EvalConfig
from autogram.loader.loader import load_dataset

_ABILENE = "data/crosscheck-samples/abilene_sample_1000.pkl"


@pytest.fixture(scope="session")
def abilene():
    if not os.path.exists(_ABILENE):
        pytest.skip(f"sample data not found: {_ABILENE}")
    return load_dataset(_ABILENE, name="abilene")


@pytest.fixture(scope="session")
def eval_cfg():
    return EvalConfig()
