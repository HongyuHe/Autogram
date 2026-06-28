"""Schema-generality gate: the second benchmark recovers its planted catalogue end-to-end.

``benchmark2`` is a structurally different schema (``tx_/rx_/if_/dem[]`` syntax, scalar cells,
a regex demand matcher) whose adapter is *compiled from a SchemaSpec*, not the hardcoded
CrossCheck path.  Running the real engine over its planted data must rediscover all eight
testable laws, proving the four CrossCheck seams (name parser, role grounding, cell codec,
loader) are genuinely induced from the spec rather than wired in.  The run is scripted
(deterministic, no LLM) and reads no ground-truth catalogue during learning.
"""

from __future__ import annotations

import numpy as np

from autogram.schema.benchmark2 import (
    BENCH2_TARGETS,
    benchmark2_spec,
    make_benchmark2_frame,
    run_benchmark2,
)


def test_benchmark2_recovers_full_catalogue():
    """seed=0 yields 8/8 strict recall over the planted benchmark2 targets."""
    rep, portfolio, ds, n_eval, n_uniq = run_benchmark2(
        n_snapshots=300, n_nodes=4, seed=0, iterations=150)
    assert rep.n_targets == 8
    assert rep.recall == 1.0
    assert rep.strict_recall == 1.0
    assert rep.n_full == 8
    assert n_eval > 0 and len(portfolio) > 0


def test_benchmark2_targets_are_relabelled_catalogue():
    """The planted target set is the CrossCheck fingerprints re-keyed as B-I*, exact-strict."""
    tids = {t.tid for t in BENCH2_TARGETS}
    assert tids == {"B-I1", "B-I2", "B-I4", "B-I5", "B-I6", "B-I7", "B-I8", "B-I9"}


def test_benchmark2_transit_breaks_direct_routing_identity():
    """Conserved circulation keeps conservation exact but breaks ``term == sum(ingress)``.

    The transit term is what lets the 4-term conservation law (I7) survive assembly; this
    asserts the data actually realises it: per node, ``sum(ingress) > termination`` (and
    ``sum(egress) > origination``) by the circulation, while node conservation still holds
    exactly.
    """
    df, nodes = make_benchmark2_frame(n_snapshots=64, n_nodes=4, seed=0)
    for x in nodes:
        ingress = sum(df[c].to_numpy() for c in df.columns
                      if c.startswith("if_") and c.endswith("_in") and f"if_{x}_from_" in c)
        egress = sum(df[c].to_numpy() for c in df.columns
                     if c.startswith("if_") and c.endswith("_out") and f"if_{x}_to_" in c)
        term = df[f"rx_{x}_snk"].to_numpy()
        orig = df[f"tx_{x}_src"].to_numpy()
        # direct-routing identities are broken by the transit circulation ...
        assert np.all(ingress > term + 1e-9)
        assert np.all(egress > orig + 1e-9)
        # ... yet node flow conservation (the circulation cancels) holds exactly.
        assert np.allclose(orig + ingress, term + egress, rtol=0, atol=1e-6)


def test_benchmark2_spec_is_compilable_and_distinct():
    """The spec compiles and is not the CrossCheck low_/high_ schema."""
    spec = benchmark2_spec()
    assert spec.name == "benchmark2"
    pats = " ".join(p.regex for p in spec.patterns)
    assert "tx_" in pats and "if_" in pats and "dem" in pats
    assert "low_" not in pats and "high_" not in pats
