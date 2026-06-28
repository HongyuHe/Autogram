"""A structurally-different second benchmark that proves the schema is *induced*, not hardcoded.

The CrossCheck path is the faithfulness anchor (``crosscheck.py``): it shows the declarative
:class:`SchemaSpec` can reproduce the four hardcoded seams.  This module is the complementary
*generality* evidence: a synthetic schema whose column syntax, demand encoding, and cell format
are all different from CrossCheck, fed through the **same** engine via a compiled adapter, with a
planted ground truth the recall grader can check.

What is deliberately different from CrossCheck (exercises the *other* code branches):

* **Name syntax**   -- ``tx_<R>_src`` / ``rx_<R>_snk`` / ``if_<X>_to_<Y>_out`` /
  ``if_<X>_from_<Y>_in`` / ``dem[<S>=><D>]`` instead of ``low_*`` / ``high_*``.
* **Demand matcher** -- a *regex* with named ``src`` / ``dst`` groups, not the ``split`` prefix
  parser CrossCheck uses for ``high_<src>_<dst>``.
* **Cell codec**    -- plain numeric scalars, not the ``{ground_truth, hidden_ground_truth}``
  dict cells; this drives the ``"scalar"`` codec branch (clean == observed, no injected noise).

What is deliberately *reused* (the documented PoC boundary, design Sec. 6.5-6.7): the engine's
**role vocabulary** -- origination / termination / egress / ingress / demand and the four binder
enumeration strategies.  Inventing brand-new role *names* would require regenerating
``default_grammar``/typecheck/proposer/recall and is out of scope; the seams that genuinely
differ between schemas -- the name parser, the grounding templates, and the cell codec -- are
exactly seams 1/3/4, and those are induced here entirely from the spec.

Because the role vocabulary is reused, ``assemble._residual_keys`` emits the *same* fingerprints
as for CrossCheck, so the planted targets can reuse the catalogue keysets verbatim (no
transcription drift) -- only the labels and the expected strictness change (every law is planted
noise-free, so the two sub-noise *soft* laws of CrossCheck, I5/I6, here hold EXACTLY).

The data is a direct-routed random asymmetric demand mesh: each demand rides its own one-hop
link, which makes every conservation law hold *exactly* and lets the run be a clean pass/fail
generality check rather than a noisy approximation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..dsl import ast as A
from ..evaluator.gate import Verdict
from ..config import RunConfig
from ..loader.loader import load_dataframe, Dataset
from ..proposer.scripted_backend import ScriptedProposer
from ..search.loop import learn
from ..search.recall import RecallReport, Target, TESTABLE_TARGETS, score_against
from .compiler import compile_spec
from .spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    RefTemplate,
    RoleOntology,
    SchemaSpec,
)


# --------------------------------------------------------------------------- the schema spec

def benchmark2_spec() -> SchemaSpec:
    """A bounded declarative spec for the structurally-different second schema.

    Mirrors ``crosscheck_spec`` role-for-role (so the reused vocabulary lines up) but with new
    name syntax, a regex demand matcher, and a scalar cell codec.
    """
    patterns = (
        ColumnPattern(
            name="origination", matcher="regex", kind="low", direction="origination",
            regex=r"^tx_(?P<node>.+)_src$",
            node_groups=("node",), src_group="node", token_groups=("node",)),
        ColumnPattern(
            name="termination", matcher="regex", kind="low", direction="termination",
            regex=r"^rx_(?P<node>.+)_snk$",
            node_groups=("node",), dst_group="node", token_groups=("node",)),
        ColumnPattern(
            name="egress", matcher="regex", kind="low", direction="egress",
            regex=r"^if_(?P<node>.+)_to_(?P<peer>.+)_out$",
            node_groups=("node", "peer"), src_group="node", peer_group="peer",
            token_groups=("node",)),
        ColumnPattern(
            name="ingress", matcher="regex", kind="low", direction="ingress",
            regex=r"^if_(?P<node>.+)_from_(?P<peer>.+)_in$",
            node_groups=("node", "peer"), dst_group="node", peer_group="peer",
            token_groups=("node",)),
        # demand as a REGEX matcher (CrossCheck used a split prefix parser here instead).
        ColumnPattern(
            name="demand", matcher="regex", kind="high", direction="demand",
            regex=r"^dem\[(?P<src>.+)=>(?P<dst>.+)\]$",
            node_groups=("src", "dst"), src_group="src", dst_group="dst",
            token_groups=()),
    )

    # Reuse the engine's role tables verbatim (the documented PoC boundary).
    ontology = RoleOntology(
        binders=A.BINDERS,
        ref_roles={b: tuple(rs) for b, rs in A.REF_ROLES.items()},
        fam_roles={b: tuple(rs) for b, rs in A.FAM_ROLES.items()},
        ops=A.OPS,
        agg_kinds=A.AGG_KINDS,
        ref_glyphs=dict(A._ROLE_GLYPH),
        fam_glyphs=dict(A._FAM_GLYPH),
    )

    ref_templates = (
        RefTemplate("cell", "self", "{col}"),
        RefTemplate("node", "origination", "tx_{X}_src"),
        RefTemplate("node", "termination", "rx_{X}_snk"),
        RefTemplate("node", "demand_self", "dem[{X}=>{X}]"),
        RefTemplate("link", "egress", "if_{X}_to_{Y}_out"),
        RefTemplate("link", "ingress_rev", "if_{Y}_from_{X}_in"),
        RefTemplate("link", "egress_rev", "if_{Y}_to_{X}_out"),
        RefTemplate("link", "ingress", "if_{X}_from_{Y}_in"),
        RefTemplate("link", "demand", "dem[{X}=>{Y}]"),
        RefTemplate("link", "demand_rev", "dem[{Y}=>{X}]"),
    )

    # Identical family predicates to CrossCheck (same role semantics, different columns).
    family_selectors = (
        FamilySelector("node", "demand_row", "high",
                       predicates=(("src", "==", "X"), ("dst", "!=", "X"))),
        FamilySelector("node", "demand_col", "high",
                       predicates=(("dst", "==", "X"), ("src", "!=", "X"))),
        FamilySelector("node", "ingress_fam", "low", match_direction="ingress",
                       predicates=(("local", "==", "X"),)),
        FamilySelector("node", "egress_fam", "low", match_direction="egress",
                       predicates=(("local", "==", "X"),)),
        FamilySelector("network", "all_orig", "low", match_direction="origination"),
        FamilySelector("network", "all_term", "low", match_direction="termination"),
        FamilySelector("network", "all_demand", "high",
                       predicates=(("src", "!=", "@dst"),)),
    )

    binder_enumerate = {
        "cell": "per_measured_col",
        "node": "per_node",
        "link": "per_directed_link",
        "network": "singleton",
    }

    return SchemaSpec(
        name="benchmark2",
        patterns=patterns,
        ontology=ontology,
        ref_templates=ref_templates,
        family_selectors=family_selectors,
        binder_enumerate=binder_enumerate,
        cell_codec=CellCodec(kind="scalar"),
        noisy_kind="low",
        demand_kind="high",
        link_marker_dir="egress",
        notes="Structurally-different schema (tx_/rx_/if_/dem[] syntax, scalar cells, regex "
              "demand matcher); reuses the engine role vocabulary.",
    )


# --------------------------------------------------------------------------- planted data

def make_benchmark2_frame(n_snapshots: int = 400, n_nodes: int = 4,
                          seed: int = 0) -> Tuple[pd.DataFrame, List[str]]:
    """Generate a random asymmetric demand mesh with conserved transit that plants the laws.

    For each snapshot a non-negative asymmetric demand matrix ``H`` is drawn with a zero
    diagonal.  Every demand ``H[s, d]`` is carried on its direct one-hop link, and a per-
    snapshot *circulation* ``c`` is added to each clockwise ring link ``R_i -> R_{i+1}`` to
    model transit traffic a node forwards but neither originates nor terminates.  Because the
    circulation enters one outgoing and one incoming link per node by the same amount, it
    cancels in every conservation law, so:

    * ``egress[s->d] = ingress[d<-s]``                   (I4 link two-end agreement),
    * ``origination(x) = sum_d H[x, d]``                 (I5 = demand row-sum),
    * ``termination(x) = sum_s H[s, x]``                 (I6 = demand col-sum),
    * node and network conservation hold exactly         (I7, I8; the circulation cancels),
    * ``H[x, x] = 0``                                     (I2 zero self-demand),
    * all values are non-negative                        (I1),
    * ``H`` asymmetric  =>  ``egress[x->y] != egress[y->x]`` (I9 directionality).

    The transit term is what makes this a *faithful* multi-hop benchmark: with pure direct
    routing the non-physical identities ``sum(egress@x) = origination(x)`` and
    ``sum(ingress@x) = termination(x)`` would also hold, and that 2-term subset law makes the
    genuine 4-term conservation law (I7) look like padded bloat to the portfolio assembler,
    which then evicts it.  Real networks have transit (``termination != sum of ingress``), so
    adding the conserved circulation removes the artifact at its source and lets I7 surface --
    exactly as it does on the real Abilene/GEANT data.

    Columns are plain scalars (the ``"scalar"`` codec branch); there is no injected noise.
    """
    rng = np.random.default_rng(seed)
    nodes = [f"R{i + 1}" for i in range(n_nodes)]
    H = {}
    for s in nodes:
        for d in nodes:
            if s == d:
                H[(s, d)] = np.zeros(n_snapshots)
            else:
                H[(s, d)] = rng.uniform(1.0, 100.0, n_snapshots).round(3)

    cols: dict = {}
    # demands (high), including the zero self-demand diagonal.
    for s in nodes:
        for d in nodes:
            cols[f"dem[{s}=>{d}]"] = H[(s, d)]
    # routing: each demand on its own one-hop link.
    for s in nodes:
        for d in nodes:
            if s == d:
                continue
            cols[f"if_{s}_to_{d}_out"] = H[(s, d)]      # egress at s toward d
            cols[f"if_{d}_from_{s}_in"] = H[(s, d)]     # ingress at d from s
    # conserved transit: a per-snapshot circulation on the clockwise ring R_i -> R_{i+1}.
    # It adds the same amount to one outgoing and one incoming link per node, so it cancels in
    # node/network conservation (I7/I8 stay exact) while breaking the direct-routing identities
    # sum(egress@x) = origination(x) and sum(ingress@x) = termination(x) -- the non-physical
    # subset laws that would otherwise mask the 4-term conservation law during assembly.
    circ = rng.uniform(20.0, 50.0, n_snapshots).round(3)
    for i, s in enumerate(nodes):
        d = nodes[(i + 1) % n_nodes]
        cols[f"if_{s}_to_{d}_out"] = cols[f"if_{s}_to_{d}_out"] + circ
        cols[f"if_{d}_from_{s}_in"] = cols[f"if_{d}_from_{s}_in"] + circ
    # node aggregates (origination/termination = demand row/col sums; transit-independent).
    for x in nodes:
        cols[f"tx_{x}_src"] = sum(H[(x, d)] for d in nodes)   # origination = row sum
        cols[f"rx_{x}_snk"] = sum(H[(s, x)] for s in nodes)   # termination = col sum

    df = pd.DataFrame(cols)
    df["timestamp"] = np.arange(n_snapshots)
    return df, nodes


# --------------------------------------------------------------------------- planted targets

# benchmark2-native relabelling of the (reused) catalogue fingerprints.
_B2_LABEL = {
    "I1": "B-I1 non-negativity            v >= 0",
    "I2": "B-I2 zero self-demand          dem[X=>X] = 0",
    "I4": "B-I4 link two-end agreement    egress[X->Y] = ingress_rev[Y<-X]",
    "I5": "B-I5 origination = row-sum     o(X) = SUM dem[X,*]",
    "I6": "B-I6 termination = col-sum     t(X) = SUM dem[*,X]",
    "I7": "B-I7 node flow conservation    o + SUM i = t + SUM e",
    "I8": "B-I8 network totals balance    SUM o = SUM t = SUM dem",
    "I9": "B-I9 directionality anti       egress[X->Y] != egress[Y->X]",
}


def _planted_exact(t: Target) -> Target:
    """Re-key a CrossCheck target for benchmark2: same fingerprint, exact-strictness.

    The keysets/binder/kind are reused by reference (so the grader's notion of "the same
    relation" cannot drift).  benchmark2 plants every law noise-free, so any equality the
    catalogue marks as merely SOFT_STRUCTURAL (I5/I6, the sub-noise structural deficits) is
    here accepted as EXACT as well.
    """
    expect = t.expect
    if t.kind == "eq" and Verdict.EXACT not in expect:
        expect = (Verdict.EXACT,) + expect
    return Target("B-" + t.tid, _B2_LABEL.get(t.tid, t.label),
                  t.binder, t.kind, t.keysets, expect, t.wildcard_single_ge)


BENCH2_TARGETS: Tuple[Target, ...] = tuple(_planted_exact(t) for t in TESTABLE_TARGETS)


# --------------------------------------------------------------------------- the run

def benchmark2_config(n_nodes: int = 4, iterations: int = 200, seed: int = 0) -> RunConfig:
    """A scripted, reproducible RunConfig for the benchmark2 run."""
    rc = RunConfig(dataset="benchmark2", data_path="")
    rc.grammar.proposer = "scripted"            # deterministic; emits enumerate_candidates(G)
    rc.search.iterations = iterations
    rc.search.anti_seeds = True                 # seed the I9 same-family separation niche
    rc.seed = seed
    rc.reseed()
    return rc


def run_benchmark2(n_snapshots: int = 400, n_nodes: int = 4, seed: int = 0,
                   iterations: int = 200):
    """Build the adapter + planted data, run the real engine, and grade against the planted set.

    Returns ``(report, portfolio, dataset, n_evaluated, n_unique)``.
    """
    spec = benchmark2_spec()
    adapter = compile_spec(spec)
    df, _nodes = make_benchmark2_frame(n_snapshots=n_snapshots, n_nodes=n_nodes, seed=seed)
    ds: Dataset = load_dataframe(df, adapter, name="benchmark2")

    rc = benchmark2_config(n_nodes=n_nodes, iterations=iterations, seed=seed)
    proposer = ScriptedProposer()
    res = learn(ds, rc, proposer)
    report: RecallReport = score_against(res.portfolio, BENCH2_TARGETS)
    return report, res.portfolio, ds, res.n_evaluated, res.n_unique
