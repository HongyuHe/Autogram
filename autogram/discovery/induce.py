"""Schema induction from column names (P2): roles are *created*, not chosen from a vocabulary.

A :class:`SchemaInducer` reads column NAMES (and optionally a few sample rows) and emits an open
:class:`~autogram.schema.spec.SchemaSpec` describing the dataset's entities, relations and
families.  The deployed inducer is an LLM (:class:`LLMInducer`); the offline/test path is a
deterministic heuristic (:class:`HeuristicInducer`) so the whole pipeline runs end-to-end with
no network and no key.  Both implement the same ``induce`` interface.

The heuristic inducer assumes a *general*, structured naming convention of the form

    <kind>_<token>_<token>_...

and infers everything from co-occurrence statistics, never from hardcoded spellings:

* **entities** are the high-cardinality tokens that fill *both* slots of a two-token "pair"
  kind (a relation matrix such as ``flow_<src>_<dst>``);
* **keywords** are the low-cardinality tokens that fill the second slot of a single-entity
  measured column (``meas_<entity>_<keyword>``);
* **connectors** are the low-cardinality tokens that sit *between* two entities in a directed
  measured column (``meas_<entity>_<connector>_<entity>``).

Because the spelling of every token (kind, keyword, connector, entity) is discovered rather
than assumed, a consistent rename of the columns induces an equivalent schema -- which is what
makes the discovered invariants rename-invariant.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..schema.adapter import SchemaAdapter
from ..schema.compiler import compile_spec
from ..schema.spec import (
    CellCodec,
    ColumnPattern,
    FamilySelector,
    RefTemplate,
    RoleOntology,
    SchemaSpec,
)


# ---------------------------------------------------------------------------
# Inducer interface
# ---------------------------------------------------------------------------

class SchemaInducer:
    """Interface: column names (+ optional sample rows) -> a :class:`SchemaSpec`."""

    def induce(self, columns: Sequence[str],
               sample_rows: Optional[Sequence[dict]] = None) -> SchemaSpec:  # pragma: no cover
        raise NotImplementedError


class LLMInducer(SchemaInducer):
    """Deployed inducer: ask an LLM to emit the SchemaSpec from names + sample rows.

    A ``responder`` callable (prompt -> JSON spec) is injected so the deployment can plug in any
    model; with no responder this raises, and the offline pipeline uses :class:`HeuristicInducer`
    instead.  Kept thin on purpose: the heuristic inducer is the reference behaviour the LLM is
    expected to reproduce.
    """

    def __init__(self, responder=None):
        self.responder = responder

    def induce(self, columns, sample_rows=None) -> SchemaSpec:  # pragma: no cover
        if self.responder is None:
            raise RuntimeError(
                "LLMInducer needs a responder (LLM); use HeuristicInducer for the offline path")
        import json

        from ..schema import spec as S  # local import to keep parsing logic isolated
        prompt = _llm_prompt(columns, sample_rows)
        payload = json.loads(self.responder(prompt))
        return _spec_from_json(payload, S)


# ---------------------------------------------------------------------------
# Heuristic inducer
# ---------------------------------------------------------------------------

def _tokenize(columns: Sequence[str]) -> Dict[str, List[str]]:
    return {c: c.split("_") for c in columns}


@dataclass
class _Induced:
    measured_kind: str
    demand_kind: Optional[str]
    entities: Tuple[str, ...]
    keywords: Tuple[str, ...]
    connectors: Tuple[str, ...]


class HeuristicInducer(SchemaInducer):
    """Deterministic, network-free schema inducer (the offline reference)."""

    min_entities = 3

    def analyse(self, columns: Sequence[str]) -> _Induced:
        toks = _tokenize(columns)
        usable = {c: t for c, t in toks.items() if len(t) >= 2}
        by_kind: Dict[str, List[List[str]]] = defaultdict(list)
        for c, t in usable.items():
            by_kind[t[0]].append(t[1:])

        # 1. demand kind: a kind whose two-token columns have high cardinality in BOTH slots.
        demand_kind = None
        best_score = 0
        entities: set = set()
        for kind, rows in by_kind.items():
            pairs = [r for r in rows if len(r) == 2]
            if not pairs:
                continue
            pos0 = {r[0] for r in pairs}
            pos1 = {r[1] for r in pairs}
            score = min(len(pos0), len(pos1))
            overlap = len(pos0 & pos1)
            if score >= self.min_entities and overlap >= 1 and score > best_score:
                best_score = score
                demand_kind = kind
                entities = pos0 | pos1

        # Fallback: no clean pair kind -> entities are tokens flanking a connector.
        if demand_kind is None:
            entities = self._entities_from_connectors(by_kind)

        # 2. measured kind: the non-demand kind with the most columns.
        meas_candidates = [(k, len(v)) for k, v in by_kind.items() if k != demand_kind]
        if not meas_candidates:
            measured_kind = demand_kind
        else:
            measured_kind = max(meas_candidates, key=lambda kv: kv[1])[0]

        # 3. keywords + connectors from the measured kind.
        keywords: set = set()
        connectors: set = set()
        for r in by_kind.get(measured_kind, []):
            if len(r) == 2 and r[0] in entities and r[1] not in entities:
                keywords.add(r[1])
            elif len(r) == 3 and r[0] in entities and r[2] in entities and r[1] not in entities:
                connectors.add(r[1])

        return _Induced(
            measured_kind=measured_kind,
            demand_kind=demand_kind,
            entities=tuple(sorted(entities)),
            keywords=tuple(sorted(keywords)),
            connectors=tuple(sorted(connectors)),
        )

    @staticmethod
    def _entities_from_connectors(by_kind) -> set:
        ents: set = set()
        # tokens flanking any middle token in a 3-token column are entity candidates
        flank: Dict[str, int] = defaultdict(int)
        mids: Dict[str, int] = defaultdict(int)
        for rows in by_kind.values():
            for r in rows:
                if len(r) == 3:
                    flank[r[0]] += 1
                    flank[r[2]] += 1
                    mids[r[1]] += 1
        for t, _ in flank.items():
            if t not in mids:
                ents.add(t)
        return ents

    def induce(self, columns, sample_rows=None) -> SchemaSpec:
        info = self.analyse(columns)
        return _build_spec(info)


# ---------------------------------------------------------------------------
# SchemaSpec construction from the induced structure
# ---------------------------------------------------------------------------

def _ent_alt(entities: Sequence[str]) -> str:
    # longest-first alternation so n10 is tried before n1
    return "(?:" + "|".join(re.escape(e) for e in sorted(entities, key=len, reverse=True)) + ")"


def _build_spec(info: _Induced) -> SchemaSpec:
    mk = info.measured_kind
    dk = info.demand_kind
    ent = _ent_alt(info.entities) if info.entities else r"[^_]+"
    patterns: List[ColumnPattern] = []

    # single-entity measured columns: <mk>_<entity>_<keyword>
    for kw in info.keywords:
        patterns.append(ColumnPattern(
            name=f"{mk}_{kw}", matcher="regex", kind="meas", direction=kw,
            regex=fr"^{re.escape(mk)}_(?P<node>{ent})_{re.escape(kw)}$",
            node_groups=("node",), src_group="node", token_groups=("node",)))

    # directed measured columns: <mk>_<A>_<connector>_<B>  (src=A, peer=B)
    for c in info.connectors:
        patterns.append(ColumnPattern(
            name=f"{mk}_{c}", matcher="regex", kind="meas", direction=c,
            regex=fr"^{re.escape(mk)}_(?P<node>{ent})_{re.escape(c)}_(?P<peer>{ent})$",
            node_groups=("node", "peer"), src_group="node", peer_group="peer",
            token_groups=("node", "peer")))

    # demand-matrix columns: <dk>_<src>_<dst>
    if dk is not None:
        patterns.append(ColumnPattern(
            name=f"{dk}_pair", matcher="regex", kind="demand", direction="demand",
            regex=fr"^{re.escape(dk)}_(?P<src>{ent})_(?P<dst>{ent})$",
            node_groups=("src", "dst"), src_group="src", dst_group="dst",
            token_groups=("src", "dst")))

    # --- ontology ----------------------------------------------------------
    kw_roles = {kw: f"m_{kw}" for kw in info.keywords}
    binders = ["cell", "node", "network"]
    ref_roles: Dict[str, Tuple[str, ...]] = {"cell": ("self",)}
    fam_roles: Dict[str, Tuple[str, ...]] = {"cell": ()}

    node_refs = [kw_roles[kw] for kw in info.keywords]
    node_fams: List[str] = []
    if dk is not None:
        node_refs.append("demand_self")
        node_fams += ["demand_row", "demand_col"]
    for c in info.connectors:
        node_fams.append(f"fam_{c}")
    ref_roles["node"] = tuple(node_refs)
    fam_roles["node"] = tuple(node_fams)

    link_refs: List[str] = []
    for j, c in enumerate(info.connectors):
        link_refs += [f"o{j}", f"o{j}_rev"]
    if dk is not None:
        link_refs += ["demand", "demand_rev"]
    if info.connectors:
        binders.append("link")
        ref_roles["link"] = tuple(link_refs)
        fam_roles["link"] = ()

    net_fams = [f"all_{kw_roles[kw]}" for kw in info.keywords]
    if dk is not None:
        net_fams.append("all_demand")
    ref_roles["network"] = ()
    fam_roles["network"] = tuple(net_fams)

    onto = RoleOntology(binders=tuple(binders), ref_roles=ref_roles, fam_roles=fam_roles)

    # --- ref templates -----------------------------------------------------
    ref_templates: List[RefTemplate] = [RefTemplate("cell", "self", "{col}")]
    for kw in info.keywords:
        ref_templates.append(RefTemplate("node", kw_roles[kw], f"{mk}_{{X}}_{kw}"))
    if dk is not None:
        ref_templates.append(RefTemplate("node", "demand_self", f"{dk}_{{X}}_{{X}}"))
    for j, c in enumerate(info.connectors):
        ref_templates.append(RefTemplate("link", f"o{j}", f"{mk}_{{X}}_{c}_{{Y}}"))
        ref_templates.append(RefTemplate("link", f"o{j}_rev", f"{mk}_{{Y}}_{c}_{{X}}"))
    if dk is not None and info.connectors:
        ref_templates.append(RefTemplate("link", "demand", f"{dk}_{{X}}_{{Y}}"))
        ref_templates.append(RefTemplate("link", "demand_rev", f"{dk}_{{Y}}_{{X}}"))

    # --- family selectors --------------------------------------------------
    selectors: List[FamilySelector] = []
    if dk is not None:
        selectors.append(FamilySelector("node", "demand_row", "demand", "demand",
                                        (("src", "==", "X"), ("dst", "!=", "X"))))
        selectors.append(FamilySelector("node", "demand_col", "demand", "demand",
                                        (("dst", "==", "X"), ("src", "!=", "X"))))
    for c in info.connectors:
        selectors.append(FamilySelector("node", f"fam_{c}", "meas", c, (("src", "==", "X"),)))
    for kw in info.keywords:
        selectors.append(FamilySelector("network", f"all_{kw_roles[kw]}", "meas", kw, ()))
    if dk is not None:
        selectors.append(FamilySelector("network", "all_demand", "demand", "demand",
                                        (("src", "!=", "@dst"),)))

    # --- binder enumeration ------------------------------------------------
    binder_enumerate = {"cell": "per_measured_col", "node": "per_node", "network": "singleton"}
    if info.connectors:
        binder_enumerate["link"] = "per_directed_link"

    link_marker = info.connectors[0] if info.connectors else "demand"
    return SchemaSpec(
        name="induced",
        patterns=tuple(patterns),
        ontology=onto,
        ref_templates=tuple(ref_templates),
        family_selectors=tuple(selectors),
        binder_enumerate=binder_enumerate,
        cell_codec=CellCodec(kind="scalar"),
        noisy_kind="meas",
        demand_kind="demand",
        link_marker_dir=link_marker,
        notes="induced from column names by HeuristicInducer",
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def induce_spec(columns: Sequence[str], inducer: Optional[SchemaInducer] = None,
                sample_rows=None) -> SchemaSpec:
    inducer = inducer or HeuristicInducer()
    return inducer.induce(columns, sample_rows)


def induce_adapter(columns: Sequence[str], inducer: Optional[SchemaInducer] = None,
                   sample_rows=None) -> SchemaAdapter:
    """Induce a spec from names and compile it to a runnable adapter."""
    return compile_spec(induce_spec(columns, inducer, sample_rows))


# ---------------------------------------------------------------------------
# LLM glue (only exercised on the deployed path)
# ---------------------------------------------------------------------------

def _llm_prompt(columns, sample_rows):  # pragma: no cover
    head = "\n".join(columns[:200])
    return ("Induce a SchemaSpec (entities, relations, families) from these column names. "
            "Return JSON matching autogram.schema.spec.SchemaSpec.\n\nColumns:\n" + head)


def _spec_from_json(payload, S):  # pragma: no cover
    onto = payload["ontology"]
    return S.SchemaSpec(
        name=payload.get("name", "induced"),
        patterns=tuple(S.ColumnPattern(**p) for p in payload["patterns"]),
        ontology=S.RoleOntology(
            binders=tuple(onto["binders"]),
            ref_roles={k: tuple(v) for k, v in onto["ref_roles"].items()},
            fam_roles={k: tuple(v) for k, v in onto["fam_roles"].items()}),
        ref_templates=tuple(S.RefTemplate(**t) for t in payload["ref_templates"]),
        family_selectors=tuple(
            S.FamilySelector(
                binder=s["binder"], family_role=s["family_role"], match_kind=s["match_kind"],
                match_direction=s.get("match_direction"),
                predicates=tuple(tuple(p) for p in s.get("predicates", ())))
            for s in payload["family_selectors"]),
        binder_enumerate=dict(payload["binder_enumerate"]),
        cell_codec=S.CellCodec(**payload.get("cell_codec", {"kind": "scalar"})),
        noisy_kind=payload.get("noisy_kind", "meas"),
        demand_kind=payload.get("demand_kind", "demand"),
        link_marker_dir=payload.get("link_marker_dir", "demand"),
    )
