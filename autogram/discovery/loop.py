"""Discovery loop: LLM schema induction -> exhaustive enumeration -> solver/data evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import DiscoveryConfig, SearchConfig
from ..dsl.binders import (
    enumerate_bindings,
    resolve_family,
    resolve_ref,
)
from ..dsl.evaluate import typed_group_key, typed_unique
from ..dsl.grammar import Grammar, grammar_from_adapter
from ..loader.loader import Dataset, build_dataset, load_dataframe
from ..schema.compiler import compile_spec
from ..schema.spec import ColumnPattern, FamilySelector, RefTemplate, RelatedTemplate
from .archive import ParetoArchive
from .evaluate import DataOnlyEvaluator, Evaluation
from .induce import SchemaInducer, induce_spec, make_inducer
from .propose import EnumerationProposer


@dataclass
class DiscoveryResult:
    portfolio: List[Evaluation]
    archive: ParetoArchive
    dataset: Dataset
    grammar: Grammar
    rounds_run: int
    progress_history: List[float] = field(default_factory=list)
    reinductions: int = 0
    diagnostics: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Discovered invariants on {self.dataset.name!r} "
                 f"({len(self.portfolio)} accepted, exhaustive enumeration):", "-" * 78]
        adapter = getattr(self.dataset.name_model, "adapter", None)
        for ev in self.portfolio:
            lines.append("  " + ev.summary(adapter))
        if not self.portfolio:
            lines.append("  (none)")
        if self.diagnostics:
            lines.append("-" * 78)
            lines.append("  diagnostics:")
            for msg in self.diagnostics:
                lines.append("  - " + msg)
        lines.append("-" * 78)
        lines.append("  statistic: hold-rate with Wilson confidence interval")
        return "\n".join(lines)


def _dataset_from_columns(columns: Sequence[str], matrix: np.ndarray, adapter, name: str, timestamps=None) -> Dataset:
    return build_dataset(columns, matrix, adapter, name, timestamps)


def _runtime_column_roles(ds: Dataset, G: Grammar) -> tuple[tuple[object, ...], ...]:
    pairs = []
    seen = set()
    for binder in G.binders:
        bindings = enumerate_bindings(binder, ds.name_model)
        for role in G.refs_for(binder):
            for binding in bindings:
                column = resolve_ref(
                    role,
                    binder,
                    binding,
                    ds.name_model,
                )
                if column is None:
                    continue
                key = (
                    binder,
                    typed_group_key(column),
                    role,
                )
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((binder, column, role))
    return tuple(pairs)


def _attach_runtime_column_roles(ds: Dataset, G: Grammar) -> Grammar:
    return replace(
        G,
        column_roles=_runtime_column_roles(ds, G),
    )


def _validate_runtime_role_groundings(ds: Dataset, G: Grammar) -> None:
    """Fail before enumeration when numeric grammar roles resolve only to context."""
    observed = set(ds.observed.names)
    invalid = []
    for binder in G.binders:
        bindings = enumerate_bindings(binder, ds.name_model)
        for role in G.refs_for(binder):
            for binding in bindings:
                column = resolve_ref(
                    role,
                    binder,
                    binding,
                    ds.name_model,
                )
                if column is not None and column not in observed:
                    invalid.append(
                        f"ref {binder}/{role} -> {column!r}"
                    )
        for role in G.fams_for(binder):
            for binding in bindings:
                for column in resolve_family(
                    role,
                    binder,
                    binding,
                    ds.name_model,
                ):
                    if column not in observed:
                        invalid.append(
                            f"family {binder}/{role} -> {column!r}"
                        )
    if invalid:
        raise ValueError(
            "numeric grammar role groundings are absent from the "
            "observed frame: "
            + "; ".join(dict.fromkeys(invalid[:8]))
        )


def _make_proposer(ds: Dataset, G: Grammar, scfg: SearchConfig):
    if scfg.proposer != "enumeration":
        raise ValueError("v2 supports only proposer='enumeration'")
    return EnumerationProposer(
        G,
        column_roles=(
            G.column_roles
            or _runtime_column_roles(ds, G)
        ),
    )


def _run_dataset(ds: Dataset, G: Grammar, *, proposer, dcfg: DiscoveryConfig, scfg: SearchConfig) -> DiscoveryResult:
    _validate_runtime_role_groundings(ds, G)
    evaluator = DataOnlyEvaluator(ds, dcfg)
    proposer_obj = proposer or _make_proposer(ds, G, scfg)
    logically_screened = isinstance(proposer_obj, EnumerationProposer)
    archive = ParetoArchive(legacy_compat=G.legacy_compat)
    candidates = proposer_obj.propose(0, (), None)
    diagnostics: List[str] = []
    seen_diagnostics = set()
    for rule in candidates:
        ev = evaluator.evaluate(rule, logically_screened=logically_screened)
        if "grounded 0 points" in ev.reason:
            key = (ev.rule.binder, ev.reason)
            if key not in seen_diagnostics and len(diagnostics) < 20:
                diagnostics.append(f"{ev.rule.unparse()}: {ev.reason}")
                seen_diagnostics.add(key)
        archive.add(ev)
    portfolio = archive.portfolio(non_redundant=True)
    return DiscoveryResult(
        portfolio=portfolio, archive=archive, dataset=ds, grammar=G,
        rounds_run=1, progress_history=[archive.progress()], reinductions=0,
        diagnostics=diagnostics)


def discover(columns: Sequence[str], matrix: np.ndarray, *,
             inducer: Optional[SchemaInducer] = None,
             proposer=None,
             llm_responder=None,
             discovery_cfg: Optional[DiscoveryConfig] = None,
             search_cfg: Optional[SearchConfig] = None,
             name: str = "synthetic", timestamps=None,
             sample_rows=None) -> DiscoveryResult:
    """Run guarantees-first discovery from an observed numeric matrix."""
    dcfg = discovery_cfg or DiscoveryConfig()
    scfg = search_cfg or SearchConfig()
    if inducer is None:
        inducer = make_inducer("subagent", responder=llm_responder) if llm_responder is not None else make_inducer("subagent")
    ds, G, _spec = prepare_columns(columns, matrix, inducer=inducer, search_cfg=scfg,
                                   name=name, timestamps=timestamps, sample_rows=sample_rows)
    return run_prepared(ds, G, discovery_cfg=dcfg, search_cfg=scfg, proposer=proposer)


def prepare_columns(columns: Sequence[str], matrix: np.ndarray, *,
                    inducer: Optional[SchemaInducer] = None,
                    search_cfg: Optional[SearchConfig] = None,
                    name: str = "synthetic", timestamps=None, sample_rows=None):
    """Induce the schema and build ``(dataset, grammar, spec)`` once from a numeric matrix."""
    scfg = search_cfg or SearchConfig()
    inducer = inducer or make_inducer("subagent")
    spec = induce_spec(columns, inducer, sample_rows)
    spec = _apply_search_temporal_bounds(spec, scfg)
    adapter = compile_spec(spec)
    ds = _dataset_from_columns(columns, matrix, adapter, name, timestamps)
    G = grammar_from_adapter(
        adapter,
        scfg.max_complexity,
        scfg.max_add_arity,
        max_rules=scfg.max_rules,
        max_nonlinear_leaves=scfg.max_nonlinear_leaves,
        max_linear_leaves=scfg.max_linear_leaves,
        max_conditioned_rules=scfg.max_conditioned_rules,
    )
    G = _attach_runtime_column_roles(ds, G)
    _validate_runtime_role_groundings(ds, G)
    return ds, G, spec


def discover_dataframe(df, *, inducer: Optional[SchemaInducer] = None,
                       proposer=None, discovery_cfg: Optional[DiscoveryConfig] = None,
                       search_cfg: Optional[SearchConfig] = None,
                       name: str = "dataframe") -> DiscoveryResult:
    """Run discovery on a pandas DataFrame, decoding only observed ``ground_truth`` values."""
    dcfg = discovery_cfg or DiscoveryConfig()
    scfg = search_cfg or SearchConfig()
    ds, G, _spec = prepare_dataframe(df, inducer=inducer, search_cfg=scfg, name=name)
    return run_prepared(ds, G, discovery_cfg=dcfg, search_cfg=scfg, proposer=proposer)


def prepare_dataframe(df, *, inducer: Optional[SchemaInducer] = None,
                      search_cfg: Optional[SearchConfig] = None, name: str = "dataframe"):
    """Induce the schema and build ``(dataset, grammar, spec)`` once.

    Separating induction (one expensive/non-deterministic LLM call) from evaluation lets a
    calibration loop re-score the *same* induced grammar under many generic-knob settings
    (tolerance/threshold/band) without re-inducing -- the schema is a property of the dataset's
    column names, not of the numeric thresholds being tuned.
    """
    scfg = search_cfg or SearchConfig()
    inducer = inducer or make_inducer("subagent")
    induced_spec = induce_spec(
        list(df.columns),
        inducer,
        sample_rows=None,
    )
    runtime_spec = normalize_dataframe_spec(
        df,
        induced_spec,
        scfg,
    )
    ds, G = build_dataframe_grammar(
        df,
        runtime_spec,
        search_cfg=scfg,
        name=name,
    )
    return ds, G, runtime_spec


def build_dataframe_grammar(df, spec, *, search_cfg: Optional[SearchConfig] = None,
                            name: str = "dataframe"):
    """Compile a (possibly capability-widened) spec into ``(dataset, grammar)`` -- no induction.

    Used by the calibration loop's grammar re-induction tier: after a fresh induction is widened
    (more aggregations / higher degree), this rebuilds the runnable grammar from the edited spec.
    """
    scfg = search_cfg or SearchConfig()
    spec = normalize_dataframe_spec(df, spec, scfg)
    adapter = compile_spec(spec)
    timestamps = df["timestamp"].values if "timestamp" in df.columns else None
    ds = load_dataframe(df, adapter, name, timestamps=timestamps)
    G = grammar_from_adapter(
        adapter,
        scfg.max_complexity,
        scfg.max_add_arity,
        max_rules=scfg.max_rules,
        max_nonlinear_leaves=scfg.max_nonlinear_leaves,
        max_linear_leaves=scfg.max_linear_leaves,
        max_conditioned_rules=scfg.max_conditioned_rules,
    )
    G = _attach_runtime_column_roles(ds, G)
    _validate_runtime_role_groundings(ds, G)
    return ds, G


def normalize_dataframe_spec(
    df,
    spec,
    search_cfg: Optional[SearchConfig] = None,
):
    """Return the exact post-profile, post-search spec used at runtime."""

    scfg = search_cfg or SearchConfig()
    return _apply_search_temporal_bounds(
        _augment_profiled_dataframe_spec(df, spec),
        scfg,
    )


def _apply_search_temporal_bounds(spec, search_cfg: SearchConfig):
    updates = {}
    if int(search_cfg.max_lag) > 0:
        updates["max_lag"] = int(search_cfg.max_lag)
    if search_cfg.windows:
        updates["windows"] = tuple(sorted({
            int(window) for window in search_cfg.windows
        }))
    return replace(spec, **updates) if updates else spec


def _augment_profiled_dataframe_spec(df, spec):
    """Add a generic singleton record binder from explicit DataFrame profile metadata."""

    from ..loader.gtib import AUTOGRAM_PROFILE_ATTR

    profile = getattr(df, "attrs", {}).get(
        AUTOGRAM_PROFILE_ATTR
    )
    if not profile:
        return spec

    time_index = str(profile.get("time_index") or "")
    group_keys = tuple(str(c) for c in profile.get("group_keys", ()) if c in df.columns)
    condition_names = tuple(
        str(c) for c in profile.get("condition_columns", ()) if c in df.columns
    )
    metadata_columns = tuple(
        str(column)
        for column in profile.get("metadata_columns", ())
        if column in df.columns
    )
    metadata = {time_index, *group_keys, *condition_names, *metadata_columns} - {""}
    family_members = {
        str(column)
        for columns in profile.get("families", {}).values()
        for column in columns
    }
    numeric_columns = [
        str(c) for c in df.columns
        if (c not in metadata or pd.api.types.is_bool_dtype(df[c]))
        and (
            pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
        )
    ]
    measured = [
        column for column in numeric_columns
        if column not in family_members
    ]
    if not measured:
        raise ValueError("profiled DataFrame contains no numeric or Boolean measured columns")

    def role_name(column: str, used: set[str]) -> str:
        base = re.sub(r"\W+", "_", column).strip("_") or "value"
        if base[0].isdigit():
            base = "v_" + base
        role = base
        serial = 2
        while role in used:
            role = f"{base}_{serial}"
            serial += 1
        used.add(role)
        return role

    onto = spec.ontology
    binders = ("record",)
    used: set[str] = set()
    roles: list[str] = []
    patterns = list(spec.patterns)
    templates = []
    existing_patterns = {p.name for p in patterns}
    existing_templates = {(t.binder, t.role) for t in templates}
    column_roles: dict[str, str] = {}
    for index, column in enumerate(numeric_columns):
        existing = next(
            (
                t.role for t in templates
                if t.binder == "record" and t.template == column
            ),
            None,
        )
        role = existing or role_name(column, used)
        column_roles[column] = role
        if column in measured and role not in roles:
            roles.append(role)
        pattern_name = f"tabular_exact_{index}_{role}"
        if pattern_name not in existing_patterns:
            patterns.append(ColumnPattern(
                name=pattern_name,
                matcher="regex",
                kind="tabular",
                direction=role,
                regex=rf"^{re.escape(column)}$",
            ))
        if column in measured and ("record", role) not in existing_templates:
            templates.append(RefTemplate("record", role, column))

    selectors = []
    family_roles = []
    existing_selectors = {(s.binder, s.family_role) for s in selectors}
    for family_name, columns in profile.get("families", {}).items():
        role = re.sub(r"\W+", "_", str(family_name)).strip("_") or "family"
        if role not in family_roles:
            family_roles.append(role)
        if ("record", role) not in existing_selectors:
            selectors.append(FamilySelector(
                binder="record",
                family_role=role,
                match_kind="tabular",
                columns=tuple(str(c) for c in columns if c in column_roles),
            ))

    ref_roles = {"record": tuple(roles)}
    fam_roles = {"record": tuple(family_roles)}
    profile_agg_kinds = tuple(
        str(kind)
        for kind in profile.get("agg_kinds", ())
    )
    agg_kinds = (
        tuple(dict.fromkeys((
            *onto.agg_kinds,
            *profile_agg_kinds,
        )))
        if getattr(spec, "aggregations_widened", False)
        else profile_agg_kinds or tuple(onto.agg_kinds)
    )
    pinned = bool(
        spec.temporal_bounds_widened
        or spec.advanced_bounds_widened
        or spec.degree_widened
    )
    ontology = replace(
        onto,
        binders=binders,
        ref_roles=ref_roles,
        fam_roles=fam_roles,
        ops=tuple(dict.fromkeys(
            tuple(
                op
                for op in onto.ops
                if op not in ("<", ">", "~∝")
            )
            + (("<", ">") if time_index else ())
            + (
                ("~∝",)
                if (
                    (
                        profile.get("proportional", False)
                        and not pinned
                    )
                    or getattr(
                        spec,
                        "proportional_widened",
                        False,
                    )
                )
                else ()
            )
        )),
        agg_kinds=agg_kinds,
    )
    condition_columns = {
        name: typed_unique(
            df[name].to_numpy(dtype=object),
            drop_missing=True,
        )
        for name in condition_names
    }
    boolean_roles = {
        "record": tuple(
            column_roles[column]
            for column in measured
            if pd.api.types.is_bool_dtype(df[column])
        ),
    }
    related_templates = []
    existing_related = set()
    for role, raw in profile.get("related_aggregates", {}).items():
        key = ("record", str(role))
        if key in existing_related:
            continue
        related_templates.append(RelatedTemplate(
            binder="record",
            role=str(role),
            relation=str(raw["relation"]),
            column=str(raw["column"]),
            mode=str(raw["mode"]),
            parent_keys=tuple(str(value) for value in raw.get("parent_keys", ())),
            child_keys=tuple(str(value) for value in raw.get("child_keys", ())),
            partition_keys=tuple(str(value) for value in raw.get("partition_keys", ())),
            parent_time=str(raw["parent_time"]),
            child_time=str(raw["child_time"]),
            window_seconds=int(raw["window_seconds"]),
            reset_column=str(raw.get("reset_column", "")),
            validity_columns=tuple(str(value) for value in raw.get("validity_columns", ())),
            span_start=str(raw.get("span_start", "")),
            span_end=str(raw.get("span_end", "")),
            filter_column=str(raw.get("filter_column", "")),
            filter_values=tuple(raw.get("filter_values", ())),
        ))
    advanced_enabled = (
        bool(spec.advanced_enabled)
        if pinned
        else bool(profile.get("advanced", False))
    )
    return replace(
        spec,
        patterns=tuple(patterns),
        ontology=ontology,
        ref_templates=tuple(templates),
        family_selectors=tuple(selectors),
        binder_enumerate={"record": "singleton"},
        cell_codec=replace(spec.cell_codec, kind="scalar"),
        time_index=time_index,
        group_keys=group_keys,
        condition_columns=condition_columns,
        conditional_enabled=bool(
            condition_columns
            and (
                spec.conditional_enabled
                if pinned
                else True
            )
        ),
        temporal_enabled=bool(
            time_index
            and (
                spec.temporal_enabled
                if pinned
                else True
            )
        ),
        max_lag=(
            max(
                int(getattr(spec, "max_lag", 0)),
                int(profile.get("max_lag", 0)),
            )
            if getattr(spec, "temporal_bounds_widened", False)
            else int(profile.get("max_lag", 0))
        ),
        windows=(
            tuple(sorted({
                *(
                    int(window)
                    for window in getattr(spec, "windows", ())
                ),
                *(
                    int(window)
                    for window in profile.get(
                        "temporal_windows",
                        (),
                    )
                ),
            }))
            if getattr(spec, "temporal_bounds_widened", False)
            else tuple(sorted({
                int(window)
                for window in profile.get(
                    "temporal_windows",
                    (),
                )
            }))
        ),
        related_templates=(
            tuple(related_templates)
            if advanced_enabled
            else ()
        ),
        boolean_roles=boolean_roles,
        role_exclusions=tuple(
            exclusion
            for exclusion in getattr(spec, "role_exclusions", ())
            if set(exclusion) <= set(roles)
        ),
        advanced_enabled=advanced_enabled,
        run_lengths=(
            tuple(sorted({
                *(
                    int(window)
                    for window in getattr(spec, "run_lengths", ())
                ),
                *(
                    int(window)
                    for window in profile.get("run_lengths", ())
                ),
            }))
            if getattr(spec, "advanced_bounds_widened", False)
            else tuple(sorted({
                int(window)
                for window in profile.get("run_lengths", ())
            }))
        ),
        max_conjunction_terms=(
            max(
                int(getattr(spec, "max_conjunction_terms", 3)),
                int(profile.get("max_conjunction_terms", 3)),
            )
            if getattr(spec, "advanced_bounds_widened", False)
            else max(
                2,
                int(profile.get("max_conjunction_terms", 3)),
            )
        ),
        metadata_columns=metadata_columns,
        band_enabled=(
            bool(spec.band_enabled)
            if pinned
            else bool(profile.get("band_enabled", False))
        ),
        max_degree=(
            max(1, int(spec.max_degree))
            if pinned
            else max(1, int(profile.get("max_degree", 0)))
        ),
    )


def run_prepared(ds: Dataset, G: Grammar, *, discovery_cfg: Optional[DiscoveryConfig] = None,
                 search_cfg: Optional[SearchConfig] = None, proposer=None) -> DiscoveryResult:
    """Enumerate + evaluate on an already-prepared ``(dataset, grammar)`` (no induction)."""
    dcfg = discovery_cfg or DiscoveryConfig()
    scfg = search_cfg or SearchConfig()
    return _run_dataset(ds, G, proposer=proposer, dcfg=dcfg, scfg=scfg)


def discover_synthetic(synth, *, discovery_cfg=None, search_cfg=None,
                       inducer=None, proposer=None, llm_responder=None) -> DiscoveryResult:
    return discover(synth.columns, synth.matrix, inducer=inducer, proposer=proposer,
                    llm_responder=llm_responder, discovery_cfg=discovery_cfg,
                    search_cfg=search_cfg, name="synthetic", timestamps=synth.timestamps)
