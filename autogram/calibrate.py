"""Publishing: the calibration orchestrator (`autogram calibrate`) and preflight (`autogram precheck`).

``calibrate`` runs the whole loop to completion. It tunes the engine's generic knobs on a wired,
editable synthetic-proxy suite (a `RegimeSpec`), (re-)proposes a grammar, iterates a generic-knob
relaxation ladder (one fixed *global* band by default), and, when recall **stalls**,
re-induces the grammar with **widened capabilities** (more aggregations, then products/ratios).
It reports recovery of the user's known invariants -- recall on a held-out validation split so it
cannot be fit to. The objective is recall *subject to* a false-discovery ceiling (null
acceptance), never recall alone.
"""

from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import DiscoveryConfig, SearchConfig
from .discovery.export import write_rules_dl
from .discovery.induce import induce_spec, make_inducer
from .discovery.induce import _spec_to_json
from .discovery.known import KnownInvariant, abstract_shapes, load_known, recover_known
from .discovery.known import _signature as _known_signature
from .discovery.known import _matching_signatures as _known_matching_signatures
from .discovery.known import _canonicalize as _known_canonicalize
from .discovery.loop import (
    build_dataframe_grammar,
    normalize_dataframe_spec,
    run_prepared,
)
from .discovery.propose import EnumerationProposer
from .discovery.regime import RegimeSpec, abstract_from_shapes
from .discovery.subagent import HARNESSES, configured_harness
from .discovery.validate import (
    CalibrationGridError, null_definitions_at, null_equalities_at, null_temporal_at,
    prepare_proxy_suite, prepare_runtime_null_controls, relation_signature_matches, tune_joint,
)
from .schema.compiler import compile_spec


@dataclass
class CalibrationConfig:
    seed: int = 0
    max_iterations: int = 0              # 0 = full knob ladder per grammar tier (default)
    validation_frac: float = 0.3         # held-out split for honest recall
    harness: str = field(default_factory=configured_harness)
    backend: str = "subagent"
    null_floor: float = 0.5              # threshold never drops below the false-discovery floor
    band_mode: str = "global"            # DEFAULT for calibration: one fixed global tolerance; "adaptive" = per-candidate self-calibrated band. (The engine's own DiscoveryConfig default stays "adaptive".)
    ci_alpha: float = 0.05
    tolerance: Optional[float] = None
    hold_rate_threshold: Optional[float] = None
    max_capability_tiers: int = 5        # grammar re-induction tiers when recall stalls
    regime: Optional[RegimeSpec] = None  # wired, editable synthetic-proxy suite (item 7)
    save_rules: bool = True              # persist the learned portfolio to <rules_dir>/<name>_<ts>.dl
    rules_dir: str = "rules"
    max_complexity: int = 10
    max_add_arity: int = 2
    max_rules: int = 500_000
    max_nonlinear_leaves: int = 0
    max_linear_leaves: int = 0
    max_conditioned_rules: int = 0
    max_lag: int = 0
    windows: tuple[int, ...] = ()


def _git_short() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _update_fingerprint(
    digest,
    value,
    seen: set[int] | None = None,
) -> None:
    seen = seen if seen is not None else set()
    if value is None or isinstance(value, (bool, int, str)):
        digest.update(f"{type(value).__name__}:{value!r};".encode())
        return
    if isinstance(value, float):
        text = (
            "nan"
            if math.isnan(value)
            else "inf"
            if value == math.inf
            else "-inf"
            if value == -math.inf
            else value.hex()
        )
        digest.update(f"float:{text};".encode())
        return
    if isinstance(value, np.generic):
        _update_fingerprint(digest, value.item(), seen)
        return

    identity = id(value)
    if identity in seen:
        digest.update(b"<cycle>;")
        return
    seen.add(identity)
    try:
        if is_dataclass(value):
            digest.update(
                f"dataclass:{type(value).__qualname__};".encode()
            )
            _update_fingerprint(digest, asdict(value), seen)
        elif isinstance(value, pd.DataFrame):
            digest.update(b"dataframe;")
            _update_fingerprint(
                digest,
                [str(column) for column in value.columns],
                seen,
            )
            _update_fingerprint(
                digest,
                [str(dtype) for dtype in value.dtypes],
                seen,
            )
            _update_fingerprint(
                digest,
                value.index.to_numpy(),
                seen,
            )
            for column in value.columns:
                _update_fingerprint(
                    digest,
                    value[column].to_numpy(),
                    seen,
                )
            _update_fingerprint(digest, dict(value.attrs), seen)
        elif isinstance(value, np.ndarray):
            digest.update(
                f"ndarray:{value.dtype.str}:{value.shape};".encode()
            )
            if value.dtype.hasobject:
                for item in value.flat:
                    _update_fingerprint(digest, item, seen)
            else:
                digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict;")
            for key in sorted(value, key=lambda item: repr(item)):
                _update_fingerprint(digest, key, seen)
                _update_fingerprint(digest, value[key], seen)
        elif isinstance(value, (list, tuple)):
            digest.update(f"{type(value).__name__};".encode())
            for item in value:
                _update_fingerprint(digest, item, seen)
        elif isinstance(value, (set, frozenset)):
            digest.update(f"{type(value).__name__};".encode())
            for item in sorted(value, key=repr):
                _update_fingerprint(digest, item, seen)
        elif hasattr(value, "__dict__"):
            digest.update(
                f"object:{type(value).__qualname__};".encode()
            )
            _update_fingerprint(digest, vars(value), seen)
        else:
            digest.update(
                f"{type(value).__qualname__}:{value!r};".encode()
            )
    finally:
        seen.remove(identity)


def _fingerprint(value) -> str:
    digest = hashlib.sha256()
    _update_fingerprint(digest, value)
    return digest.hexdigest()


def _json_normalize(value):
    """Round-trip a value through JSON so it holds only JSON-native types (tuples become lists).

    This makes the serialized ``normalized_spec`` and its ``_json_fingerprint`` agree exactly with
    what a consumer reads back from the report.
    """
    return json.loads(json.dumps(value, ensure_ascii=False))


def _json_fingerprint(value) -> str:
    """Hash a JSON-native value by its canonical serialization.

    Unlike :func:`_fingerprint` (which distinguishes tuples from lists and is meant for internal
    objects), this hashes the exact bytes ``json.dumps(value, sort_keys=True)`` would produce, so a
    consumer can recompute the digest directly from the ``normalized_spec`` a report serializes.
    """
    canonical = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _engine_source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _calibration_provenance(df, known, cfg) -> dict[str, str]:
    return {
        "engine_source_sha256": _engine_source_fingerprint(),
        "input_sha256": _fingerprint(df),
        "known_sha256": _fingerprint(known),
        "calibration_config_sha256": _fingerprint(cfg),
    }


def precheck(harness: str | None = None, backend: str = "subagent") -> dict:
    """Validate the runtime can induce schemas before a calibration run (`autogram precheck`)."""
    harness = (
        configured_harness()
        if harness is None
        else str(harness).strip().lower()
    )
    issues: List[str] = []
    spec = HARNESSES.get(harness)
    if backend == "subagent":
        if spec is None:
            issues.append(f"unknown harness {harness!r}; choose from {sorted(HARNESSES)}")
        else:
            cmd = os.environ.get("AUTOGRAM_SUBAGENT_COMMAND", spec.command)
            if shutil.which(cmd) is None:
                issues.append(
                    f"subagent CLI {cmd!r} not found on PATH; install/authenticate it or use "
                    f"--schema-backend openai with OPENAI_API_KEY")
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY is not set for the openai backend")
    probe = "skipped"
    if not issues:
        try:
            inducer = (
                make_inducer("subagent", harness=harness)
                if backend == "subagent"
                else make_inducer(backend)
            )
            induced = inducer.induce((
                "measurement_n0_source",
                "measurement_n0_destination",
                "flow_n0_n0",
            ))
            compile_spec(induced)
            probe = "induced-and-compiled"
        except Exception as exc:
            issues.append(f"minimal induction probe failed: {exc}")
            probe = "failed"
    return {
        "ok": not issues,
        "issues": issues,
        "harness": harness,
        "backend": backend,
        "probe": probe,
    }


class _ColumnScaleView:
    """Minimal frame-like view used only for signature canonicalisation.

    `known._canonicalize` drops summed members whose observed data is negligible against the
    anchor's scale, which is a *data-dependent* transform. The split has to apply the same
    transform, or two catalogue entries that recovery cannot tell apart -- `total == SUM(a)` and
    `total == SUM(a, z)` with `z` identically zero -- can land on opposite sides and the validation
    half stops being held out. Only `has` and `col` are needed for that, so a full `Frame` (which is
    not built until after the split) is unnecessary.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def has(self, column: str) -> bool:
        return column in self._df.columns

    def col(self, column: str) -> np.ndarray:
        return pd.to_numeric(self._df[column], errors="coerce").to_numpy(dtype=float)


def _split_known(known: List[KnownInvariant], frac: float, seed: int,
                 frame=None, zero_tol: float = 1e-4):
    """Partition known invariants into a calibration set and a structurally disjoint validation set.

    The split is by *recovery equivalence*, not by list position and not by exact signature. Two
    catalogue entries can denote the same relation in three ways: they can be literal aliases
    (``x == y`` and ``y == x`` canonicalise identically), or they can differ only in a numeric
    threshold by less than the matcher's tolerance, in which case a single discovered rule recovers
    both. Either way, splitting them apart would mean tuning on a "held-out" invariant: the
    calibration entry would drag its twin across with it, and the calibration/validation gap would
    stop being an overfitting alarm.

    Equivalence is therefore decided by `relation_signature_matches` -- the very predicate
    `recover_known` uses -- and entries are grouped into connected components under it, because
    tolerance-based matching is not transitive and a chain of near-identical thresholds must still
    travel together.
    """
    if len(known) < 2:
        raise ValueError(
            "calibration requires at least two known invariants "
            "for a disjoint held-out validation split"
        )
    if not 0.0 < float(frac) < 1.0:
        raise ValueError("validation_frac must be strictly between 0 and 1")
    signatures = [_known_signature(invariant) for invariant in known]
    # `recover_known` does not compare signatures literally: an *approximate* equality is also
    # satisfied by the exact rule that recovers it (`known._matching_signatures`). Two catalogue
    # entries written as `x ~= y` and `x == y` are therefore recovered by one and the same rule, so
    # the split has to see them as the same relation even though their signatures differ. Compare
    # the expanded candidate sets, exactly as recovery does.
    expansions = [
        None if signature is None else _known_matching_signatures(signature)
        for signature in signatures
    ]
    if frame is not None:
        # Canonicalise exactly as `recover_known` does, so two entries it would satisfy with one
        # rule are grouped together here too.
        expansions = [
            None if candidates is None
            else [_known_canonicalize(candidate, frame, zero_tol) for candidate in candidates]
            for candidates in expansions
        ]
    parent = list(range(len(known)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for i in range(len(known)):
        if expansions[i] is None:
            # An unrecognised form has no canonical identity to compare, so it stands alone rather
            # than being pooled with every other unrecognised entry.
            continue
        for j in range(i + 1, len(known)):
            if expansions[j] is None:
                continue
            if any(
                relation_signature_matches(left, right)
                for left in expansions[i]
                for right in expansions[j]
            ):
                union(i, j)

    groups: dict = {}
    for index in range(len(known)):
        groups.setdefault(find(index), []).append(known[index])
    if len(groups) < 2:
        raise ValueError(
            "calibration requires at least two distinct known-invariant relations for a "
            "disjoint held-out validation split; the catalogue collapses to "
            f"{len(groups)} once entries that the recovery matcher cannot tell apart are merged"
        )
    order = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(order)
    n_val = min(
        len(order) - 1,
        max(1, int(round(frac * len(order)))),
    )
    val_keys = set(order[:n_val])
    calib = [inv for key, members in groups.items() if key not in val_keys for inv in members]
    valid = [inv for key, members in groups.items() if key in val_keys for inv in members]
    return calib, valid


def _derive_regime(cfg: "CalibrationConfig", calib: List[KnownInvariant]) -> RegimeSpec:
    """The proxy suite calibration will tune on.

    A caller-supplied ``cfg.regime`` is authoritative and returned unchanged (its active entries are
    used as-is).  Otherwise the positive proxy suite is derived *only* from the relation shapes
    present in the calibration-split known invariants -- never the held-out validation split, never
    a hard-coded full suite.  If no supported shape can be derived, calibration fails loudly rather
    than silently proxying every shape (the null control is always added separately downstream).
    """
    if cfg.regime is not None:
        if not cfg.regime.active_entries():
            raise ValueError(
                "custom CalibrationConfig.regime has no active proxy entries; activate at least "
                "one shape or omit the regime to derive it from the calibration invariants")
        return cfg.regime
    regime = abstract_from_shapes(abstract_shapes(calib))
    if not regime.active_entries():
        raise ValueError(
            "no supported proxy shape can be derived from the calibration invariants "
            "(supported forms include equality, sums, ratios, proportionality, one-sided bounds, "
            "temporal deltas/windows, conditions, related aggregates, and Boolean/categorical "
            "definitions); "
            "supply a custom CalibrationConfig.regime to calibrate this dataset")
    return regime


def _make_calibration_inducer(cfg: "CalibrationConfig"):
    if cfg.backend == "subagent":
        return make_inducer("subagent", harness=cfg.harness)
    return make_inducer(cfg.backend)


def _knob_schedule(base: DiscoveryConfig, null_floor: float = 0.5) -> List[DiscoveryConfig]:
    """The Tuner's generic-knob relaxation ladder (tight -> loose).

    Every step touches only dataset-agnostic knobs (band mode, tolerance, hold-rate threshold) --
    never the user's specific invariants.  The base band mode (``base.band_mode`` -- ``global`` for
    calibration by default, though the ladder honours whatever mode ``base`` carries) is exercised
    first; later rungs lower the threshold and then widen to a looser fixed **global** band, which
    helps systematic-offset laws (e.g. I5/I6) whose whole population sits at one scale.
    """
    wide = max(2.0 * base.tolerance, 0.1)   # genuinely looser than the base band, so the fallback
                                            # rungs are real relaxations and not no-ops even when the
                                            # base is already a fixed global band
    low = max(null_floor, min(base.hold_rate_threshold, 0.60))
    ladder = [
        base,                                                            # base band (global by default), tuned knobs
        replace(base, hold_rate_threshold=low),                          # lower threshold
        replace(base, band_mode="global", tolerance=wide),               # fixed global-band fallback
        replace(base, band_mode="global", tolerance=wide, hold_rate_threshold=low),
    ]
    out: List[DiscoveryConfig] = []
    for c in ladder:
        key = (c.band_mode, round(c.tolerance, 4), round(c.hold_rate_threshold, 4))
        if not out or key != (out[-1].band_mode, round(out[-1].tolerance, 4),
                              round(out[-1].hold_rate_threshold, 4)):
            out.append(c)
    return out


def _capability_tiers() -> List[dict]:
    """Grammar-widening tiers, applied to a freshly re-induced spec when recall stalls.

    Tier 0 is the model's own proposal; each later tier raises a capability floor so the search
    space strictly grows (more aggregations, then decidable-nonlinear products/ratios).
    """
    return [
        {},                                              # tier 0: as the subagent proposed
        {"all_aggs": True},                              # tier 1: enable SUM/AVG/MIN/MAX
        {"all_aggs": True, "max_degree": 2, "proportional": True},
        {
            "all_aggs": True,
            "max_degree": 2,
            "proportional": True,
            "temporal": True,
            "max_lag": 60,
            "windows": (10, 45, 60),
        },
        {
            "all_aggs": True,
            "max_degree": 2,
            "proportional": True,
            "temporal": True,
            "max_lag": 60,
            "windows": (10, 45, 60),
            "advanced": True,
            "run_lengths": (10,),
            "max_conjunction_terms": 3,
        },
    ]


def _widen_spec(spec, *, all_aggs: bool = False, max_degree: Optional[int] = None,
                drop_exclusions: bool = False, proportional: bool = False,
                temporal: bool = False, max_lag: int = 0, windows=(),
                advanced: bool = False, run_lengths=(),
                max_conjunction_terms: int = 3):
    """Return a capability-widened copy of a GrammarSpec (frozen dataclasses -> ``replace``)."""
    onto = spec.ontology
    if all_aggs:
        agg = tuple(dict.fromkeys(tuple(onto.agg_kinds) + ("SUM", "AVG", "MIN", "MAX")))
        onto = replace(onto, agg_kinds=agg)
    if proportional:
        onto = replace(onto, ops=tuple(dict.fromkeys(tuple(onto.ops) + ("~∝",))))
    md = max(spec.max_degree, max_degree) if max_degree is not None else spec.max_degree
    excl = () if drop_exclusions else spec.role_exclusions
    return replace(
        spec,
        ontology=onto,
        aggregations_widened=bool(
            spec.aggregations_widened or all_aggs
        ),
        temporal_bounds_widened=bool(
            spec.temporal_bounds_widened or temporal
        ),
        advanced_bounds_widened=bool(
            spec.advanced_bounds_widened or advanced
        ),
        degree_widened=bool(
            spec.degree_widened or max_degree is not None
        ),
        proportional_widened=bool(
            spec.proportional_widened or proportional
        ),
        max_degree=md,
        role_exclusions=excl,
        temporal_enabled=bool(spec.temporal_enabled or temporal),
        max_lag=max(int(spec.max_lag), int(max_lag)),
        windows=tuple(sorted({*spec.windows, *(int(window) for window in windows)})),
        advanced_enabled=bool(spec.advanced_enabled or advanced),
        run_lengths=tuple(sorted({
            *spec.run_lengths,
            *(int(window) for window in run_lengths),
        })),
        max_conjunction_terms=max(
            int(spec.max_conjunction_terms),
            int(max_conjunction_terms),
        ),
    )


def _merge_specs(base, new):
    """Union two specs' search spaces so re-induction can only *grow* the grammar (item 2).

    A fresh induction each tier is non-deterministic: a role/pattern/family present in an earlier
    tier can be absent from a later proposal, so a bare re-induction does **not** guarantee the
    "search space strictly grows" property the tiers rely on.  This folds the new proposal into the
    accumulated ``base`` spec, keeping ``base`` authoritative on every conflict so no existing
    grounding is silently redefined, and only *adding* what ``new`` proposes:

    * ``patterns`` / ``ref_templates`` / ``family_selectors`` -- base kept; a new entry is appended
      only when its identity key (pattern name, ``(binder, role)``, ``(binder, family_role)``) is
      unseen, so a shared role keeps ``base``'s grounding.
    * ``ontology`` -- binders, per-binder ref/family roles, ops and agg kinds are unioned; base
      glyphs win.
    * ``binder_enumerate`` -- base strategy wins per binder; new binders are added.  ``max_degree``
      is the max of the two.
    * ``role_exclusions`` and all dataset-level constants (codec, kinds, link marker, name) are
      taken from ``base`` unchanged.

    The result therefore admits **every** rule ``base`` did (a genuine superset) plus the novel
    vocabulary ``new`` contributes -- regardless of what the fresh proposal omitted.
    """
    seen_pat = {p.name for p in base.patterns}
    patterns = base.patterns + tuple(p for p in new.patterns if p.name not in seen_pat)

    seen_ref = {(t.binder, t.role) for t in base.ref_templates}
    ref_templates = base.ref_templates + tuple(
        t for t in new.ref_templates if (t.binder, t.role) not in seen_ref)

    seen_fam = {(s.binder, s.family_role) for s in base.family_selectors}
    family_selectors = base.family_selectors + tuple(
        s for s in new.family_selectors if (s.binder, s.family_role) not in seen_fam)

    ob, on = base.ontology, new.ontology

    def _union_roles(a, b):
        out = {k: tuple(v) for k, v in a.items()}
        for k, v in b.items():
            out[k] = tuple(dict.fromkeys(tuple(out.get(k, ())) + tuple(v)))
        return out

    ontology = replace(
        ob,
        binders=tuple(dict.fromkeys(tuple(ob.binders) + tuple(on.binders))),
        ref_roles=_union_roles(ob.ref_roles, on.ref_roles),
        fam_roles=_union_roles(ob.fam_roles, on.fam_roles),
        ops=tuple(dict.fromkeys(tuple(ob.ops) + tuple(on.ops))),
        agg_kinds=tuple(dict.fromkeys(tuple(ob.agg_kinds) + tuple(on.agg_kinds))),
        ref_glyphs={**on.ref_glyphs, **ob.ref_glyphs},
        fam_glyphs={**on.fam_glyphs, **ob.fam_glyphs},
    )

    conditions = {
        key: tuple(values)
        for key, values in base.condition_columns.items()
    }
    for key, values in new.condition_columns.items():
        conditions[key] = tuple(dict.fromkeys(
            (*conditions.get(key, ()), *tuple(values))
        ))
    seen_related = {
        (template.binder, template.role)
        for template in base.related_templates
    }
    related_templates = base.related_templates + tuple(
        template
        for template in new.related_templates
        if (template.binder, template.role) not in seen_related
    )
    boolean_roles = _union_roles(
        base.boolean_roles,
        new.boolean_roles,
    )

    return replace(
        base,
        patterns=patterns,
        ontology=ontology,
        ref_templates=ref_templates,
        family_selectors=family_selectors,
        binder_enumerate={
            **new.binder_enumerate,
            **base.binder_enumerate,
        },
        max_degree=max(base.max_degree, new.max_degree),
        time_index=base.time_index or new.time_index,
        group_keys=tuple(dict.fromkeys(
            (*base.group_keys, *new.group_keys)
        )),
        condition_columns=conditions,
        temporal_enabled=bool(
            base.temporal_enabled or new.temporal_enabled
        ),
        max_lag=max(base.max_lag, new.max_lag),
        windows=tuple(sorted({*base.windows, *new.windows})),
        conditional_enabled=bool(
            base.conditional_enabled or new.conditional_enabled
        ),
        max_condition_values=max(
            base.max_condition_values,
            new.max_condition_values,
        ),
        related_templates=related_templates,
        boolean_roles=boolean_roles,
        advanced_enabled=bool(
            base.advanced_enabled or new.advanced_enabled
        ),
        run_lengths=tuple(sorted({
            *base.run_lengths,
            *new.run_lengths,
        })),
        max_conjunction_terms=max(
            base.max_conjunction_terms,
            new.max_conjunction_terms,
        ),
        metadata_columns=tuple(dict.fromkeys(
            (*base.metadata_columns, *new.metadata_columns)
        )),
        band_enabled=bool(base.band_enabled or new.band_enabled),
        aggregations_widened=bool(
            base.aggregations_widened
            or new.aggregations_widened
        ),
        temporal_bounds_widened=bool(
            base.temporal_bounds_widened
            or new.temporal_bounds_widened
        ),
        advanced_bounds_widened=bool(
            base.advanced_bounds_widened
            or new.advanced_bounds_widened
        ),
        degree_widened=bool(
            base.degree_widened or new.degree_widened
        ),
        proportional_widened=bool(
            base.proportional_widened
            or new.proportional_widened
        ),
    )


def _spec_summary(spec, tier: int, caps: dict) -> dict:
    onto = spec.ontology
    return {
        "tier": tier,
        "capabilities_forced": (caps or "as-induced"),
        "name": spec.name,
        "binders": list(onto.binders),
        "agg_kinds": list(onto.agg_kinds),
        "max_degree": spec.max_degree,
        "temporal_enabled": spec.temporal_enabled,
        "max_lag": spec.max_lag,
        "windows": list(spec.windows),
        "advanced_enabled": spec.advanced_enabled,
        "run_lengths": list(spec.run_lengths),
        "max_conjunction_terms": spec.max_conjunction_terms,
        "role_exclusions": len(spec.role_exclusions),
        "n_ref_roles": sum(len(v) for v in onto.ref_roles.values()),
        "n_fam_roles": sum(len(v) for v in onto.fam_roles.values()),
    }


def calibrate(df, known_path: str, cfg: Optional[CalibrationConfig] = None,
              name: str = "calibrate") -> dict:
    """Full calibration loop: jointly tune knobs on the proxy suite, discover on the dataset, report recall.

    Objective: recall on a *held-out* validation split, subject to a false-discovery ceiling
    (null acceptance) -- never recall alone.  The proxy suite is a caller-supplied ``RegimeSpec`` or
    one derived from the calibration-split shapes; every selected positive proxy and the always-on
    null control are prepared once and reused to jointly pick one shared (tolerance, hold-rate
    threshold).  The relaxation ladder then runs on the real data (one fixed global band by default),
    re-inducing the grammar with widened capabilities on stall, and every rung must clear the same
    zero-null-equality gate before it can win.  The learned portfolio is persisted to
    ``<rules_dir>/<name>_<timestamp>.dl`` and echoed into the report as ``learned_invariants``.
    """
    cfg = cfg or CalibrationConfig()
    profile = dict(
        getattr(df, "attrs", {}).get("autogram_profile", {})
    )
    tiers = _capability_tiers()[:max(1, cfg.max_capability_tiers)]
    advanced_possible = bool(profile.get("advanced", False)) or any(
        bool(caps.get("advanced", False))
        for caps in tiers
    )
    if advanced_possible and not 1 <= int(cfg.max_rules) <= 500_000:
        raise ValueError(
            "advanced calibration requires finite max_rules "
            "between 1 and 500000"
        )
    known = load_known(known_path)
    calib, valid = _split_known(
        known,
        cfg.validation_frac,
        cfg.seed,
        frame=_ColumnScaleView(df),
    )
    inducer = _make_calibration_inducer(cfg)

    # 1) proxy suite -- a caller-supplied regime is authoritative; otherwise it is derived from the
    #    calibration-split shapes only.  The null control is always included by prepare_proxy_suite.
    regime = _derive_regime(cfg, calib)
    null_max_conjunction_terms = int(
        profile.get("max_conjunction_terms", 3)
    )
    for caps in tiers:
        null_max_conjunction_terms = max(
            null_max_conjunction_terms,
            int(caps.get("max_conjunction_terms", 3)),
        )
    scfg = SearchConfig(
        seed=cfg.seed,
        max_complexity=cfg.max_complexity,
        max_add_arity=cfg.max_add_arity,
        max_rules=cfg.max_rules,
        max_nonlinear_leaves=cfg.max_nonlinear_leaves,
        max_linear_leaves=cfg.max_linear_leaves,
        max_conditioned_rules=cfg.max_conditioned_rules,
        max_lag=cfg.max_lag,
        windows=cfg.windows,
    )

    # 2) prepare every selected positive proxy + the null control ONCE (schema induction per proxy);
    #    the prepared grammars are reused for every joint-tuning grid cell and every ladder rung.
    suite = prepare_proxy_suite(
        regime,
        seed=cfg.seed,
        inducer=inducer,
        null_windows=(),
        null_max_conjunction_terms=null_max_conjunction_terms,
        null_search_cfg=SearchConfig(seed=cfg.seed),
    )

    # 3) jointly tune ONE shared (tolerance, hold-rate threshold) across the whole selected suite +
    #    null, under the calibration band mode.  A setting is eligible only when every positive proxy
    #    hits its recovery target with a compact, scaled-slack-free portfolio and the null accepts no
    #    equality; the grid expands on stall and fails loudly (with per-proxy evidence) otherwise.
    joint = tune_joint(
        suite,
        seed=cfg.seed,
        band_mode=cfg.band_mode,
        ci_alpha=cfg.ci_alpha,
        null_floor=cfg.null_floor,
        initial_tolerance=cfg.tolerance,
        initial_threshold=cfg.hold_rate_threshold,
    )
    del suite
    gc.collect()
    base = DiscoveryConfig(seed=cfg.seed,
                           tolerance=float(joint["tolerance"]),
                           hold_rate_threshold=float(joint["hold_rate_threshold"]),
                           band_mode=cfg.band_mode,
                           ci_alpha=cfg.ci_alpha)
    schedule = _knob_schedule(base, null_floor=cfg.null_floor)
    iteration_budget = (
        int(cfg.max_iterations)
        if cfg.max_iterations and cfg.max_iterations > 0
        else None
    )

    # 4) outer grammar-capability loop + inner knob ladder
    history: List[dict] = []
    grammar_specs: List[dict] = []
    best = None            # (recall, dcfg, res, tier, caps, null_eq, null_temporal)
    reinductions = 0
    global_iter = 0
    accumulated = None     # running union of induced specs -> re-induction can only grow it (item 2)
    # Memoize the null gate by (band_mode, tolerance, threshold): the same relaxation-ladder rungs
    # recur in every grammar tier, and the null grammar is prepared once, so each unique config only
    # needs scoring once across all tiers.
    null_cache: dict = {}

    for ti, caps in enumerate(tiers):
        if iteration_budget is not None and global_iter >= iteration_budget:
            break
        spec = induce_spec(list(df.columns), inducer)     # (re-)propose the grammar (columns only)
        if ti > 0:
            reinductions += 1
            # Fold the fresh (non-deterministic) proposal back into the accumulated grammar so a
            # later tier can never drop a role/pattern an earlier tier already had -- this is what
            # makes "the search space strictly grows" across tiers actually hold (item 2).
            spec = _merge_specs(accumulated, spec)
        spec = _widen_spec(spec, all_aggs=caps.get("all_aggs", False),
                           max_degree=caps.get("max_degree"),
                           drop_exclusions=caps.get("drop_exclusions", False),
                           proportional=caps.get("proportional", False),
                           temporal=caps.get("temporal", False),
                           max_lag=caps.get("max_lag", 0),
                           windows=caps.get("windows", ()),
                           advanced=caps.get("advanced", False),
                           run_lengths=caps.get("run_lengths", ()),
                           max_conjunction_terms=caps.get("max_conjunction_terms", 3))
        accumulated = spec
        runtime_spec = normalize_dataframe_spec(df, spec, scfg)
        ds, G = build_dataframe_grammar(
            df,
            runtime_spec,
            search_cfg=scfg,
            name=name,
        )
        summary = _spec_summary(runtime_spec, ti, caps)
        summary["runtime"] = {
            "agg_kinds": list(getattr(G, "agg_kinds", ())),
            "max_degree": getattr(G, "max_degree", None),
            "temporal_enabled": getattr(G, "temporal_enabled", False),
            "max_lag": getattr(G, "max_lag", 0),
            "windows": list(getattr(G, "windows", ())),
            "advanced_enabled": getattr(G, "advanced_enabled", False),
            "run_lengths": list(getattr(G, "run_lengths", ())),
            "max_conjunction_terms": getattr(
                G,
                "max_conjunction_terms",
                None,
            ),
            "max_complexity": getattr(G, "max_complexity", None),
            "max_add_arity": getattr(G, "max_add_arity", None),
        }
        proposer = EnumerationProposer(G)
        runtime_nulls = prepare_runtime_null_controls(
            ds,
            G,
            scfg,
            seed=cfg.seed + ti * 100_003,
            proposer=proposer,
        )
        summary["runtime"]["candidate_count"] = (
            runtime_nulls.candidate_counts["all"]
        )
        summary["runtime"]["null_candidate_counts"] = dict(
            runtime_nulls.candidate_counts
        )
        normalized_spec = _json_normalize(_spec_to_json(runtime_spec))
        summary["normalized_spec"] = normalized_spec
        summary["normalized_spec_sha256"] = _json_fingerprint(
            normalized_spec
        )
        grammar_specs.append(summary)

        def _null_gate(
            dcfg: DiscoveryConfig,
        ) -> tuple[int, int, int]:
            key = (
                ti,
                dcfg.band_mode,
                dcfg.tolerance,
                dcfg.hold_rate_threshold,
            )
            if key not in null_cache:
                null_cache[key] = (
                    null_equalities_at(
                        runtime_nulls.null,
                        dcfg,
                        cfg.seed,
                    ),
                    null_temporal_at(
                        runtime_nulls.temporal_null,
                        dcfg,
                        cfg.seed,
                    ),
                    null_definitions_at(
                        runtime_nulls.definition_null,
                        dcfg,
                        cfg.seed,
                    ),
                )
            return null_cache[key]

        for dcfg in schedule:
            if iteration_budget is not None and global_iter >= iteration_budget:
                break
            global_iter += 1
            res = run_prepared(
                ds,
                G,
                discovery_cfg=dcfg,
                search_cfg=scfg,
                proposer=proposer,
            )
            rec = recover_known(res, calib)["recall"]
            # Same zero-null-equality gate as tuning, at THIS rung's (tolerance, threshold), reusing
            # the once-prepared null grammar (memoized across tiers): a proxy-safe base is not enough
            # if a relaxed rung is unsafe, so an unsafe rung can never become the winner.
            null_eq, null_temporal, null_definitions = _null_gate(dcfg)
            null_safe = (
                null_eq == 0
                and null_temporal == 0
                and null_definitions == 0
            )
            history.append({
                "iteration": global_iter,
                "grammar_tier": ti,
                "tolerance": round(dcfg.tolerance, 4),
                "hold_rate_threshold": round(dcfg.hold_rate_threshold, 4),
                "band_mode": dcfg.band_mode,
                "calibration_recall": round(rec, 4),
                "rules_learned": len(res.portfolio),
                "null_equalities": null_eq,
                "null_temporal": null_temporal,
                "null_definitions": null_definitions,
                "null_safe": null_safe,
            })
            if null_safe and (best is None or rec > best[0]):
                best = (
                    rec,
                    dcfg,
                    res,
                    ti,
                    caps,
                    null_eq,
                    null_temporal,
                    null_definitions,
                )
            if null_safe and rec >= 1.0:
                break
        if best is not None and best[0] >= 1.0:
            break                                          # solved -- no need to widen further

    if best is None:
        raise CalibrationGridError(
            "no null-safe ladder rung was found on the real data at any tuned or relaxed setting; "
            "raise the null-floor headroom or supply a custom regime")

    (
        recall,
        best_dcfg,
        best_res,
        best_tier,
        best_caps,
        best_null,
        best_temporal_null,
        best_definition_null,
    ) = best
    report_all = recover_known(best_res, known)
    report_val = recover_known(best_res, valid) if valid else report_all

    # Persist the learned invariants by default: a human-readable .dl file + the rules in the report.
    rules_file = None
    if cfg.save_rules:
        rules_file = write_rules_dl(best_res, name, out_dir=cfg.rules_dir,
                                    seed=cfg.seed, proposer="enumeration", git=_git_short())
    from .discovery.export import _adapter_of
    from .dsl.parser import rule_to_dict
    from .dsl.render import render_rule
    adapter = _adapter_of(best_res)
    learned_invariants = [
        {
            "rule": ev.rule.unparse(),
            "rule_payload": rule_to_dict(ev.rule),
            "rule_explicit": render_rule(ev.rule, adapter),
            "hold_rate": round(ev.hold_rate, 4),
            "hold_rate_ci": [round(ev.hold_rate_lo, 4), round(ev.hold_rate_hi, 4)],
            "eps": ev.eps,
            "strictness": ev.strictness,
            "support": round(ev.support, 3),
            "parameters": dict(getattr(ev, "parameters", {})),
        }
        for ev in best_res.portfolio
    ]
    return {
        "config": {
            "band_mode": best_dcfg.band_mode,
            "tolerance": round(best_dcfg.tolerance, 4),
            "hold_rate_threshold": round(best_dcfg.hold_rate_threshold, 4),
            "ci_alpha": best_dcfg.ci_alpha,
            "grammar_tier": best_tier,
            "capabilities": (best_caps or "as-induced"),
            "grid_expansions": joint["expansions"],
        },
        "iterations": len(history),
        "grammar_reinductions": reinductions,
        "trajectory": history,
        "grammar_specs": grammar_specs,
        "regime_proxies": [{"shape": e.shape, "noise": e.noise, "active": e.active}
                           for e in regime.entries],
        "proxies": {
            "shapes": joint["proxy_shapes"],
            "per_proxy": joint["per_proxy"],
            "selected_null_equalities": joint["selected_null_equalities"],
            "selected_null_temporal": joint.get("selected_null_temporal", 0),
            "selected_null_definitions": joint.get("selected_null_definitions", 0),
            "grid_expansions": joint["expansions"],
        },
        "induction": {
            "backend": cfg.backend,
            "harness": cfg.harness,
        },
        "recall_all": report_all["recall"],
        "recall_validation": report_val["recall"],
        "recovered_all": f"{report_all['recovered']}/{report_all['total']}",
        "false_discovery": {
            "null_equalities_accepted": best_null,
            "null_temporal_accepted": best_temporal_null,
            "null_definitions_accepted": best_definition_null,
        },
        "n_rules_learned": len(best_res.portfolio),
        "rules_file": rules_file,
        "provenance": _calibration_provenance(df, known, cfg),
        "learned_invariants": learned_invariants,
        "invariants": report_all["invariants"],
        "limits": ("Known-invariant recall is a lower bound under representativeness, not a "
                   "guarantee of discovering unknown invariants (see docs/calibration_protocol.md)."),
    }
