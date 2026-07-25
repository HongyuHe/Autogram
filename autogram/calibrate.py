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

import os
import random
import shutil
from dataclasses import dataclass, replace
from typing import List, Optional

from .config import DiscoveryConfig, SearchConfig
from .discovery.export import write_rules_dl
from .discovery.induce import induce_spec, make_inducer
from .discovery.known import KnownInvariant, abstract_shapes, load_known, recover_known
from .discovery.loop import build_dataframe_grammar, run_prepared
from .discovery.regime import RegimeSpec, abstract_from_shapes
from .discovery.subagent import HARNESSES
from .discovery.validate import (
    CalibrationGridError, null_equalities_at, prepare_proxy_suite, tune_joint,
)


@dataclass
class CalibrationConfig:
    seed: int = 0
    max_iterations: int = 0              # 0 = full knob ladder per grammar tier (default)
    validation_frac: float = 0.3         # held-out split for honest recall
    harness: str = "copilot"
    backend: str = "subagent"
    null_floor: float = 0.5              # threshold never drops below the false-discovery floor
    band_mode: str = "global"            # DEFAULT for calibration: one fixed global tolerance; "adaptive" = per-candidate self-calibrated band. (The engine's own DiscoveryConfig default stays "adaptive".)
    max_capability_tiers: int = 3        # grammar re-induction tiers when recall stalls
    regime: Optional[RegimeSpec] = None  # wired, editable synthetic-proxy suite (item 7)
    save_rules: bool = True              # persist the learned portfolio to <rules_dir>/<name>_<ts>.dl
    rules_dir: str = "rules"


def _git_short() -> str:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def precheck(harness: str = "copilot", backend: str = "subagent") -> dict:
    """Validate the runtime can induce schemas before a calibration run (`autogram precheck`)."""
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
    return {"ok": not issues, "issues": issues, "harness": harness, "backend": backend}


def _split_known(known: List[KnownInvariant], frac: float, seed: int):
    if len(known) <= 1:
        return known, list(known)
    rng = random.Random(seed)
    idx = list(range(len(known)))
    rng.shuffle(idx)
    n_val = max(1, int(round(frac * len(known))))
    val_idx = set(idx[:n_val])
    calib = [known[i] for i in range(len(known)) if i not in val_idx]
    valid = [known[i] for i in range(len(known)) if i in val_idx]
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
            "(only ==, ~=, reference==sum, ==0, <|>, >=0 and <=0 forms are abstractable); "
            "supply a custom CalibrationConfig.regime to calibrate this dataset")
    return regime


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
        {"all_aggs": True, "max_degree": 2},             # tier 2: enable products/ratios (item 6)
    ]


def _widen_spec(spec, *, all_aggs: bool = False, max_degree: Optional[int] = None,
                drop_exclusions: bool = False):
    """Return a capability-widened copy of a GrammarSpec (frozen dataclasses -> ``replace``)."""
    onto = spec.ontology
    if all_aggs:
        agg = tuple(dict.fromkeys(tuple(onto.agg_kinds) + ("SUM", "AVG", "MIN", "MAX")))
        onto = replace(onto, agg_kinds=agg)
    md = max(spec.max_degree, max_degree) if max_degree is not None else spec.max_degree
    excl = () if drop_exclusions else spec.role_exclusions
    return replace(spec, ontology=onto, max_degree=md, role_exclusions=excl)


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

    return replace(base, patterns=patterns, ontology=ontology,
                   ref_templates=ref_templates, family_selectors=family_selectors,
                   binder_enumerate={**new.binder_enumerate, **base.binder_enumerate},
                   max_degree=max(base.max_degree, new.max_degree))


def _spec_summary(spec, tier: int, caps: dict) -> dict:
    onto = spec.ontology
    return {
        "tier": tier,
        "capabilities_forced": (caps or "as-induced"),
        "name": spec.name,
        "binders": list(onto.binders),
        "agg_kinds": list(onto.agg_kinds),
        "max_degree": spec.max_degree,
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
    known = load_known(known_path)
    calib, valid = _split_known(known, cfg.validation_frac, cfg.seed)
    inducer = make_inducer("subagent", harness=cfg.harness)

    # 1) proxy suite -- a caller-supplied regime is authoritative; otherwise it is derived from the
    #    calibration-split shapes only.  The null control is always included by prepare_proxy_suite.
    regime = _derive_regime(cfg, calib)

    # 2) prepare every selected positive proxy + the null control ONCE (schema induction per proxy);
    #    the prepared grammars are reused for every joint-tuning grid cell and every ladder rung.
    suite = prepare_proxy_suite(regime, seed=cfg.seed, inducer=inducer)

    # 3) jointly tune ONE shared (tolerance, hold-rate threshold) across the whole selected suite +
    #    null, under the calibration band mode.  A setting is eligible only when every positive proxy
    #    hits its recovery target with a compact, scaled-slack-free portfolio and the null accepts no
    #    equality; the grid expands on stall and fails loudly (with per-proxy evidence) otherwise.
    joint = tune_joint(suite, seed=cfg.seed, band_mode=cfg.band_mode, null_floor=cfg.null_floor)
    base = DiscoveryConfig(seed=cfg.seed,
                           tolerance=float(joint["tolerance"]),
                           hold_rate_threshold=float(joint["hold_rate_threshold"]),
                           band_mode=cfg.band_mode)
    scfg = SearchConfig(seed=cfg.seed)

    schedule = _knob_schedule(base, null_floor=cfg.null_floor)
    knob_budget = cfg.max_iterations if cfg.max_iterations and cfg.max_iterations > 0 else len(schedule)
    tiers = _capability_tiers()[:max(1, cfg.max_capability_tiers)]

    # 4) outer grammar-capability loop + inner knob ladder
    history: List[dict] = []
    grammar_specs: List[dict] = []
    best = None            # (recall, dcfg, res, tier, caps, null_eq)
    reinductions = 0
    prev_tier_best = -1.0
    global_iter = 0
    accumulated = None     # running union of induced specs -> re-induction can only grow it (item 2)
    # Memoize the null gate by (band_mode, tolerance, threshold): the same relaxation-ladder rungs
    # recur in every grammar tier, and the null grammar is prepared once, so each unique config only
    # needs scoring once across all tiers.
    null_cache: dict = {}

    def _null_gate(dcfg: DiscoveryConfig) -> int:
        key = (dcfg.band_mode, dcfg.tolerance, dcfg.hold_rate_threshold)
        if key not in null_cache:
            null_cache[key] = null_equalities_at(suite.null, dcfg, cfg.seed)
        return null_cache[key]

    for ti, caps in enumerate(tiers):
        spec = induce_spec(list(df.columns), inducer)     # (re-)propose the grammar (columns only)
        if ti > 0:
            reinductions += 1
            # Fold the fresh (non-deterministic) proposal back into the accumulated grammar so a
            # later tier can never drop a role/pattern an earlier tier already had -- this is what
            # makes "the search space strictly grows" across tiers actually hold (item 2).
            spec = _merge_specs(accumulated, spec)
        spec = _widen_spec(spec, all_aggs=caps.get("all_aggs", False),
                           max_degree=caps.get("max_degree"),
                           drop_exclusions=caps.get("drop_exclusions", False))
        accumulated = spec
        grammar_specs.append(_spec_summary(spec, ti, caps))
        ds, G = build_dataframe_grammar(df, spec, search_cfg=scfg, name=name)

        tier_best = -1.0
        for dcfg in schedule[:knob_budget]:
            global_iter += 1
            res = run_prepared(ds, G, discovery_cfg=dcfg, search_cfg=scfg)
            rec = recover_known(res, calib)["recall"]
            # Same zero-null-equality gate as tuning, at THIS rung's (tolerance, threshold), reusing
            # the once-prepared null grammar (memoized across tiers): a proxy-safe base is not enough
            # if a relaxed rung is unsafe, so an unsafe rung can never become the winner.
            null_eq = _null_gate(dcfg)
            null_safe = null_eq == 0
            history.append({
                "iteration": global_iter,
                "grammar_tier": ti,
                "tolerance": round(dcfg.tolerance, 4),
                "hold_rate_threshold": round(dcfg.hold_rate_threshold, 4),
                "band_mode": dcfg.band_mode,
                "calibration_recall": round(rec, 4),
                "rules_learned": len(res.portfolio),
                "null_equalities": null_eq,
                "null_safe": null_safe,
            })
            tier_best = max(tier_best, rec)
            if null_safe and (best is None or rec > best[0]):
                best = (rec, dcfg, res, ti, caps, null_eq)
            if null_safe and rec >= 1.0:
                break
        if best is not None and best[0] >= 1.0:
            break                                          # solved -- no need to widen further
        if tier_best <= prev_tier_best + 1e-9:
            break                                          # stall: widening the grammar didn't help
        prev_tier_best = tier_best

    if best is None:
        raise CalibrationGridError(
            "no null-safe ladder rung was found on the real data at any tuned or relaxed setting; "
            "raise the null-floor headroom or supply a custom regime")

    recall, best_dcfg, best_res, best_tier, best_caps, best_null = best
    report_all = recover_known(best_res, known)
    report_val = recover_known(best_res, valid) if valid else report_all

    # Persist the learned invariants by default: a human-readable .dl file + the rules in the report.
    rules_file = None
    if cfg.save_rules:
        rules_file = write_rules_dl(best_res, name, out_dir=cfg.rules_dir,
                                    seed=cfg.seed, proposer="enumeration", git=_git_short())
    from .discovery.export import _adapter_of
    from .dsl.render import render_rule
    adapter = _adapter_of(best_res)
    learned_invariants = [
        {
            "rule": ev.rule.unparse(),
            "rule_explicit": render_rule(ev.rule, adapter),
            "hold_rate": round(ev.hold_rate, 4),
            "hold_rate_ci": [round(ev.hold_rate_lo, 4), round(ev.hold_rate_hi, 4)],
            "eps": ev.eps,
            "strictness": ev.strictness,
            "support": round(ev.support, 3),
        }
        for ev in best_res.portfolio
    ]
    return {
        "config": {
            "band_mode": best_dcfg.band_mode,
            "tolerance": round(best_dcfg.tolerance, 4),
            "hold_rate_threshold": round(best_dcfg.hold_rate_threshold, 4),
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
            "grid_expansions": joint["expansions"],
        },
        "recall_all": report_all["recall"],
        "recall_validation": report_val["recall"],
        "recovered_all": f"{report_all['recovered']}/{report_all['total']}",
        "false_discovery": {"null_equalities_accepted": best_null},
        "n_rules_learned": len(best_res.portfolio),
        "rules_file": rules_file,
        "learned_invariants": learned_invariants,
        "invariants": report_all["invariants"],
        "limits": ("Known-invariant recall is a lower bound under representativeness, not a "
                   "guarantee of discovering unknown invariants (see docs/calibration_protocol.md)."),
    }
