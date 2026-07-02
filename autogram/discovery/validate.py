"""Synthetic proxy and two-phase evaluation helpers for v2.

The proxy phase tunes generic knobs on synthetic or observed-only data.  The frozen phase runs the
same pipeline on CrossCheck DataFrames without reading clean frames or a target-invariant catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Dict, List, Optional, Sequence

from ..config import DiscoveryConfig, SearchConfig
from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_family, resolve_ref
from . import synth as S
from .induce import SchemaInducer, make_inducer
from .loop import DiscoveryResult, discover, discover_dataframe


def _operand_sig(rule, binder, binding, nm):
    left, right, op = rule.atom.left, rule.atom.right, rule.atom.op

    def ref_col(t):
        return resolve_ref(t.role, binder, binding, nm) if isinstance(t, A.Ref) else None

    def fam_cols(t):
        return resolve_family(t.family_role, binder, binding, nm) if isinstance(t, A.Agg) else None

    lc, rc = ref_col(left), ref_col(right)
    if lc is not None and rc is not None and op in ("~=", "=="):
        return ("pair", frozenset({lc, rc}))
    if lc is not None and rc is not None and op == "<|>":
        return ("presence_pair", frozenset({lc, rc}))
    for a, b in ((left, right), (right, left)):
        if op in ("~=", "==") and isinstance(a, A.Ref) and isinstance(b, A.Agg) and b.kind == "SUM":
            ac = ref_col(a)
            bc = fam_cols(b)
            if ac is not None and bc:
                return ("ref_sum", (ac, frozenset(bc)))
    if op in ("~=", "==") and isinstance(left, A.Add) and isinstance(right, A.Add):
        lsig = _add_ref_agg_sig(left, binder, binding, nm)
        rsig = _add_ref_agg_sig(right, binder, binding, nm)
        if lsig is not None and rsig is not None:
            return ("agg_ref_balance", frozenset({lsig, rsig}))
    for a, b in ((left, right), (right, left)):
        if op in ("~=", "==") and isinstance(a, A.Ref) and isinstance(b, A.Const) and b.value == 0:
            ac = ref_col(a)
            if ac is not None:
                return ("zero", ac)
    return None


def _add_ref_agg_sig(term, binder, binding, nm):
    refs = [t for t in term.terms if isinstance(t, A.Ref)]
    aggs = [t for t in term.terms if isinstance(t, A.Agg) and t.kind == "SUM"]
    if len(refs) != 1 or len(aggs) != 1 or len(term.terms) != 2:
        return None
    rc = resolve_ref(refs[0].role, binder, binding, nm)
    fc = resolve_family(aggs[0].family_role, binder, binding, nm)
    if rc is None or not fc:
        return None
    return (rc, frozenset(fc))


def rule_relations(rule: A.Rule, ds) -> set:
    nm = ds.name_model
    rels = set()
    for b in enumerate_bindings(rule.binder, nm):
        sig = _operand_sig(rule, rule.binder, b, nm)
        if sig is not None:
            rels.add(sig)
    return rels


def portfolio_relations(result: DiscoveryResult) -> set:
    rels: set = set()
    for ev in result.portfolio:
        rels |= rule_relations(ev.rule, result.dataset)
    return rels


@dataclass
class Recovery:
    two_end: float
    row_sum: float
    col_sum: float
    self_zero: float
    offset_pair: float
    agg_ref_balance: float
    presence_pair: float
    nonneg: bool
    recovered: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _pairs(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "pair"}


def _refsums(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "ref_sum"}


def _zeros(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "zero"}


def _agg_ref_balances(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "agg_ref_balance"}


def _presence_pairs(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "presence_pair"}


def score_recovery(result: DiscoveryResult, planted: dict, frac: float = 0.8) -> Recovery:
    rels = portfolio_relations(result)
    pairs, refsums, zeros = _pairs(rels), _refsums(rels), _zeros(rels)
    balances, presence = _agg_ref_balances(rels), _presence_pairs(rels)

    def cov(found, target):
        target = set(target)
        return 0.0 if not target else len(found & target) / len(target)

    two = cov(pairs, planted.get("two_end", set()))
    off = cov(pairs, planted.get("offset_pair", set()))
    row = cov(refsums, set(planted.get("row_sum", [])))
    col = cov(refsums, set(planted.get("col_sum", [])))
    sz = cov(zeros, set(planted.get("self_zero", [])))
    bal = cov(balances, set(planted.get("agg_ref_balance", [])))
    pres = cov(presence, set(planted.get("presence_pair", set())))
    nonneg = any(e.rule.atom.op == ">=" and isinstance(e.rule.atom.right, A.Const)
                 and e.rule.atom.right.value == 0 for e in result.portfolio)
    recovered = any(x >= frac for x in (two, row, col, sz, off, bal, pres)) or nonneg
    return Recovery(two, row, col, sz, off, bal, pres, nonneg, recovered)


@dataclass
class PlantRecover:
    noise_levels: List[float]
    recovered: Dict[str, Dict[float, bool]] = field(default_factory=dict)
    detail: Dict[str, Dict[float, dict]] = field(default_factory=dict)


_PLANT_FAMILIES = (
    "row_sum", "col_sum", "two_end", "self_zero",
    "offset_pair", "agg_ref_balance", "presence_pair",
)

_FAMILY_TOLERANCE = {
    "offset_pair": 0.01,
    "agg_ref_balance": 0.02,
    "presence_pair": 0.05,
}

_FAMILY_THRESHOLD = {
    "offset_pair": 0.62,
}


def _small_search(seed: int = 0, family: str = "") -> SearchConfig:
    return SearchConfig(seed=seed)


def _fast_eval(seed: int = 0, family: str = "", tolerance: float | None = None,
               hold_rate_threshold: float | None = None) -> DiscoveryConfig:
    tol = _FAMILY_TOLERANCE.get(family, 0.08)
    if tolerance is not None:
        tol = tolerance
    thr = _FAMILY_THRESHOLD.get(family, 0.9)
    if hold_rate_threshold is not None:
        thr = hold_rate_threshold
    return DiscoveryConfig(seed=seed, tolerance=tol, hold_rate_threshold=thr)


def plant_and_recover(noise_levels: Sequence[float] = (0.0, 0.02),
                      n_entities: int = 4, n_snapshots: int = 180,
                      seed: int = 0, inducer: Optional[SchemaInducer] = None) -> PlantRecover:
    pr = PlantRecover(noise_levels=list(noise_levels))
    for family in _PLANT_FAMILIES:
        pr.recovered[family] = {}
        pr.detail[family] = {}
        for nz in noise_levels:
            data = S.make_synthetic(n_entities=n_entities, n_snapshots=n_snapshots,
                                    noise=nz, seed=seed, families=(family,))
            res = discover(data.columns, data.matrix, inducer=inducer or make_inducer("subagent"),
                           discovery_cfg=_fast_eval(seed, family),
                           search_cfg=_small_search(seed, family),
                           name=f"proxy_{family}", timestamps=data.timestamps)
            rec = score_recovery(res, data.planted)
            found = bool(getattr(rec, family) >= 0.8)
            pr.recovered[family][nz] = found
            pr.detail[family][nz] = {**rec.as_dict(), "n_accepted": len(res.portfolio),
                                     "scaled_slack_rules": scaled_slack_rules(res),
                                     "rules": [e.rule.unparse() for e in res.portfolio]}
    return pr


def null_accepted(n_entities: int = 4, n_snapshots: int = 160, seed: int = 0,
                  inducer: Optional[SchemaInducer] = None) -> int:
    data = S.make_null(n_entities=n_entities, n_snapshots=n_snapshots, seed=seed)
    res = discover(data.columns, data.matrix, inducer=inducer or make_inducer("subagent"),
                   discovery_cfg=_fast_eval(seed), search_cfg=_small_search(seed),
                   name="null", timestamps=data.timestamps)
    return len([e for e in res.portfolio if e.rule.atom.op in ("~=", "==")])


def tune_threshold_tolerance(seed: int = 0) -> dict:
    """Jointly pick a tolerance/threshold pair on the approximate-offset proxy."""
    best = None
    for threshold, tolerance in itertools.product((0.58, 0.62, 0.66, 0.72), (0.005, 0.01, 0.02, 0.05)):
        data = S.make_synthetic(n_entities=3, n_snapshots=120, noise=0.0, seed=seed,
                                families=("offset_pair",), offset_hold_rate=0.67,
                                offset_factor=0.98)
        res = discover(data.columns, data.matrix, inducer=make_inducer("subagent"),
                       discovery_cfg=_fast_eval(seed, "offset_pair", tolerance, threshold),
                       search_cfg=_small_search(seed, "offset_pair"),
                       name="tune_offset", timestamps=data.timestamps)
        rec = score_recovery(res, data.planted)
        compact = len(res.portfolio) < 120 and not scaled_slack_rules(res)
        ok = rec.offset_pair >= 0.8 and compact
        score = (int(ok), rec.offset_pair, -len(res.portfolio), -abs(tolerance - 0.01), threshold)
        candidate = {
            "threshold": threshold,
            "tolerance": tolerance,
            "offset_recovery": rec.offset_pair,
            "accepted": len(res.portfolio),
            "ok": ok,
        }
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1]


def validate_runtime_config(discovery: DiscoveryConfig, search: SearchConfig, seed: int = 0,
                            noise: float = 0.02) -> Dict[str, bool]:
    """Check the exact returned runtime config on a representative noisy proxy grid."""
    out: Dict[str, bool] = {}
    for family in _PLANT_FAMILIES:
        data = S.make_synthetic(n_entities=4, n_snapshots=120, noise=noise, seed=seed,
                                families=(family,))
        res = discover(data.columns, data.matrix, inducer=make_inducer("subagent"),
                       discovery_cfg=discovery, search_cfg=search,
                       name=f"runtime_{family}", timestamps=data.timestamps)
        rec = score_recovery(res, data.planted)
        out[family] = bool(getattr(rec, family) >= 0.8)
    return out


def proxy_tune(seed: int = 0) -> dict:
    tuned_pair = tune_threshold_tolerance(seed)
    pr = plant_and_recover(seed=seed)
    family_ok = {fam: any(by_noise.values()) for fam, by_noise in pr.recovered.items()}
    runtime_discovery = DiscoveryConfig(
        seed=seed,
        tolerance=max(0.05, tuned_pair["tolerance"]),
        hold_rate_threshold=tuned_pair["threshold"],
    )
    runtime_search = SearchConfig(seed=seed)
    runtime_recovery = validate_runtime_config(runtime_discovery, runtime_search, seed)
    return {
        "discovery": runtime_discovery,
        "search": runtime_search,
        "runtime_discovery": runtime_discovery,
        "runtime_search": runtime_search,
        "runtime_recovery": runtime_recovery,
        "tuned_threshold_tolerance": tuned_pair,
        "family_ok": family_ok,
        "ok": all(family_ok.values()) and all(runtime_recovery.values()),
    }


def frozen_crosscheck_eval(frames: Dict[str, object], seed: int = 0) -> dict:
    tuned = proxy_tune(seed)
    out = {}
    for name, df in frames.items():
        res = discover_dataframe(df, inducer=make_inducer("subagent"),
                                 discovery_cfg=tuned["discovery"], search_cfg=tuned["search"],
                                 name=name)
        out[name] = {
            "accepted": len(res.portfolio),
            "rules": [e.rule.unparse() for e in res.portfolio],
            "families": structural_families(res),
        }
    return out


def structural_families(result: DiscoveryResult) -> List[str]:
    fams = set()
    for ev in result.portfolio:
        atom = ev.rule.atom
        if atom.op == "!=":
            fams.add("same-family separation")
        elif atom.op in (">=", "<=") and (isinstance(atom.left, A.Const) or isinstance(atom.right, A.Const)):
            fams.add("one-sided nonnegativity/bound")
        elif any(isinstance(t, A.Agg) and t.kind == "SUM" for t in (atom.left, atom.right)):
            fams.add("aggregate sum conservation")
        elif atom.op == "<|>":
            fams.add("presence/existence pairing")
        elif isinstance(atom.left, A.Ref) and isinstance(atom.right, A.Ref):
            fams.add("pairwise equality/order")
    return sorted(fams)


def _term_has_scaled_slack(term) -> bool:
    if isinstance(term, A.Scale):
        return term.coeff < 0.0 or abs(term.coeff) < 1.0
    if isinstance(term, A.Add):
        return any(_term_has_scaled_slack(t) for t in term.terms)
    return False


def is_scaled_slack_rule(rule: A.Rule) -> bool:
    return rule.atom.op in ("<=", ">=") and (
        _term_has_scaled_slack(rule.atom.left) or _term_has_scaled_slack(rule.atom.right)
    )


def scaled_slack_rules(result: DiscoveryResult) -> List[str]:
    return [e.rule.unparse() for e in result.portfolio if is_scaled_slack_rule(e.rule)]


def portfolio_quality(seed: int = 0) -> dict:
    data = S.make_synthetic(n_entities=5, n_snapshots=220, noise=0.02, seed=seed)
    res = discover(data.columns, data.matrix, discovery_cfg=_fast_eval(seed),
                   search_cfg=_small_search(seed), name="quality", timestamps=data.timestamps)
    bad = [e.rule.unparse() for e in res.portfolio if e.hold_rate_lo < 0.9]
    slack = scaled_slack_rules(res)
    independent = len(res.portfolio)
    total = len(res.archive.representatives())
    return {"ok": bool(res.portfolio and not bad), "accepted": len(res.portfolio),
            "archive_representatives": total,
            "independent_survivors": independent,
            "compactness_ratio": (independent / total) if total else 0.0,
            "rules": [e.rule.unparse() for e in res.portfolio], "bad_hold_rate_rules": bad,
            "scaled_slack_rules": slack, "families": structural_families(res)}


def run_all(seed: int = 0) -> dict:
    tuned = proxy_tune(seed)
    pq = portfolio_quality(seed)
    return {
        "proxy_ok": bool(tuned["ok"]),
        "synthetic_recovery": {"ok": bool(tuned["ok"]), "families": tuned["family_ok"]},
        "tuned_threshold_tolerance": tuned["tuned_threshold_tolerance"],
        "runtime_recovery": tuned["runtime_recovery"],
        "null_equalities_accepted": null_accepted(seed=seed),
        "portfolio_quality": pq,
    }
