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
from .loop import DiscoveryResult, discover, prepare_columns, run_prepared


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
    nonneg: float
    nonpos: float
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


def _portfolio_one_sided_columns(result: DiscoveryResult, op: str) -> set:
    """Columns C for which the portfolio contains ``[forall b] <ref over C> <op> 0`` (op in >=,<=).

    This is the one-sided analogue of :func:`portfolio_relations`: it grounds each one-sided rule
    over its bindings and collects the reference columns, so ``score_recovery`` can express nonneg /
    nonpos recovery as *column coverage* on the same numeric scale as the joint (pair/sum) families.
    """
    ds = result.dataset
    nm = ds.name_model
    cols: set = set()
    for ev in result.portfolio:
        atom = ev.rule.atom
        if atom.op != op:
            continue
        if not (isinstance(atom.right, A.Const) and float(atom.right.value) == 0.0):
            continue
        if not isinstance(atom.left, A.Ref):
            continue
        for b in enumerate_bindings(ev.rule.binder, nm):
            c = resolve_ref(atom.left.role, ev.rule.binder, b, nm)
            if c is not None:
                cols.add(c)
    return cols


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
    # explicit one-sided families: coverage of the planted nonneg / nonpos columns by >=0 / <=0 rules
    nn = cov(_portfolio_one_sided_columns(result, ">="), set(planted.get("nonneg", set())))
    npos = cov(_portfolio_one_sided_columns(result, "<="), set(planted.get("nonpos", set())))
    recovered = any(x >= frac for x in (two, row, col, sz, off, bal, pres, nn, npos))
    return Recovery(two, row, col, sz, off, bal, pres, nn, npos, recovered)


# --- joint proxy tuning (item: wire relevant proxy shapes into calibration) -----------------
#
# Calibration tunes ONE shared (tolerance, hold-rate threshold) on the whole selected proxy suite
# at once, instead of tuning on a single offset proxy and checking the null separately.  A setting
# is *eligible* only when every selected positive proxy reaches its recovery target (0.8), every
# produced portfolio is compact and free of scaled-slack rules, and the null control accepts zero
# ``~=`` / ``==`` rules.  Among eligible settings we prefer the strictest: smallest tolerance, then
# highest threshold, then smaller total portfolio, and only then strongest recovery.

_PROXY_RECOVERY_TARGET = 0.8


@dataclass
class ProxyOutcome:
    """One positive proxy's result at a single (tolerance, threshold) grid cell."""
    shape: str
    recovery: float
    accepted: int
    compact: bool
    scaled_slack: List[str] = field(default_factory=list)


@dataclass
class GridCandidate:
    """A (tolerance, threshold) setting scored on every selected proxy plus the null control."""
    tolerance: float
    hold_rate_threshold: float
    proxies: List[ProxyOutcome]
    null_equalities: int

    @property
    def recovery_ok(self) -> bool:
        return all(p.recovery >= _PROXY_RECOVERY_TARGET for p in self.proxies)

    @property
    def compact_ok(self) -> bool:
        return all(p.compact and not p.scaled_slack for p in self.proxies)

    @property
    def null_safe(self) -> bool:
        return self.null_equalities == 0

    @property
    def eligible(self) -> bool:
        return self.recovery_ok and self.compact_ok and self.null_safe

    @property
    def total_portfolio(self) -> int:
        return sum(p.accepted for p in self.proxies)

    @property
    def recovery_strength(self) -> float:
        return min((p.recovery for p in self.proxies), default=0.0)

    def evidence(self) -> dict:
        return {
            "tolerance": round(self.tolerance, 4),
            "hold_rate_threshold": round(self.hold_rate_threshold, 4),
            "null_equalities": self.null_equalities,
            "eligible": self.eligible,
            "recovery_ok": self.recovery_ok,
            "compact_ok": self.compact_ok,
            "null_safe": self.null_safe,
            "proxies": [{"shape": p.shape, "recovery": round(p.recovery, 4),
                         "accepted": p.accepted, "compact": p.compact,
                         "scaled_slack": p.scaled_slack} for p in self.proxies],
        }


def _candidate_pref_key(c: GridCandidate):
    # smallest tolerance, then highest threshold, then smaller total portfolio, then strongest
    # recovery (recovery strength is only a later tie-breaker, never the primary objective).
    return (c.tolerance, -c.hold_rate_threshold, c.total_portfolio, -c.recovery_strength)


def select_candidate(candidates) -> Optional[GridCandidate]:
    """The strictest eligible candidate, or ``None`` when none is eligible."""
    eligible = [c for c in candidates if c.eligible]
    if not eligible:
        return None
    return min(eligible, key=_candidate_pref_key)


class CalibrationGridError(RuntimeError):
    """No (tolerance, threshold) setting recovered every selected proxy under the null floor."""


@dataclass
class PreparedProxy:
    """A proxy's dataset + grammar, induced/compiled ONCE and reused across grid cells and rungs."""
    shape: str
    ds: object
    G: object
    planted: dict


@dataclass
class ProxySuite:
    """The selected positive proxies plus the always-on null control, each prepared once."""
    positives: List[PreparedProxy]
    null: PreparedProxy

    def shapes(self) -> List[str]:
        return [p.shape for p in self.positives]


def prepare_proxy_suite(regime, seed: int = 0,
                        inducer: Optional[SchemaInducer] = None) -> ProxySuite:
    """Generate + induce every active positive proxy and the null control exactly once.

    The prepared (dataset, grammar) pairs are reused for every joint-tuning grid cell and for every
    relaxation-ladder rung's null gate, so schema induction happens once per proxy.  Proxy values
    are synthetic and only ever feed the *proxy* grammars; real-data grammar induction is untouched
    (it still receives only the dataset's own column names).
    """
    from .regime import KNOWN_SHAPES, generate as _gen, generate_null as _gen_null
    # Validate every active shape up front: a custom regime built directly (bypassing
    # RegimeSpec.add) can carry an unknown shape, which would otherwise fail obscurely later at
    # make_synthetic (empty planted set) or getattr(rec, shape).  Fail loudly *before* generation.
    unknown = sorted({e.shape for e in regime.active_entries() if e.shape not in KNOWN_SHAPES})
    if unknown:
        raise ValueError(
            f"custom regime has unknown proxy shape(s) {unknown}; choose from {list(KNOWN_SHAPES)}")
    inducer = inducer or make_inducer("subagent")
    positives: List[PreparedProxy] = []
    for e in regime.active_entries():
        data = _gen(e, seed=seed)
        ds, G, _spec = prepare_columns(data.columns, data.matrix, inducer=inducer,
                                       search_cfg=_small_search(seed, e.shape),
                                       name=f"proxy_{e.shape}", timestamps=data.timestamps)
        positives.append(PreparedProxy(e.shape, ds, G, data.planted))
    ndata = _gen_null(seed=seed)
    nds, nG, _nspec = prepare_columns(ndata.columns, ndata.matrix, inducer=inducer,
                                      search_cfg=_small_search(seed), name="null_proxy",
                                      timestamps=ndata.timestamps)
    return ProxySuite(positives=positives, null=PreparedProxy("null", nds, nG, {}))


def null_equalities_at(prepared_null: PreparedProxy, dcfg: DiscoveryConfig, seed: int = 0) -> int:
    """Count ``~=`` / ``==`` rules the prepared null accepts at ``dcfg`` (reusing its grammar)."""
    res = run_prepared(prepared_null.ds, prepared_null.G, discovery_cfg=dcfg,
                       search_cfg=_small_search(seed))
    return len([e for e in res.portfolio if e.rule.atom.op in ("~=", "==")])


def evaluate_grid_candidate(suite: ProxySuite, tolerance: float, hold_rate_threshold: float,
                            seed: int = 0, band_mode: str = "global") -> GridCandidate:
    """Score one (tolerance, threshold) on every prepared positive proxy plus the null control."""
    proxies: List[ProxyOutcome] = []
    for p in suite.positives:
        dcfg = DiscoveryConfig(seed=seed, tolerance=tolerance,
                               hold_rate_threshold=hold_rate_threshold, band_mode=band_mode)
        res = run_prepared(p.ds, p.G, discovery_cfg=dcfg, search_cfg=_small_search(seed, p.shape))
        rec = score_recovery(res, p.planted)
        slack = scaled_slack_rules(res)
        proxies.append(ProxyOutcome(
            shape=p.shape,
            recovery=float(getattr(rec, p.shape)),
            accepted=len(res.portfolio),
            compact=len(res.portfolio) < 120 and not slack,
            scaled_slack=slack,
        ))
    ndcfg = DiscoveryConfig(seed=seed, tolerance=tolerance,
                            hold_rate_threshold=hold_rate_threshold, band_mode=band_mode)
    return GridCandidate(tolerance, hold_rate_threshold, proxies,
                         null_equalities_at(suite.null, ndcfg, seed))


_BASE_THRESHOLDS = (0.58, 0.62, 0.66, 0.72)
_BASE_TOLERANCES = (0.005, 0.01, 0.02, 0.05)


def tune_joint(suite, seed: int = 0, band_mode: str = "global", null_floor: float = 0.5,
               max_expansions: int = 3, thresholds=None, tolerances=None, evaluate=None) -> dict:
    """Jointly pick one (tolerance, threshold) that safely recovers *every* selected proxy.

    Sweeps a base grid; if nothing is eligible it expands in the direction implied by the failure
    (threshold down toward the null floor, tolerance up), bounded by ``max_expansions``.  If still
    nothing is eligible it raises :class:`CalibrationGridError` with per-proxy evidence.  Every grid
    cell reuses the once-prepared proxy grammars in ``suite``.  ``evaluate`` is injectable so the
    orchestration can be exercised without running discovery.
    """
    evaluate = evaluate or evaluate_grid_candidate
    thresholds = list(thresholds or _BASE_THRESHOLDS)
    tolerances = list(tolerances or _BASE_TOLERANCES)
    expansions = 0
    candidates: List[GridCandidate] = []
    best: Optional[GridCandidate] = None
    # Memoize by (tolerance, threshold): each expansion re-lists earlier cells, so caching keeps
    # every unique grid cell evaluated exactly once across the growing grids.
    cache: dict = {}

    def _cell(tol, thr):
        key = (tol, thr)
        if key not in cache:
            cache[key] = evaluate(suite, tol, thr, seed=seed, band_mode=band_mode)
        return cache[key]

    while True:
        candidates = [_cell(tol, thr) for thr in thresholds for tol in tolerances]
        best = select_candidate(candidates)
        if best is not None or expansions >= max_expansions:
            break
        expansions += 1
        new_thr = max(null_floor, round(min(thresholds) - 0.04, 4))
        new_tol = round(max(tolerances) * 2.0, 4)
        thresholds = sorted({t for t in thresholds + [new_thr] if t >= null_floor})
        tolerances = sorted(set(tolerances + [new_tol]))
    if best is None:
        evidence = [c.evidence()
                    for c in sorted(candidates, key=lambda c: -c.recovery_strength)[:6]]
        raise CalibrationGridError(
            "no (tolerance, threshold) setting recovered every selected proxy under the null floor "
            f"after {expansions} grid expansion(s); per-proxy evidence: {evidence}")
    return {
        "tolerance": best.tolerance,
        "hold_rate_threshold": best.hold_rate_threshold,
        "expansions": expansions,
        "selected_null_equalities": best.null_equalities,
        "proxy_shapes": [p.shape for p in best.proxies],
        "per_proxy": [{"shape": p.shape, "recovery": round(p.recovery, 4), "accepted": p.accepted,
                       "compact": p.compact, "scaled_slack": p.scaled_slack}
                      for p in best.proxies],
    }


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
                  inducer: Optional[SchemaInducer] = None, regime=None) -> int:
    # The always-on false-discovery control proxy comes from the (wired) RegimeSpec when given.
    if regime is not None:
        from .regime import generate_null as _regime_null
        data = _regime_null(seed=seed, n_entities=n_entities, n_snapshots=n_snapshots)
    else:
        data = S.make_null(n_entities=n_entities, n_snapshots=n_snapshots, seed=seed)
    res = discover(data.columns, data.matrix, inducer=inducer or make_inducer("subagent"),
                   discovery_cfg=_fast_eval(seed), search_cfg=_small_search(seed),
                   name="null", timestamps=data.timestamps)
    return len([e for e in res.portfolio if e.rule.atom.op in ("~=", "==")])


def _offset_sweep(thresholds, tolerances, data, seed):
    # Induce the proxy schema ONCE, then re-evaluate across the (threshold, tolerance) grid.
    # The induced grammar depends only on the synthetic column names, not on the numeric knobs
    # being swept, so re-inducing per grid cell was pure redundant cost (and a flakiness source).
    from .loop import prepare_columns, run_prepared
    ds, G, _spec = prepare_columns(data.columns, data.matrix, inducer=make_inducer("subagent"),
                                   search_cfg=_small_search(seed, "offset_pair"),
                                   name="tune_offset", timestamps=data.timestamps)
    best = None
    for threshold, tolerance in itertools.product(thresholds, tolerances):
        res = run_prepared(ds, G,
                           discovery_cfg=_fast_eval(seed, "offset_pair", tolerance, threshold),
                           search_cfg=_small_search(seed, "offset_pair"))
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
    return best


def tune_threshold_tolerance(seed: int = 0, max_expansions: int = 3,
                             null_floor: float = 0.5, regime=None) -> dict:
    """Jointly pick a tolerance/threshold pair on the approximate-offset proxy.

    Starts on a base grid and, if no pair fits, widens the ranges (threshold down toward the
    ``null_floor``, tolerance up) and re-sweeps -- so calibration adapts to datasets whose
    operating point sits outside the shipped grid (item 3).  Expansion is bounded and the
    threshold never drops below the false-discovery ``null_floor``.

    When a ``regime`` (RegimeSpec) is supplied the approximate-offset proxy is generated from its
    ``offset_pair`` entry, so the tuner's synthetic data is the wired, editable proxy suite rather
    than a hard-coded generator.
    """
    if regime is not None:
        from .regime import generate as _regime_generate
        offset = next((e for e in regime.active_entries() if e.shape == "offset_pair"), None)
        data = (_regime_generate(offset, seed=seed) if offset is not None
                else S.make_synthetic(n_entities=3, n_snapshots=120, noise=0.0, seed=seed,
                                      families=("offset_pair",), offset_hold_rate=0.67, offset_factor=0.98))
    else:
        data = S.make_synthetic(n_entities=3, n_snapshots=120, noise=0.0, seed=seed,
                                families=("offset_pair",), offset_hold_rate=0.67, offset_factor=0.98)
    thresholds = [0.58, 0.62, 0.66, 0.72]
    tolerances = [0.005, 0.01, 0.02, 0.05]
    best = _offset_sweep(thresholds, tolerances, data, seed)
    expansions = 0
    while not best[1]["ok"] and expansions < max_expansions:
        expansions += 1
        new_thr = max(null_floor, round(min(thresholds) - 0.04, 4))
        new_tol = round(max(tolerances) * 2.0, 4)
        thresholds = sorted({t for t in thresholds + [new_thr] if t >= null_floor})
        tolerances = sorted(set(tolerances + [new_tol]))
        best = _offset_sweep(thresholds, tolerances, data, seed)
    result = dict(best[1])
    result["expansions"] = expansions
    return result


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
