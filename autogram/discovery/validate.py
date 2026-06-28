"""Validation without ground truth: adversarial sanity checks and proxy signals.

The engine has no oracle, so we prove discovery through proxy signals (held-out coverage with a
Wilson CI, stability, lift percentile, MDL gain) and adversarial checks the *system never sees
the answer to*:

* **plant-and-recover** -- plant rules in synthetic data and sweep the noise; check the engine
  recovers them while self-calibration tracks the noise.
* **null dataset** -- independent columns must yield ~no accepted rules (FDR control).
* **tautology rejection** -- self-comparisons are inadmissible and spurious pairings get ~0 lift.
* **rename invariance** -- consistently renaming columns recovers the same relationships.
* **ablations** -- dropping stability admits unstable/overfit rules (a regime rule that holds
  on most rows but breaks on a late time block); disabling the lift/null guard admits spurious
  null-correlated rules; dropping name-induction fails on a renamed schema.

Recovery is judged by *grounding* discovered rules back to the planted column relationships --
never by matching a rule's textual form -- so it is robust to how roles happen to be named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..config import DiscoveryConfig, SearchConfig
from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_family, resolve_ref
from ..dsl.grammar import grammar_from_adapter
from ..dsl.typecheck import is_admissible
from .induce import HeuristicInducer, SchemaInducer
from .loop import DiscoveryResult, discover
from . import synth as S


# ---------------------------------------------------------------------------
# Grounded relation extraction (the recovery judge)
# ---------------------------------------------------------------------------

def _operand_sig(left, right, binder, binding, nm):
    """Canonical relation signature for one binding (column-name based, role-name free)."""
    def ref_col(t):
        return resolve_ref(t.role, binder, binding, nm) if isinstance(t, A.Ref) else None

    def fam_cols(t):
        return resolve_family(t.family_role, binder, binding, nm) if isinstance(t, A.Agg) else None

    lc, rc = ref_col(left), ref_col(right)
    if lc is not None and rc is not None:
        return ("pair", frozenset({lc, rc}))
    # Ref vs SUM(family)
    for a, b in ((left, right), (right, left)):
        if isinstance(a, A.Ref) and isinstance(b, A.Agg) and b.kind == "SUM":
            ac = ref_col(a)
            bc = fam_cols(b)
            if ac is not None and bc:
                return ("ref_sum", (ac, frozenset(bc)))
    # Ref vs 0
    for a, b in ((left, right), (right, left)):
        if isinstance(a, A.Ref) and isinstance(b, A.Const) and b.value == 0:
            ac = ref_col(a)
            if ac is not None:
                return ("zero", ac)
    return None


def rule_relations(rule: A.Rule, ds) -> set:
    nm = ds.name_model
    rels = set()
    for b in enumerate_bindings(rule.binder, nm):
        sig = _operand_sig(rule.atom.left, rule.atom.right, rule.binder, b, nm)
        if sig is not None:
            rels.add(sig)
    return rels


def portfolio_relations(result: DiscoveryResult) -> set:
    rels: set = set()
    for ev in result.portfolio:
        rels |= rule_relations(ev.rule, result.dataset)
    return rels


# ---------------------------------------------------------------------------
# Recovery scoring against planted structure
# ---------------------------------------------------------------------------

def _pairs(rels: set) -> set:
    return {fs for tag, fs in rels if tag == "pair"}


def _refsums(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "ref_sum"}


def _zeros(rels: set) -> set:
    return {payload for tag, payload in rels if tag == "zero"}


@dataclass
class Recovery:
    two_end: float
    row_sum: float
    col_sum: float
    self_zero: float
    recovered: bool

    def as_dict(self) -> dict:
        return {"two_end": self.two_end, "row_sum": self.row_sum,
                "col_sum": self.col_sum, "self_zero": self.self_zero,
                "recovered": self.recovered}


def score_recovery(result: DiscoveryResult, planted: dict, frac: float = 0.9) -> Recovery:
    rels = portfolio_relations(result)
    pairs, refsums, zeros = _pairs(rels), _refsums(rels), _zeros(rels)

    def cov(found, target):
        target = set(target)
        if not target:
            return 0.0
        return len(found & target) / len(target)

    two = cov(pairs, planted.get("two_end", set()))
    row = cov(refsums, set(planted.get("row_sum", [])))
    col = cov(refsums, set(planted.get("col_sum", [])))
    sz = cov(zeros, set(planted.get("self_zero", [])))
    recovered = (two >= frac) or (row >= frac) or (col >= frac) or (sz >= frac)
    return Recovery(two, row, col, sz, recovered)


# ---------------------------------------------------------------------------
# Adversarial checks
# ---------------------------------------------------------------------------

def _small_search(seed: int = 0, rounds: int = 5, proposals: int = 90) -> SearchConfig:
    return SearchConfig(rounds=rounds, proposals_per_round=proposals, seed=seed)


def _fast_eval(seed: int = 0, n_perm: int = 16) -> DiscoveryConfig:
    return DiscoveryConfig(n_perm=n_perm, seed=seed)


@dataclass
class PlantRecover:
    noise_levels: List[float]
    recovered: Dict[str, Dict[float, bool]] = field(default_factory=dict)
    detail: Dict[str, Dict[float, dict]] = field(default_factory=dict)


_PLANT_FAMILIES = ("row_sum", "col_sum", "two_end", "self_zero")


def _family_recovered(rec: Recovery, family: str) -> bool:
    return bool(getattr(rec, family) >= 0.9)


def plant_and_recover(noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.1),
                      n_entities: int = 5, n_snapshots: int = 300,
                      seed: int = 0, inducer: Optional[SchemaInducer] = None) -> PlantRecover:
    pr = PlantRecover(noise_levels=list(noise_levels))
    for family in _PLANT_FAMILIES:
        pr.recovered[family] = {}
        pr.detail[family] = {}
        for nz in noise_levels:
            data = S.make_synthetic(n_entities=n_entities, n_snapshots=n_snapshots,
                                   noise=nz, seed=seed, families=(family,))
            res = discover(data.columns, data.matrix, inducer=inducer,
                          discovery_cfg=_fast_eval(seed),
                          search_cfg=_small_search(seed, rounds=5, proposals=120),
                          name=f"plant_{family}_n{nz}", timestamps=data.timestamps)
            rec = score_recovery(res, data.planted)
            found = _family_recovered(rec, family)
            pr.recovered[family][nz] = found
            pr.detail[family][nz] = {
                **rec.as_dict(),
                "family_recovered": found,
                "n_accepted": len(res.portfolio),
                "rules": [e.rule.unparse() for e in res.portfolio],
                "operating_cov": [round(e.operating_cov, 3) for e in res.portfolio],
            }
    return pr


def null_accepted(n_entities: int = 5, n_snapshots: int = 300, seed: int = 0,
                  inducer: Optional[SchemaInducer] = None) -> int:
    data = S.make_null(n_entities=n_entities, n_snapshots=n_snapshots, seed=seed)
    res = discover(data.columns, data.matrix, inducer=inducer,
                   discovery_cfg=_fast_eval(seed), search_cfg=_small_search(seed),
                   name="null", timestamps=data.timestamps)
    return len(res.portfolio)


@dataclass
class TautologyCheck:
    self_comparison_admissible: bool
    nonneg_accepted: bool


def tautology_check(seed: int = 0, inducer: Optional[SchemaInducer] = None) -> TautologyCheck:
    """Self-comparisons must be inadmissible; non-pairing ordering forms must not be accepted."""
    data = S.make_synthetic(n_entities=5, n_snapshots=200, noise=0.02, seed=seed)
    res = discover(data.columns, data.matrix, inducer=inducer,
                   discovery_cfg=_fast_eval(seed), search_cfg=_small_search(seed, rounds=2),
                   name="taut", timestamps=data.timestamps)
    G = res.grammar
    taut = A.Rule("link", A.Compare(A.Ref(G.refs_for("link")[0]), "~=",
                                    A.Ref(G.refs_for("link")[0])))
    ok, _ = is_admissible(taut, G)
    # any accepted rule that is a bare "v >= 0" style (no name pairing) would be a tautology win
    nonneg = any(isinstance(e.rule.atom.right, A.Const) and e.rule.atom.right.value == 0
                 and e.rule.atom.op in (">=", "<=") for e in res.portfolio)
    return TautologyCheck(self_comparison_admissible=ok, nonneg_accepted=nonneg)


def _token_map(v1: S.Vocab, v2: S.Vocab, n_entities: int) -> Dict[str, str]:
    m = {v2.meas: v1.meas, v2.demand: v1.demand, v2.src: v1.src, v2.dst: v1.dst,
         v2.to: v1.to, v2.frm: v1.frm}
    for i in range(n_entities):
        m[v2.entity(i)] = v1.entity(i)
    return m


def _translate(rels: set, tokmap: Dict[str, str]) -> set:
    def tr(col: str) -> str:
        return "_".join(tokmap.get(t, t) for t in col.split("_"))

    out = set()
    for tag, payload in rels:
        if tag == "pair":
            out.add(("pair", frozenset(tr(c) for c in payload)))
        elif tag == "ref_sum":
            ref, fam = payload
            out.add(("ref_sum", (tr(ref), frozenset(tr(c) for c in fam))))
        elif tag == "zero":
            out.add(("zero", tr(payload)))
    return out


@dataclass
class RenameInvariance:
    base_relations: int
    renamed_relations: int
    overlap: float
    invariant: bool


def rename_invariance(n_entities: int = 5, n_snapshots: int = 300, seed: int = 0,
                      frac: float = 0.9) -> RenameInvariance:
    v1 = S.Vocab()
    v2 = S.Vocab(meas="signal", demand="route", src="out", dst="inn",
                 to="unto", frm="fro", entity_prefix="z")
    d1 = S.make_synthetic(n_entities=n_entities, n_snapshots=n_snapshots, noise=0.02,
                          seed=seed, vocab=v1)
    d2 = S.make_synthetic(n_entities=n_entities, n_snapshots=n_snapshots, noise=0.02,
                          seed=seed, vocab=v2)
    r1 = discover(d1.columns, d1.matrix, discovery_cfg=_fast_eval(seed),
                  search_cfg=_small_search(seed), name="base", timestamps=d1.timestamps)
    r2 = discover(d2.columns, d2.matrix, discovery_cfg=_fast_eval(seed),
                  search_cfg=_small_search(seed), name="renamed", timestamps=d2.timestamps)
    rel1 = portfolio_relations(r1)
    rel2 = _translate(portfolio_relations(r2), _token_map(v1, v2, n_entities))
    inter = rel1 & rel2
    denom = max(1, min(len(rel1), len(rel2)))
    overlap = len(inter) / denom
    return RenameInvariance(len(rel1), len(rel2), overlap,
                            invariant=overlap >= frac and len(rel1) > 0)


class FixedVocabInducer(SchemaInducer):
    """An inducer that ignores the actual names and always emits the *default* vocab spec.

    Used only by the ablation: on a renamed dataset its patterns no longer match, so nothing
    grounds and discovery fails -- demonstrating that real name induction is load-bearing.
    """

    def induce(self, columns, sample_rows=None):
        from .synth import Vocab
        fixed_cols = _default_schema_columns(Vocab(), n=6)
        return HeuristicInducer().induce(fixed_cols)


def _default_schema_columns(vocab: S.Vocab, n: int) -> List[str]:
    data = S.make_synthetic(n_entities=n, n_snapshots=2, noise=0.0, seed=0, vocab=vocab)
    return data.columns


@dataclass
class Ablations:
    drop_stability_more_overfit: bool
    drop_lift_admits_spurious: bool
    drop_induction_fails: bool
    detail: dict = field(default_factory=dict)


def _ablation_search(seed: int) -> SearchConfig:
    """A slightly larger budget so the regime trap is reliably surfaced by random search."""
    return SearchConfig(rounds=6, proposals_per_round=160, seed=seed)


def ablations(seed: int = 0) -> Ablations:
    detail: dict = {}

    # --- drop the lift/null guard (alpha=1, lift test removed) on NULL data ------------------
    # Claim demonstrated: with the name-permutation lift/null guard ENABLED, independent columns
    # yield ~no accepted rules (FDR control); DISABLING it admits spurious null-correlated rules
    # whose residual is not actually small relative to the magnitude-fair null.  (Direct literal
    # self-comparisons stay structurally inadmissible regardless of lift -- see dsl.typecheck --
    # so we phrase the claim as "spurious null-correlated rules", not "tautologies".)
    base_null = null_accepted(seed=seed)
    data = S.make_null(n_entities=5, n_snapshots=300, seed=seed)
    no_lift_cfg = DiscoveryConfig(n_perm=16, alpha=1.0, require_lift=False,
                                  require_null_support=False, require_parsimony=False,
                                  seed=seed)
    res_nolift = discover(data.columns, data.matrix, discovery_cfg=no_lift_cfg,
                          search_cfg=_small_search(seed), name="null_nolift",
                          timestamps=data.timestamps)
    pre_mdl_nolift = len(res_nolift.portfolio)
    # base must control false discovery (~0); dropping the lift guard is measured before the
    # parsimony screen, so the drop-lift signal cannot be gutted by rejecting negative-MDL rules.
    drop_lift = bool(base_null == 0 and pre_mdl_nolift > base_null and pre_mdl_nolift >= 3)
    detail["drop_lift"] = {"base_null_accepted": base_null,
                           "no_lift_accepted": pre_mdl_nolift,
                           "pre_mdl_no_lift_spurious": pre_mdl_nolift,
                           "no_lift_lift_percentiles":
                               [round(e.lift_percentile, 3) for e in res_nolift.portfolio]}

    # --- drop stability on a dataset that contains a genuine REGIME/OVERFIT trap -------------
    # The regime dataset keeps the stable demand row/column-sum invariants but makes the directed
    # two-end agreement hold only on the first 80% of snapshots (a late-block regime shift).  That
    # rule passes support and the name-permutation lift (its pooled residual is still small) but
    # its coverage collapses on the late time block, so it is admissible by every test EXCEPT
    # stability.  Hence enabling stability rejects it and disabling stability admits it: dropping
    # the stability gate demonstrably admits MORE (unstable) rules than the strict run.
    regime = S.make_synthetic(n_entities=5, n_snapshots=300, noise=0.05, seed=seed,
                              unstable_frac=0.2, regime_factor=1.8)
    strict = discover(regime.columns, regime.matrix, discovery_cfg=_fast_eval(seed),
                      search_cfg=_ablation_search(seed), name="strict",
                      timestamps=regime.timestamps)
    loose_cfg = DiscoveryConfig(n_perm=16, require_stability=False, seed=seed)
    loose = discover(regime.columns, regime.matrix, discovery_cfg=loose_cfg,
                     search_cfg=_ablation_search(seed), name="loose",
                     timestamps=regime.timestamps)
    # The overfit signature is the count of UNSTABLE rules (held-out coverage variance above the
    # stability tolerance) admitted into the portfolio.  The strict gate rejects every such rule
    # by construction, so this count is 0 under strict and >=1 under loose: dropping stability
    # demonstrably admits MORE overfit rules.  (We measure unstable-rule count rather than raw
    # portfolio size, which is polluted by stochastic stable demand-aggregate combos that both
    # runs may or may not surface.)
    strict_unstable = [e for e in strict.portfolio if e.stability_margin <= 0.0]
    loose_unstable = [e for e in loose.portfolio if e.stability_margin <= 0.0]
    drop_stability = bool(len(loose_unstable) > len(strict_unstable) and len(loose_unstable) >= 1)
    strict_rules = {e.rule.unparse() for e in strict.portfolio}
    detail["drop_stability"] = {
        "strict_count": len(strict.portfolio),
        "loose_count": len(loose.portfolio),
        "strict_unstable_count": len(strict_unstable),
        "loose_unstable_count": len(loose_unstable),
        "strict_rules": sorted(strict_rules),
        "loose_rules": sorted(e.rule.unparse() for e in loose.portfolio),
        "overfit_admitted_by_loose": [
            {"rule": e.rule.unparse(), "stability_std": round(e.stability_std, 3),
             "lift_percentile": round(e.lift_percentile, 4), "mdl_gain": round(e.mdl_gain, 3),
             "stability_margin": round(e.stability_margin, 4),
             "in_strict": e.rule.unparse() in strict_rules}
            for e in loose_unstable],
    }

    # --- drop name-induction: a fixed-vocab inducer on a renamed schema -> nothing grounds ----
    v2 = S.Vocab(meas="signal", demand="route", src="out", dst="inn",
                 to="unto", frm="fro", entity_prefix="z")
    renamed = S.make_synthetic(n_entities=5, n_snapshots=200, noise=0.02, seed=seed, vocab=v2)
    res_fixed = discover(renamed.columns, renamed.matrix, inducer=FixedVocabInducer(),
                         discovery_cfg=_fast_eval(seed), search_cfg=_small_search(seed),
                         name="fixed", timestamps=renamed.timestamps)
    drop_induction = len(res_fixed.portfolio) == 0
    detail["drop_induction"] = {"renamed_accepted": len(res_fixed.portfolio)}

    return Ablations(drop_stability, drop_lift, drop_induction, detail)


def portfolio_quality(seed: int = 0) -> dict:
    """Validate the delivered discovery portfolio against its own proxy metrics."""
    data = S.make_synthetic(n_entities=6, n_snapshots=400, noise=0.02, seed=seed)
    res = discover(data.columns, data.matrix,
                   discovery_cfg=DiscoveryConfig(n_perm=16, seed=seed),
                   search_cfg=SearchConfig(rounds=6, proposals_per_round=160, seed=seed),
                   name="quality", timestamps=data.timestamps)
    negative = [e.rule.unparse() for e in res.portfolio if e.mdl_gain <= 0.0]
    lift_fail = [e.rule.unparse() for e in res.portfolio
                 if not (e.lift > 1.0 and e.lift_percentile <= 0.05)]
    stability_fail = [e.rule.unparse() for e in res.portfolio if e.stability_margin <= 0.0]
    support_fail = [e.rule.unparse() for e in res.portfolio if e.support_margin <= 0.0]
    return {
        "ok": bool(res.portfolio and not negative and not lift_fail
                   and not stability_fail and not support_fail),
        "accepted": len(res.portfolio),
        "rules": [e.rule.unparse() for e in res.portfolio],
        "negative_mdl_rules": negative,
        "lift_fail_rules": lift_fail,
        "stability_fail_rules": stability_fail,
        "support_fail_rules": support_fail,
        "mdl_gains": [round(e.mdl_gain, 4) for e in res.portfolio],
        "lift_percentiles": [round(e.lift_percentile, 4) for e in res.portfolio],
        "breadth_justification": (
            "A one-rule portfolio is acceptable only when every delivered rule beats the "
            "null, is stable above the null support, and has positive MDL gain."),
    }


# ---------------------------------------------------------------------------
# Top-level report (human-readable proxy signals + checks)
# ---------------------------------------------------------------------------

def run_all(seed: int = 0) -> dict:
    pr = plant_and_recover(seed=seed)
    levels = pr.noise_levels
    family_ok = {}
    for family, by_noise in pr.recovered.items():
        recs = [bool(by_noise[l]) for l in levels]
        # the planted family must be recovered at low noise, and recovery must degrade
        # monotonically as noise rises (self-calibration tracking noise), never reappearing after
        # it is lost.  If a noisy run fails to recover the planted family, accepting any rule is
        # penalized as an unrecovered high-noise admission.
        monotone = all(recs[i] >= recs[i + 1] for i in range(len(recs) - 1))
        low_ok = bool(recs and recs[0] and (len(recs) == 1 or recs[1]))
        no_bad_admission = all(
            recs[i] or pr.detail[family][levels[i]]["n_accepted"] == 0
            for i in range(len(levels)))
        family_ok[family] = bool(low_ok and monotone and no_bad_admission)
    plant_ok = all(family_ok.values())
    abl = ablations(seed=seed)
    pq = portfolio_quality(seed=seed)
    out = {
        "plant_and_recover": {
            fam: {str(k): v for k, v in by_noise.items()}
            for fam, by_noise in pr.recovered.items()},
        "plant_detail": {
            fam: {str(k): v for k, v in by_noise.items()}
            for fam, by_noise in pr.detail.items()},
        "plant_family_ok": family_ok,
        "plant_ok": plant_ok,
        "null_accepted": null_accepted(seed=seed),
        "tautology": tautology_check(seed=seed).__dict__,
        "rename_invariance": rename_invariance(seed=seed).__dict__,
        "portfolio_quality": pq,
        "ablations": {
            "drop_stability_more_overfit": abl.drop_stability_more_overfit,
            "drop_lift_admits_spurious": abl.drop_lift_admits_spurious,
            "drop_induction_fails": abl.drop_induction_fails,
        },
        "ablation_detail": abl.detail,
    }
    return out
