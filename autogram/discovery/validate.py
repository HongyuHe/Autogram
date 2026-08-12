"""Synthetic proxy and two-phase evaluation helpers for v2.

The proxy phase tunes generic knobs on synthetic or observed-only data.  The frozen phase runs the
same pipeline on CrossCheck DataFrames without reading clean frames or a target-invariant catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import numbers
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import DiscoveryConfig, SearchConfig
from ..dsl import ast as A
from ..dsl.binders import enumerate_bindings, resolve_family, resolve_ref
from ..dsl.evaluate import robust_median
from ..loader.loader import Dataset, Frame
from ..schema.spec import RelatedTemplate
from . import synth as S
from .induce import SchemaInducer, make_inducer
from .loop import DiscoveryResult, discover, prepare_columns, run_prepared
from .propose import EnumerationProposer
from .regime import KNOWN_SHAPES


def _equality_relation(op: str, base: tuple) -> tuple:
    return (
        "equality",
        "exact" if op == "==" else "approximate",
        base,
    )


def _unwrap_equality_relation(relation: tuple):
    if (
        isinstance(relation, tuple)
        and len(relation) == 3
        and relation[0] == "equality"
    ):
        return relation[1], relation[2]
    return None, relation


def _with_condition(rule, base):
    """Wrap a signature in its rule's condition so conditioned rules never look unconditional.

    Every atom kind must go through this. A conditioned rule states a strictly weaker claim than
    the same rule without the condition, so if the condition is dropped from the signature the
    two become indistinguishable: a conditioned discovery would be credited with recovering an
    unconditional known invariant, and a genuinely conditioned known invariant could never match.
    """
    if base is None or rule.condition is None:
        return base
    return ("conditional", (_condition_signature(rule.condition), base))


def _operand_sig(rule, binder, binding, nm, parameters=None):
    if isinstance(rule.atom, A.BooleanDefinition):
        target = (
            resolve_ref(rule.atom.target.role, binder, binding, nm)
            if isinstance(rule.atom.target, A.Ref)
            else None
        )
        predicate = _predicate_signature(
            rule.atom.predicate,
            binder,
            binding,
            nm,
            parameters or {},
        )
        if target is None or predicate is None:
            return None
        tag = (
            "sustained_definition"
            if isinstance(rule.atom.predicate, A.Sustained)
            else "conjunction_definition"
        )
        return _with_condition(rule, (tag, (target, predicate)))
    if isinstance(rule.atom, A.CategoryDefinition):
        return _with_condition(
            rule,
            (
                "categorical_definition",
                (
                    rule.atom.target_column,
                    tuple(rule.atom.cases),
                    rule.atom.default,
                ),
            ),
        )
    if isinstance(rule.atom, A.BandDefinition):
        column = (
            resolve_ref(rule.atom.term.role, binder, binding, nm)
            if isinstance(rule.atom.term, A.Ref)
            else None
        )
        center = (
            rule.atom.center
            if rule.atom.center is not None
            else (parameters or {}).get("center")
        )
        if column is None or center is None:
            return None
        base = ("healthy_band", (column, float(center)))
        return _with_condition(rule, base)
    base = _operand_sig_base(rule, binder, binding, nm)
    return _with_condition(rule, base)


def _condition_signature(condition: A.Condition):
    if condition.op == "all":
        return (
            "all",
            tuple(sorted((
                _condition_signature(child)
                for child in condition.values
                if isinstance(child, A.Condition)
            ), key=str)),
        )
    return (
        condition.column,
        condition.op,
        tuple(sorted(condition.values, key=str)),
    )


def _operand_sig_base(rule, binder, binding, nm):
    left, right, op = rule.atom.left, rule.atom.right, rule.atom.op

    def ref_col(t):
        return resolve_ref(t.role, binder, binding, nm) if isinstance(t, A.Ref) else None

    def fam_cols(t):
        return resolve_family(t.family_role, binder, binding, nm) if isinstance(t, A.Agg) else None

    lc, rc = ref_col(left), ref_col(right)
    if lc is not None and rc is not None and op in ("~=", "=="):
        return _equality_relation(
            op,
            ("pair", frozenset({lc, rc})),
        )
    if lc is not None and rc is not None and op == "<|>":
        return ("presence_pair", frozenset({lc, rc}))
    if lc is not None and rc is not None and op == "~∝":
        return ("proportional", (lc, rc))
    if lc is not None and rc is not None and op == "!=":
        return ("separation_pair", frozenset({lc, rc}))
    for a, b in ((left, right), (right, left)):
        if (
            op in ("~=", "==")
            and isinstance(a, A.Ref)
            and isinstance(b, A.RelatedAgg)
        ):
            column = ref_col(a)
            if column is not None:
                return _equality_relation(
                    op,
                    ("related_aggregate", (column, b.role)),
                )
    for a, b in ((left, right), (right, left)):
        if (
            op in (">=", "<=", ">", "<")
            and isinstance(a, A.Lag)
            and isinstance(a.term, A.Ref)
            and isinstance(b, A.Const)
            and float(b.value) == 0.0
        ):
            column = resolve_ref(a.term.role, binder, binding, nm)
            if column is not None:
                reverse = {
                    "<=": ">=",
                    ">=": "<=",
                    "<": ">",
                    ">": "<",
                }
                effective_op = op if a is left else reverse[op]
                return (
                    "lag_bound",
                    (column, int(a.steps), effective_op),
                )
    for a, b in ((left, right), (right, left)):
        if (
            op in (">=", "<=", ">", "<")
            and isinstance(a, A.Diff)
            and isinstance(a.term, A.Ref)
            and isinstance(b, A.Const)
            and float(b.value) == 0.0
        ):
            column = resolve_ref(a.term.role, binder, binding, nm)
            if column is not None:
                reverse = {"<=": ">=", ">=": "<=", "<": ">", ">": "<"}
                effective_op = op if a is left else reverse[op]
                return ("delta_bound", (column, int(a.steps), effective_op))
    for a, b in ((left, right), (right, left)):
        if (
            op in ("~=", "==")
            and isinstance(a, A.Diff)
            and isinstance(a.term, A.Ref)
            and isinstance(b, A.Const)
            and float(b.value) == 0.0
        ):
            column = resolve_ref(a.term.role, binder, binding, nm)
            if column is not None:
                return _equality_relation(
                    op,
                    ("delta_zero", (column, int(a.steps))),
                )
    for a, b in ((left, right), (right, left)):
        if op in ("~=", "==") and isinstance(a, A.Ref) and isinstance(b, A.Div):
            ac = ref_col(a)
            num = ref_col(b.num)
            den = ref_col(b.den)
            if ac is not None and num is not None and den is not None:
                return _equality_relation(
                    op,
                    ("ratio", (ac, num, den)),
                )
            if (
                ac is not None
                and isinstance(b.num, A.Rolling)
                and isinstance(b.den, A.Rolling)
                and b.num.kind == b.den.kind == "SUM"
                and b.num.window == b.den.window
                and isinstance(b.num.term, A.Ref)
                and isinstance(b.den.term, A.Ref)
            ):
                num_col = resolve_ref(b.num.term.role, binder, binding, nm)
                den_col = resolve_ref(b.den.term.role, binder, binding, nm)
                if num_col is not None and den_col is not None:
                    return _equality_relation(
                        op,
                        (
                            "windowed_ratio",
                            (ac, num_col, den_col, int(b.num.window)),
                        ),
                    )
    for a, b in ((left, right), (right, left)):
        if op in ("~=", "==") and isinstance(a, A.Ref) and isinstance(b, A.Agg) and b.kind == "SUM":
            ac = ref_col(a)
            bc = fam_cols(b)
            if ac is not None and bc:
                return _equality_relation(
                    op,
                    ("ref_sum", (ac, frozenset(bc))),
                )
    if (
        op in ("~=", "==")
        and isinstance(left, A.Agg)
        and isinstance(right, A.Agg)
        and left.kind == right.kind == "SUM"
    ):
        left_columns = fam_cols(left)
        right_columns = fam_cols(right)
        if left_columns and right_columns:
            return _equality_relation(
                op,
                (
                    "sum_balance",
                    frozenset({
                        frozenset(left_columns),
                        frozenset(right_columns),
                    }),
                ),
            )
    if op in ("~=", "==") and isinstance(left, A.Add) and isinstance(right, A.Add):
        lsig = _add_ref_agg_sig(left, binder, binding, nm)
        rsig = _add_ref_agg_sig(right, binder, binding, nm)
        if lsig is not None and rsig is not None:
            return _equality_relation(
                op,
                ("agg_ref_balance", frozenset({lsig, rsig})),
            )
    for a, b in ((left, right), (right, left)):
        if op in ("~=", "==") and isinstance(a, A.Ref) and isinstance(b, A.Const) and b.value == 0:
            ac = ref_col(a)
            if ac is not None:
                return _equality_relation(op, ("zero", ac))
    return None


def _predicate_signature(predicate, binder, binding, nm, parameters):
    if isinstance(predicate, A.Bound):
        term = _term_signature(predicate.term, binder, binding, nm)
        threshold = predicate.threshold
        if threshold is None:
            threshold = (
                parameters.get("thresholds", {})
                .get(predicate.unparse())
            )
        return None if term is None or threshold is None else (
            "bound",
            term,
            predicate.op,
            float(threshold),
        )
    if isinstance(predicate, A.Sustained):
        inner = _predicate_signature(
            predicate.predicate,
            binder,
            binding,
            nm,
            parameters,
        )
        return None if inner is None else ("sustained", int(predicate.window), inner)
    if isinstance(predicate, A.Conjunction):
        children = tuple(sorted((
            _predicate_signature(item, binder, binding, nm, parameters)
            for item in predicate.predicates
        ), key=str))
        return None if any(child is None for child in children) else children
    return None


def _term_signature(term, binder, binding, nm):
    if isinstance(term, A.Ref):
        column = resolve_ref(term.role, binder, binding, nm)
        return None if column is None else ("ref", column)
    if isinstance(term, A.Diff):
        inner = _term_signature(term.term, binder, binding, nm)
        return None if inner is None else ("delta", inner, int(term.steps))
    if isinstance(term, A.Rolling):
        inner = _term_signature(term.term, binder, binding, nm)
        return None if inner is None else (
            "rolling",
            term.kind,
            int(term.window),
            inner,
        )
    if isinstance(term, A.Add) and len(term.terms) == 2:
        left, right = term.terms
        if isinstance(right, A.Scale) and float(right.coeff) == -1.0:
            left_sig = _term_signature(left, binder, binding, nm)
            right_sig = _term_signature(right.term, binder, binding, nm)
            if left_sig is not None and right_sig is not None:
                return ("difference", left_sig, right_sig)
    if isinstance(term, A.RelatedAgg):
        return ("related", term.role)
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


def rule_relations(rule: A.Rule, ds, parameters=None) -> set:
    nm = ds.name_model
    rels = set()
    for b in enumerate_bindings(rule.binder, nm):
        sig = _operand_sig(rule, rule.binder, b, nm, parameters)
        if sig is not None:
            rels.add(sig)
        rels |= _additional_operand_sigs(rule, rule.binder, b, nm)
    return rels


def _additional_operand_sigs(rule, binder, binding, nm):
    if not isinstance(rule.atom, A.Compare):
        return set()
    if rule.atom.op not in ("~=", "=="):
        return set()
    left = _ground_term_columns(rule.atom.left, binder, binding, nm)
    right = _ground_term_columns(rule.atom.right, binder, binding, nm)
    if not left or not right:
        return set()
    if not isinstance(
        rule.atom.left,
        (A.Add, A.Agg),
    ) and not isinstance(rule.atom.right, (A.Add, A.Agg)):
        return set()
    return {("sum_balance", frozenset({left, right}))}


def _ground_term_columns(term, binder, binding, nm):
    if isinstance(term, A.Ref):
        column = resolve_ref(term.role, binder, binding, nm)
        return frozenset({column}) if column is not None else frozenset()
    if isinstance(term, A.Agg) and term.kind == "SUM":
        return frozenset(resolve_family(term.family_role, binder, binding, nm))
    if isinstance(term, A.Add):
        columns = frozenset()
        for child in term.terms:
            child_columns = _ground_term_columns(child, binder, binding, nm)
            if not child_columns:
                return frozenset()
            columns |= child_columns
        return columns
    return frozenset()


def portfolio_relations(result: DiscoveryResult) -> set:
    rels: set = set()
    for ev in result.portfolio:
        rels |= rule_relations(
            ev.rule,
            result.dataset,
            getattr(ev, "parameters", {}),
        )
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
    ratio: float = 0.0
    proportional: float = 0.0
    monotone: float = 0.0
    windowed_ratio: float = 0.0
    conditional_positive: float = 0.0
    conditional_zero: float = 0.0
    cross_grain: float = 0.0
    sustained: float = 0.0
    conjunction: float = 0.0
    categorical: float = 0.0
    healthy_band: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _relation_payloads(
    rels: set,
    tag_name: str,
    strengths=("exact", "approximate"),
) -> set:
    payloads = set()
    for relation in rels:
        strength, base = _unwrap_equality_relation(relation)
        if strength is not None and strength not in strengths:
            continue
        if (
            isinstance(base, tuple)
            and len(base) == 2
            and base[0] == tag_name
        ):
            payloads.add(base[1])
    return payloads


def _pairs(rels: set) -> set:
    return _relation_payloads(rels, "pair")


def _refsums(rels: set) -> set:
    return _relation_payloads(rels, "ref_sum")


def _zeros(rels: set) -> set:
    return _relation_payloads(rels, "zero")


def _agg_ref_balances(rels: set) -> set:
    return _relation_payloads(rels, "agg_ref_balance")


def _presence_pairs(rels: set) -> set:
    return _relation_payloads(rels, "presence_pair")


def _ratios(rels: set) -> set:
    return _relation_payloads(rels, "ratio")


def _proportionals(rels: set) -> set:
    return _relation_payloads(rels, "proportional")


def _delta_bounds(rels: set) -> set:
    return _relation_payloads(rels, "delta_bound")


def _windowed_ratios(rels: set) -> set:
    return _relation_payloads(rels, "windowed_ratio")


def _conditionals(
    rels: set,
    base_tag: str,
    strengths=("exact", "approximate"),
) -> set:
    found = set()
    for relation in rels:
        if not (
            isinstance(relation, tuple)
            and len(relation) == 2
            and relation[0] == "conditional"
            and isinstance(relation[1], tuple)
            and len(relation[1]) == 2
        ):
            continue
        condition, nested = relation[1]
        strength, base = _unwrap_equality_relation(nested)
        if strength is not None and strength not in strengths:
            continue
        if isinstance(base, tuple) and base and base[0] == base_tag:
            found.add((condition, base))
    return found


def _payloads(rels: set, tag_name: str) -> set:
    payloads = _relation_payloads(rels, tag_name)
    if tag_name in {"sustained_definition", "conjunction_definition"}:
        return {_definition_shape(payload) for payload in payloads}
    return payloads


def _definition_shape(value):
    if isinstance(value, tuple):
        if value and value[0] == "bound" and len(value) == 4:
            return (value[0], _definition_shape(value[1]), value[2], "threshold")
        if value and all(
            isinstance(item, tuple)
            and item
            and item[0] == "bound"
            for item in value
        ):
            return tuple(sorted(
                (_definition_shape(item) for item in value),
                key=str,
            ))
        return tuple(_definition_shape(item) for item in value)
    return value


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
        if ev.rule.condition is not None:
            continue
        atom = ev.rule.atom
        if not isinstance(atom, A.Compare):
            continue
        accepted_ops = {
            ">=": {">=", ">"},
            "<=": {"<=", "<"},
        }.get(op, {op})
        if atom.op not in accepted_ops:
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


def relation_signature_matches(expected, actual) -> bool:
    # Booleans and strings are exact categorical identities, never fitted thresholds: a Boolean
    # must match only a Boolean of the same value (so ``True`` never spuriously matches ``1``), and
    # a string only an equal string. Numeric tolerance is reserved for genuine numeric quantities
    # (thresholds/centers), where a YAML-declared integer bound must still match a fitted float.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return (
            isinstance(expected, bool)
            and isinstance(actual, bool)
            and expected == actual
        )
    if isinstance(expected, str) or isinstance(actual, str):
        return isinstance(expected, str) and isinstance(actual, str) and expected == actual
    if (
        isinstance(expected, numbers.Integral)
        and isinstance(actual, numbers.Integral)
    ):
        return int(expected) == int(actual)
    if (
        isinstance(expected, numbers.Real)
        and isinstance(actual, numbers.Real)
    ):
        return math.isclose(
            float(expected),
            float(actual),
            rel_tol=0.01,
            abs_tol=1e-9,
        )
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        return (
            len(expected) == len(actual)
            and all(
                relation_signature_matches(left, right)
                for left, right in zip(expected, actual)
            )
        )
    return expected == actual


def score_recovery(result: DiscoveryResult, planted: dict, frac: float = 0.8) -> Recovery:
    rels = portfolio_relations(result)
    pairs, refsums, zeros = _pairs(rels), _refsums(rels), _zeros(rels)
    exact_pairs = _relation_payloads(rels, "pair", strengths=("exact",))
    exact_zeros = _relation_payloads(rels, "zero", strengths=("exact",))
    balances, presence = _agg_ref_balances(rels), _presence_pairs(rels)
    ratios, proportionals = _ratios(rels), _proportionals(rels)
    delta_bounds, windowed_ratios = _delta_bounds(rels), _windowed_ratios(rels)
    conditional_positive_found = _conditionals(rels, "delta_bound")
    conditional_zero_found = _conditionals(
        rels,
        "delta_zero",
        strengths=("exact",),
    )
    related_found = _relation_payloads(
        rels,
        "related_aggregate",
        strengths=("exact",),
    )
    sustained_found = _payloads(rels, "sustained_definition")
    conjunction_found = _payloads(rels, "conjunction_definition")
    categorical_found = _payloads(rels, "categorical_definition")
    healthy_band_found = _payloads(rels, "healthy_band")
    conditional_band_found = _conditionals(rels, "healthy_band")

    def cov(found, target):
        target = set(target)
        if not target:
            return 0.0
        recovered = sum(
            any(
                relation_signature_matches(expected, actual)
                for actual in found
            )
            for expected in target
        )
        return recovered / len(target)

    two = cov(exact_pairs, planted.get("two_end", set()))
    off = cov(pairs, planted.get("offset_pair", set()))
    row = cov(refsums, set(planted.get("row_sum", [])))
    col = cov(refsums, set(planted.get("col_sum", [])))
    sz = cov(exact_zeros, set(planted.get("self_zero", [])))
    bal = cov(balances, set(planted.get("agg_ref_balance", [])))
    pres = cov(presence, set(planted.get("presence_pair", set())))
    # explicit one-sided families: coverage of the planted nonneg / nonpos columns by >=0 / <=0 rules
    nonneg_target = set(planted.get("nonneg", set()))
    nonpos_target = set(planted.get("nonpos", set()))
    nn = cov(_portfolio_one_sided_columns(result, ">="), nonneg_target) if nonneg_target else 0.0
    npos = cov(_portfolio_one_sided_columns(result, "<="), nonpos_target) if nonpos_target else 0.0
    ratio = cov(
        _relation_payloads(rels, "ratio", strengths=("exact",)),
        set(planted.get("ratio", set())),
    )
    proportional = cov(proportionals, set(planted.get("proportional", set())))
    monotone = cov(delta_bounds, set(planted.get("monotone", set())))
    windowed_ratio = cov(
        _relation_payloads(
            rels,
            "windowed_ratio",
            strengths=("exact",),
        ),
        set(planted.get("windowed_ratio", set())),
    )
    conditional_positive = cov(
        conditional_positive_found,
        set(planted.get("conditional_positive", set())),
    )
    conditional_zero = cov(
        conditional_zero_found,
        set(planted.get("conditional_zero", set())),
    )
    cross_grain = cov(related_found, set(planted.get("cross_grain", set())))
    sustained = cov(
        sustained_found,
        {_definition_shape(value) for value in planted.get("sustained", set())},
    )
    conjunction = cov(
        conjunction_found,
        {_definition_shape(value) for value in planted.get("conjunction", set())},
    )
    categorical = cov(categorical_found, set(planted.get("categorical", set())))
    healthy_band = cov(
        healthy_band_found | conditional_band_found,
        set(planted.get("healthy_band", set())),
    )
    recovered = any(
        x >= frac
        for x in (
            two,
            row,
            col,
            sz,
            off,
            bal,
            pres,
            nn,
            npos,
            ratio,
            proportional,
            monotone,
            windowed_ratio,
            conditional_positive,
            conditional_zero,
            cross_grain,
            sustained,
            conjunction,
            categorical,
            healthy_band,
        )
    )
    return Recovery(
        two,
        row,
        col,
        sz,
        off,
        bal,
        pres,
        nn,
        npos,
        recovered,
        ratio=ratio,
        proportional=proportional,
        monotone=monotone,
        windowed_ratio=windowed_ratio,
        conditional_positive=conditional_positive,
        conditional_zero=conditional_zero,
        cross_grain=cross_grain,
        sustained=sustained,
        conjunction=conjunction,
        categorical=categorical,
        healthy_band=healthy_band,
    )


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
    null_temporal: int = 0
    null_definitions: int = 0

    @property
    def recovery_ok(self) -> bool:
        return all(p.recovery >= _PROXY_RECOVERY_TARGET for p in self.proxies)

    @property
    def compact_ok(self) -> bool:
        return all(p.compact and not p.scaled_slack for p in self.proxies)

    @property
    def null_safe(self) -> bool:
        return (
            self.null_equalities == 0
            and self.null_temporal == 0
            and self.null_definitions == 0
        )

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
            "null_temporal": self.null_temporal,
            "null_definitions": self.null_definitions,
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
    proposer: object = None
    search_cfg: Optional[SearchConfig] = None


@dataclass
class ProxySuite:
    """The selected positive proxies plus the always-on null control, each prepared once."""
    positives: List[PreparedProxy]
    null: PreparedProxy
    temporal_null: Optional[PreparedProxy] = None
    definition_null: Optional[PreparedProxy] = None
    candidate_counts: dict[str, int] = field(default_factory=dict)

    def shapes(self) -> List[str]:
        return [p.shape for p in self.positives]


class _PreparedRuleProposer(EnumerationProposer):
    def __init__(self, grammar, rules):
        super().__init__(grammar)
        self._cache = list(rules)


def _balanced_null_numeric(
    values,
    rng,
    *,
    binary: bool,
) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    output = np.full(source.shape, np.nan, dtype=float)
    positions = np.flatnonzero(np.isfinite(source))
    if not positions.size:
        return output
    if binary:
        generated = np.arange(positions.size, dtype=float) % 2.0
        rng.shuffle(generated)
    else:
        magnitudes = np.abs(source[positions])
        positive = magnitudes[magnitudes > 0.0]
        # ``np.median`` averages the two central values, and that intermediate sum overflows near
        # the float64 ceiling.  An infinite scale would fill the null column with infinities, every
        # null candidate would then be refused for overflowing rather than on its merits, and the
        # false-discovery control would silently become vacuous.
        scale = robust_median(positive) if positive.size else 1.0
        multipliers = rng.lognormal(
            mean=0.0,
            sigma=0.5,
            size=positions.size,
        )
        # A finite scale is not enough: the lognormal multiplier is unbounded above, so a
        # ceiling-scale column still generates infinities. Those become missing data, the null
        # candidates built on them are refused for overflowing rather than judged on their merits,
        # and the false-discovery control silently goes vacuous. Shrink the scale just enough that
        # the largest drawn multiplier stays finite; the null's SHAPE (a balanced, sign-symmetric
        # lognormal spread) is what the control depends on, not its absolute magnitude.
        largest = float(np.max(multipliers)) if multipliers.size else 1.0
        if largest > 1.0 and scale > np.finfo(float).max / largest:
            scale = np.finfo(float).max / largest
        with np.errstate(over="ignore"):
            magnitudes = scale * multipliers
            # The division above can still round up, so halve until the product is finite. This
            # terminates immediately for every ordinary column and after a couple of steps at the
            # float64 ceiling.
            while not np.all(np.isfinite(magnitudes)):
                scale *= 0.5
                magnitudes = scale * multipliers
        signs = np.ones(positions.size, dtype=float)
        signs[:positions.size // 2] = -1.0
        if positions.size % 2:
            signs[-1] = 0.0
        rng.shuffle(signs)
        generated = magnitudes * signs
    output[positions] = generated
    return output


def _runtime_relation_null(
    relation,
    templates,
    parent_context,
    rng,
    *,
    definition_targets: bool,
):
    if not isinstance(relation, pd.DataFrame):
        return relation
    span_templates = [
        template for template in templates
        if template.mode == "span_any"
    ]
    if span_templates:
        records = []
        emitted = set()
        for template_index, template in enumerate(span_templates):
            required = {
                template.parent_time,
                *template.parent_keys,
            } - {""}
            if not required <= set(parent_context):
                continue
            times = np.asarray(
                parent_context[template.parent_time]
            )
            key_arrays = [
                np.asarray(parent_context[key], dtype=object)
                for key in template.parent_keys
            ]
            groups = {}
            for row in range(times.size):
                key = tuple(array[row] for array in key_arrays)
                groups.setdefault(key, []).append(row)
            filter_value = (
                template.filter_values[0]
                if template.filter_values
                else "__null__"
            )
            for key, rows in groups.items():
                ordered = sorted(rows, key=lambda row: times[row])
                for position, row in enumerate(ordered):
                    if not ((position >> template_index) & 1):
                        continue
                    signature = (
                        tuple(key),
                        template.filter_column,
                        filter_value,
                        template.span_start,
                        template.span_end,
                        times[row],
                    )
                    if signature in emitted:
                        continue
                    emitted.add(signature)
                    record = {
                        column: pd.NA
                        for column in relation.columns
                    }
                    for child_key, value in zip(
                        template.child_keys,
                        key,
                    ):
                        record[child_key] = value
                    record[template.span_start] = times[row]
                    if position + 1 < len(ordered):
                        span_end = times[ordered[position + 1]]
                    elif len(ordered) > 1:
                        span_end = (
                            times[row]
                            + (
                                times[ordered[-1]]
                                - times[ordered[-2]]
                            )
                        )
                    else:
                        span_end = times[row]
                    record[template.span_end] = span_end
                    if template.filter_column:
                        record[template.filter_column] = filter_value
                    records.append(record)
        if records:
            return pd.DataFrame.from_records(
                records,
                columns=relation.columns,
            )
    output = relation.copy(deep=True)
    structural = set()
    binary_columns = set()
    for template in templates:
        structural.update(template.child_keys)
        structural.update(template.partition_keys)
        structural.update({
            template.child_time,
            template.span_start,
            template.span_end,
        })
        if template.reset_column:
            binary_columns.add(template.reset_column)
    structural.discard("")
    for column in output.columns:
        if column in structural:
            continue
        values = output[column].to_numpy(copy=True)
        if (
            pd.api.types.is_numeric_dtype(output[column])
            or pd.api.types.is_bool_dtype(output[column])
        ):
            unique = set(
                np.asarray(values, dtype=float)[
                    np.isfinite(np.asarray(values, dtype=float))
                ].tolist()
            )
            binary = (
                pd.api.types.is_bool_dtype(output[column])
                or column in binary_columns
                or (
                    definition_targets
                    and bool(unique)
                    and not (unique - {0.0, 1.0})
                )
            )
            output[column] = _balanced_null_numeric(
                values,
                rng,
                binary=binary,
            )
        else:
            output[column] = rng.permutation(values)
    return output


def _runtime_condition_context(
    adapter,
    n_rows: int,
    rng,
    *,
    randomize: bool,
) -> dict[str, np.ndarray]:
    domains = {
        name: tuple(values)
        for name, values in getattr(
            adapter,
            "condition_columns",
            {},
        ).items()
        if values
    }
    ordered = sorted(
        domains,
        key=lambda name: (
            0
            if set(domains[name]) - {False, True, 0, 1}
            else 1,
            name,
        ),
    )
    context = {}
    stride = 1
    boolean_index = 0
    rows = np.arange(n_rows, dtype=np.int64)
    for name in ordered:
        values = np.asarray(domains[name], dtype=object)
        categorical = bool(
            set(domains[name]) - {False, True, 0, 1}
        )
        if categorical:
            indexes = (rows // stride) % len(values)
            stride *= len(values)
        else:
            indexes = (rows + boolean_index) % len(values)
            boolean_index += 1
        generated = values[indexes]
        if randomize:
            generated = rng.permutation(generated)
        context[name] = generated
    return context


def _runtime_null_dataset(
    dataset,
    *,
    seed: int,
    definition_targets: bool,
):
    rng = np.random.default_rng(int(seed))
    matrix = np.empty_like(dataset.observed.matrix, dtype=float)
    generated_columns = {}
    adapter = dataset.name_model.adapter
    boolean_columns = set()
    for binder in adapter.binders:
        for binding in enumerate_bindings(
            binder,
            dataset.name_model,
        ):
            for role in adapter.booleans_for(binder):
                column = resolve_ref(
                    role,
                    binder,
                    binding,
                    dataset.name_model,
                )
                if column is not None:
                    boolean_columns.add(column)
    for index, name in enumerate(dataset.observed.names):
        values = dataset.observed.matrix[:, index]
        finite = values[np.isfinite(values)]
        unique = set(finite.tolist())
        binary = (
            name in boolean_columns
            or (
                definition_targets
                and bool(unique)
                and not (unique - {0.0, 1.0})
            )
        )
        generated = _balanced_null_numeric(
            values,
            rng,
            binary=binary,
        )
        matrix[:, index] = generated
        generated_columns[name] = generated

    condition_context = _runtime_condition_context(
        adapter,
        dataset.observed.n_rows,
        rng,
        randomize=definition_targets,
    )
    templates = tuple(
        getattr(adapter, "related_templates", {}).values()
    )
    preserved_context = {
        dataset.time_index,
        *dataset.group_keys,
        *(
            key
            for template in templates
            for key in (
                *template.parent_keys,
                template.parent_time,
            )
        ),
    } - {""}
    row_context = {}
    for name, values in dataset.observed.row_context.items():
        if name in condition_context:
            row_context[name] = condition_context[name].copy()
        elif name in preserved_context:
            row_context[name] = np.asarray(values).copy()
        elif name in generated_columns:
            row_context[name] = generated_columns[name].copy()
        else:
            row_context[name] = rng.permutation(
                np.asarray(values)
            )

    relation_templates = {}
    for template in templates:
        relation_templates.setdefault(template.relation, []).append(
            template
        )
    relations = {
        name: _runtime_relation_null(
            relation,
            relation_templates.get(name, ()),
            row_context,
            rng,
            definition_targets=definition_targets,
        )
        for name, relation in dataset.observed.relations.items()
    }
    observed = Frame(
        matrix,
        dataset.observed.names,
        row_context=row_context,
        relations=relations,
    )
    return Dataset(
        name=f"{dataset.name}_runtime_null_{seed}",
        name_model=dataset.name_model,
        observed=observed,
        timestamps=np.asarray(dataset.timestamps).copy(),
        n_snapshots=dataset.n_snapshots,
        time_index=dataset.time_index,
        group_keys=tuple(dataset.group_keys),
        row_context=row_context,
        relations=relations,
    )


def _is_null_equality_candidate(rule: A.Rule) -> bool:
    return (
        isinstance(rule.atom, A.Compare)
        and rule.atom.op in {
            "~=",
            "==",
            "~\u221d",
            ">=",
            "<=",
            ">",
            "<",
        }
    ) or isinstance(rule.atom, A.BandDefinition)


def prepare_runtime_null_controls(
    dataset,
    grammar,
    search_cfg: SearchConfig,
    *,
    seed: int = 0,
    rules=None,
    proposer=None,
) -> ProxySuite:
    """Build independently randomized nulls over the exact runtime grammar."""

    all_rules = list(
        rules
        if rules is not None
        else (
            proposer
            if proposer is not None
            else EnumerationProposer(grammar)
        ).propose()
    )
    equality_rules = [
        rule for rule in all_rules
        if _is_null_equality_candidate(rule)
    ]
    temporal_rules = [
        rule for rule in all_rules
        if _rule_has_temporal(rule)
    ]
    definition_rules = [
        rule for rule in all_rules
        if isinstance(
            rule.atom,
            (A.BooleanDefinition, A.CategoryDefinition),
        )
    ]

    equality_dataset = _runtime_null_dataset(
        dataset,
        seed=seed + 10_001,
        definition_targets=False,
    )
    temporal_dataset = (
        _runtime_null_dataset(
            dataset,
            seed=seed + 20_003,
            definition_targets=False,
        )
        if temporal_rules
        else None
    )
    definition_dataset = (
        _runtime_null_dataset(
            dataset,
            seed=seed + 30_007,
            definition_targets=True,
        )
        if definition_rules
        else None
    )
    return ProxySuite(
        positives=[],
        null=PreparedProxy(
            "runtime_null",
            equality_dataset,
            grammar,
            {},
            _PreparedRuleProposer(grammar, equality_rules),
            search_cfg,
        ),
        temporal_null=(
            PreparedProxy(
                "runtime_temporal_null",
                temporal_dataset,
                grammar,
                {},
                _PreparedRuleProposer(grammar, temporal_rules),
                search_cfg,
            )
            if temporal_dataset is not None
            else None
        ),
        definition_null=(
            PreparedProxy(
                "runtime_definition_null",
                definition_dataset,
                grammar,
                {},
                _PreparedRuleProposer(grammar, definition_rules),
                search_cfg,
            )
            if definition_dataset is not None
            else None
        ),
        candidate_counts={
            "all": len(all_rules),
            "equalities": len(equality_rules),
            "temporal": len(temporal_rules),
            "definitions": len(definition_rules),
        },
    )


class _ColumnSchemaCache(SchemaInducer):
    """Reuse one induced spec for proxy datasets with the same generated column schema."""

    def __init__(self, delegate: SchemaInducer):
        self.delegate = delegate
        self.backend = delegate.backend
        self._specs = {}

    def induce(self, columns, sample_rows=None):
        key = tuple(str(column) for column in columns)
        if key not in self._specs:
            self._specs[key] = self.delegate.induce(columns, sample_rows)
        return self._specs[key]


def prepare_proxy_suite(regime, seed: int = 0,
                        inducer: Optional[SchemaInducer] = None,
                        null_shapes=None,
                        null_windows=(),
                        null_max_conjunction_terms: int = 3,
                        null_search_cfg: Optional[SearchConfig] = None) -> ProxySuite:
    """Generate + induce every active positive proxy and the null control exactly once.

    The prepared (dataset, grammar) pairs are reused for every joint-tuning grid cell and for every
    relaxation-ladder rung's null gate, so schema induction happens once per proxy.  Proxy values
    are synthetic and only ever feed the *proxy* grammars; real-data grammar induction is untouched
    (it still receives only the dataset's own column names).
    """
    from .regime import KNOWN_SHAPES, generate as _gen, generate_null as _gen_null
    entries = regime.active_entries()
    # Validate every active shape up front: a custom regime built directly (bypassing
    # RegimeSpec.add) can carry an unknown shape, which would otherwise fail obscurely later at
    # make_synthetic (empty planted set) or getattr(rec, shape).  Fail loudly *before* generation.
    unknown = sorted({e.shape for e in entries if e.shape not in KNOWN_SHAPES})
    if unknown:
        raise ValueError(
            f"custom regime has unknown proxy shape(s) {unknown}; choose from {list(KNOWN_SHAPES)}")
    inducer = _ColumnSchemaCache(inducer or make_inducer("subagent"))
    null_search_cfg = null_search_cfg or _small_search(seed)
    positives: List[PreparedProxy] = []
    for e in entries:
        data = _gen(e, seed=seed)
        ds, G, _spec = prepare_columns(data.columns, data.matrix, inducer=inducer,
                                       search_cfg=_small_search(seed, e.shape),
                                       name=f"proxy_{e.shape}", timestamps=data.timestamps)
        _attach_proxy_context(ds, data)
        _enable_shape_capabilities(
            G,
            e.shape,
            ds,
            temporal_window=e.temporal_window,
        )
        positives.append(PreparedProxy(
            e.shape,
            ds,
            G,
            data.planted,
            EnumerationProposer(G),
        ))
    active_shapes = {proxy.shape for proxy in positives}
    control_shapes = set(
        KNOWN_SHAPES if null_shapes is None else null_shapes
    )
    control_windows = tuple(sorted({
        *(int(entry.temporal_window) for entry in entries),
        *(int(window) for window in null_windows),
        5,
    }))
    ndata = S.enrich_null(
        _gen_null(seed=seed),
        seed=seed,
        conditions="healthy_band" in control_shapes,
        related="cross_grain" in control_shapes,
    )
    nds, nG, _nspec = prepare_columns(ndata.columns, ndata.matrix, inducer=inducer,
                                      search_cfg=null_search_cfg, name="null_proxy",
                                      timestamps=ndata.timestamps)
    _attach_proxy_context(nds, ndata)
    for shape in sorted(
        control_shapes & {"ratio", "proportional", "healthy_band", "cross_grain"}
    ):
        _enable_shape_capabilities(nG, shape, nds)
    temporal_null = None
    temporal_shapes = {
        "monotone",
        "windowed_ratio",
        "conditional_positive",
        "conditional_zero",
        "sustained",
        "conjunction",
    }
    if control_shapes & temporal_shapes:
        tdata = S.enrich_null(
            S.make_temporal_null(seed=seed),
            seed=seed + 1,
            conditions=bool(
                control_shapes & {"conditional_positive", "conditional_zero"}
            ),
        )
        tds, tG, _tspec = prepare_columns(
            tdata.columns,
            tdata.matrix,
            inducer=inducer,
            search_cfg=null_search_cfg,
            name="temporal_null_proxy",
            timestamps=tdata.timestamps,
        )
        _attach_proxy_context(tds, tdata)
        for temporal_window in control_windows:
            _enable_shape_capabilities(
                tG,
                "monotone",
                tds,
                temporal_window=temporal_window,
            )
            for shape in sorted(
                control_shapes & {
                    "windowed_ratio",
                    "conditional_positive",
                    "conditional_zero",
                }
            ):
                _enable_shape_capabilities(
                    tG,
                    shape,
                    tds,
                    temporal_window=temporal_window,
                )
        temporal_null = PreparedProxy(
            "temporal_null",
            tds,
            tG,
            {},
            EnumerationProposer(tG),
            null_search_cfg,
        )
    definition_null = None
    advanced_shapes = {"sustained", "conjunction", "categorical"}
    if control_shapes & advanced_shapes:
        ddata = S.make_definition_null(seed=seed)
        dds, dG, _dspec = prepare_columns(
            ddata.columns,
            ddata.matrix,
            inducer=inducer,
            search_cfg=null_search_cfg,
            name="definition_null_proxy",
            timestamps=ddata.timestamps,
        )
        _attach_proxy_context(dds, ddata)
        for temporal_window in control_windows:
            _enable_shape_capabilities(
                dG,
                "definition_null",
                dds,
                temporal_window=temporal_window,
                max_conjunction_terms=null_max_conjunction_terms,
            )
        definition_null = PreparedProxy(
            "definition_null",
            dds,
            dG,
            {},
            EnumerationProposer(dG),
            null_search_cfg,
        )
    return ProxySuite(
        positives=positives,
        null=PreparedProxy(
            "null",
            nds,
            nG,
            {},
            EnumerationProposer(nG),
            null_search_cfg,
        ),
        temporal_null=temporal_null,
        definition_null=definition_null,
    )


def null_equalities_at(prepared_null: PreparedProxy, dcfg: DiscoveryConfig, seed: int = 0) -> int:
    """Count algebraic, one-sided, or fitted-band laws accepted on the random null."""
    res = run_prepared(prepared_null.ds, prepared_null.G, discovery_cfg=dcfg,
                       search_cfg=prepared_null.search_cfg or _small_search(seed),
                       proposer=prepared_null.proposer)
    accepted = len([
        evaluation
        for evaluation in res.portfolio
        if (
            isinstance(evaluation.rule.atom, A.Compare)
            and evaluation.rule.atom.op in (
                "~=",
                "==",
                "~∝",
                ">=",
                "<=",
                ">",
                "<",
            )
        ) or isinstance(evaluation.rule.atom, A.BandDefinition)
    ])
    return accepted


def null_temporal_at(prepared_null: PreparedProxy | None, dcfg: DiscoveryConfig,
                     seed: int = 0) -> int:
    if prepared_null is None:
        return 0
    res = run_prepared(
        prepared_null.ds,
        prepared_null.G,
        discovery_cfg=dcfg,
        search_cfg=prepared_null.search_cfg or _small_search(seed, "monotone"),
        proposer=prepared_null.proposer,
    )
    accepted = sum(_rule_has_temporal(e.rule) for e in res.portfolio)
    return accepted


def null_definitions_at(prepared_null: PreparedProxy | None, dcfg: DiscoveryConfig,
                        seed: int = 0) -> int:
    if prepared_null is None:
        return 0
    res = run_prepared(
        prepared_null.ds,
        prepared_null.G,
        discovery_cfg=dcfg,
        search_cfg=prepared_null.search_cfg or _small_search(seed, "definition_null"),
        proposer=prepared_null.proposer,
    )
    accepted = sum(
        isinstance(evaluation.rule.atom, (A.BooleanDefinition, A.CategoryDefinition))
        for evaluation in res.portfolio
    )
    return accepted


def _release_proxy_caches(prepared: PreparedProxy | None) -> None:
    if prepared is None:
        return
    frame = getattr(getattr(prepared, "ds", None), "observed", None)
    if frame is None:
        return
    frame.term_cache.clear()
    frame.related_cache.clear()
    frame.related_index_cache.clear()


def evaluate_grid_candidate(suite: ProxySuite, tolerance: float, hold_rate_threshold: float,
                            seed: int = 0, band_mode: str = "global",
                            ci_alpha: float = 0.05) -> GridCandidate:
    """Score one (tolerance, threshold) on every prepared positive proxy plus the null control."""
    proxies: List[ProxyOutcome] = []
    for p in suite.positives:
        dcfg = DiscoveryConfig(seed=seed, tolerance=tolerance,
                               hold_rate_threshold=hold_rate_threshold,
                               band_mode=band_mode, ci_alpha=ci_alpha)
        res = run_prepared(
            p.ds,
            p.G,
            discovery_cfg=dcfg,
            search_cfg=_small_search(seed, p.shape),
            proposer=p.proposer,
        )
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
                            hold_rate_threshold=hold_rate_threshold,
                            band_mode=band_mode, ci_alpha=ci_alpha)
    return GridCandidate(
        tolerance,
        hold_rate_threshold,
        proxies,
        null_equalities_at(suite.null, ndcfg, seed),
        null_temporal_at(getattr(suite, "temporal_null", None), ndcfg, seed),
        null_definitions_at(getattr(suite, "definition_null", None), ndcfg, seed),
    )


_BASE_THRESHOLDS = (0.58, 0.62, 0.66, 0.72)
_BASE_TOLERANCES = (0.005, 0.01, 0.02, 0.05)


def tune_joint(suite, seed: int = 0, band_mode: str = "global", ci_alpha: float = 0.05,
               null_floor: float = 0.5,
               max_expansions: int = 3, thresholds=None, tolerances=None, evaluate=None,
               initial_tolerance: float | None = None,
               initial_threshold: float | None = None) -> dict:
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
    if initial_threshold is not None:
        thresholds = sorted({
            *thresholds,
            float(initial_threshold),
        })
    if initial_tolerance is not None:
        tolerances = sorted({
            *tolerances,
            float(initial_tolerance),
        })
    expansions = 0
    candidates: List[GridCandidate] = []
    best: Optional[GridCandidate] = None
    # Memoize by (tolerance, threshold): each expansion re-lists earlier cells, so caching keeps
    # every unique grid cell evaluated exactly once across the growing grids.
    cache: dict = {}

    def _cell(tol, thr):
        key = (tol, thr)
        if key not in cache:
            cache[key] = evaluate(
                suite,
                tol,
                thr,
                seed=seed,
                band_mode=band_mode,
                ci_alpha=ci_alpha,
            )
        return cache[key]

    if (
        initial_tolerance is not None
        and initial_threshold is not None
    ):
        starting = _cell(
            float(initial_tolerance),
            float(initial_threshold),
        )
        candidates = [starting]
        if starting.eligible:
            best = starting

    while best is None:
        candidates = []
        for tol, thr in sorted(
            ((tol, thr) for tol in tolerances for thr in thresholds),
            key=lambda cell: (cell[0], -cell[1]),
        ):
            candidate = _cell(tol, thr)
            candidates.append(candidate)
            if candidate.eligible:
                best = candidate
                break
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
        "selected_null_temporal": best.null_temporal,
        "selected_null_definitions": best.null_definitions,
        "proxy_shapes": [p.shape for p in best.proxies],
        "per_proxy": [{"shape": p.shape, "recovery": round(p.recovery, 4), "accepted": p.accepted,
                       "compact": p.compact, "scaled_slack": p.scaled_slack}
                      for p in best.proxies],
    }


def _rule_has_temporal(rule: A.Rule) -> bool:
    def term_has(term) -> bool:
        if isinstance(term, (A.Lag, A.Diff, A.Rolling)):
            return True
        if isinstance(term, A.Scale):
            return term_has(term.term)
        if isinstance(term, A.Add):
            return any(term_has(item) for item in term.terms)
        if isinstance(term, A.Mul):
            return term_has(term.left) or term_has(term.right)
        if isinstance(term, A.Div):
            return term_has(term.num) or term_has(term.den)
        return False

    def predicate_has(predicate) -> bool:
        if isinstance(predicate, A.Bound):
            return term_has(predicate.term)
        if isinstance(predicate, A.Sustained):
            return True
        if isinstance(predicate, A.Conjunction):
            return any(predicate_has(item) for item in predicate.predicates)
        return False

    if isinstance(rule.atom, A.Compare):
        return term_has(rule.atom.left) or term_has(rule.atom.right)
    if isinstance(rule.atom, A.BooleanDefinition):
        return predicate_has(rule.atom.predicate)
    return False


@dataclass
class PlantRecover:
    noise_levels: List[float]
    recovered: Dict[str, Dict[float, bool]] = field(default_factory=dict)
    detail: Dict[str, Dict[float, dict]] = field(default_factory=dict)


_PLANT_FAMILIES = KNOWN_SHAPES

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


def _attach_proxy_context(dataset, data) -> None:
    dataset.observed.term_cache.max_entries = 1_024
    dataset.observed.term_cache.max_bytes = 8 * 1024 * 1024
    for name, values in getattr(data, "row_context", {}).items():
        array = np.asarray(values)
        dataset.row_context[name] = array
        dataset.observed.row_context[name] = array
    dataset.relations.update(getattr(data, "relations", {}))
    dataset.observed.relations.update(getattr(data, "relations", {}))
    dataset._proxy_related_aggregates = dict(
        getattr(data, "related_aggregates", {})
    )
    dataset._proxy_planted = dict(
        getattr(data, "planted", {})
    )


def _proxy_context_values(dataset, max_values: int) -> dict:
    values = {}
    for name, array in dataset.row_context.items():
        unique = tuple(pd.unique(np.asarray(array)).tolist())
        if 1 < len(unique) <= int(max_values):
            values[name] = unique
    return values


def _proxy_boolean_roles(grammar, dataset) -> dict:
    planted_targets = {
        item[0]
        for shape in ("sustained", "conjunction")
        for item in getattr(
            dataset,
            "_proxy_planted",
            {},
        ).get(shape, ())
    }
    by_binder = {}
    for binder, roles in grammar.ref_roles.items():
        selected = []
        bindings = enumerate_bindings(
            binder,
            dataset.name_model,
        )
        for role in roles:
            columns = {
                resolve_ref(
                    role,
                    binder,
                    binding,
                    dataset.name_model,
                )
                for binding in bindings
            } - {None}
            binary = any(
                column in dataset.observed.name_to_idx
                and set(
                    np.unique(
                        dataset.observed.col(column)[
                            np.isfinite(dataset.observed.col(column))
                        ]
                    ).tolist()
                ) <= {0.0, 1.0}
                for column in columns
            )
            if columns & planted_targets or binary:
                selected.append(role)
        if selected:
            by_binder[binder] = tuple(selected)
    return by_binder


def _enable_shape_capabilities(
    grammar,
    family: str,
    dataset=None,
    temporal_window: int = 5,
    max_conjunction_terms: int = 3,
) -> None:
    temporal_window = max(1, int(temporal_window))
    if family in {"ratio", "windowed_ratio"}:
        grammar.max_degree = max(2, grammar.max_degree)
        grammar.max_degree_by_binder = {
            binder: max(2, degree)
            for binder, degree in grammar.max_degree_by_binder.items()
        }
    if family == "proportional" and "~∝" not in grammar.ops:
        grammar.ops = tuple(grammar.ops) + ("~∝",)
    if family in {
        "monotone",
        "windowed_ratio",
        "conditional_positive",
        "conditional_zero",
        "sustained",
        "conjunction",
        "definition_null",
    }:
        grammar.temporal_enabled = True
        grammar.max_lag = max(temporal_window, grammar.max_lag)
        grammar.windows = tuple(sorted({
            *grammar.windows,
            temporal_window,
        }))
        grammar.max_complexity_by_binder = {
            binder: max(10, cap)
            for binder, cap in grammar.max_complexity_by_binder.items()
        }
        if dataset is not None:
            adapter = dataset.name_model.adapter
            adapter.temporal_enabled = True
            adapter.time_index = "__time__"
            adapter.group_keys = ()
            adapter.max_lag = max(temporal_window, adapter.max_lag)
            adapter.windows = tuple(sorted({
                *adapter.windows,
                temporal_window,
            }))
            dataset.time_index = "__time__"
            dataset.observed.row_context["__time__"] = np.asarray(dataset.timestamps)
            dataset.row_context["__time__"] = np.asarray(dataset.timestamps)
    if family in {"conditional_positive", "conditional_zero"}:
        grammar.conditional_enabled = True
        grammar.condition_columns = {
            "regime": ("positive", "zero", "other"),
        }
        if dataset is not None:
            adapter = dataset.name_model.adapter
            adapter.conditional_enabled = True
            adapter.condition_columns = {
                "regime": ("positive", "zero", "other"),
            }
    if family in {"sustained", "conjunction", "categorical", "definition_null"}:
        grammar.advanced_enabled = True
        grammar.run_lengths = tuple(sorted({
            *grammar.run_lengths,
            temporal_window,
        }))
        grammar.max_conjunction_terms = max(
            int(max_conjunction_terms),
            grammar.max_conjunction_terms,
        )
        if dataset is not None:
            for binder, roles in _proxy_boolean_roles(
                grammar,
                dataset,
            ).items():
                grammar.boolean_roles[binder] = roles
                dataset.name_model.adapter.boolean_roles[binder] = roles
        if dataset is not None:
            adapter = dataset.name_model.adapter
            adapter.advanced_enabled = True
            adapter.run_lengths = tuple(sorted({
                *adapter.run_lengths,
                temporal_window,
            }))
            adapter.max_conjunction_terms = max(
                int(max_conjunction_terms),
                adapter.max_conjunction_terms,
            )
            grammar.max_complexity_by_binder = {
                binder: max(16, cap)
                for binder, cap in grammar.max_complexity_by_binder.items()
            }
    if family in {"categorical", "definition_null"}:
        values = (
            _proxy_context_values(
                dataset,
                grammar.max_condition_values,
            )
            if dataset is not None
            else {}
        )
        grammar.conditional_enabled = True
        grammar.condition_columns = values
        if dataset is not None:
            adapter = dataset.name_model.adapter
            adapter.conditional_enabled = True
            adapter.condition_columns = values
    if family == "healthy_band" and dataset is not None:
        values = _proxy_context_values(
            dataset,
            grammar.max_condition_values,
        )
        grammar.band_enabled = True
        grammar.conditional_enabled = True
        grammar.condition_columns = values
        adapter = dataset.name_model.adapter
        adapter.band_enabled = True
        adapter.conditional_enabled = True
        adapter.condition_columns = values
    if family == "cross_grain" and dataset is not None:
        adapter = dataset.name_model.adapter
        for role, raw in getattr(dataset, "_proxy_related_aggregates", {}).items():
            template = RelatedTemplate(
                binder="node",
                role=str(role),
                relation=str(raw["relation"]),
                column=str(raw["column"]),
                mode=str(raw["mode"]),
                parent_keys=tuple(raw.get("parent_keys", ())),
                child_keys=tuple(raw.get("child_keys", ())),
                partition_keys=tuple(raw.get("partition_keys", ())),
                parent_time=str(raw["parent_time"]),
                child_time=str(raw["child_time"]),
                window_seconds=int(raw["window_seconds"]),
                reset_column=str(raw.get("reset_column", "")),
                validity_columns=tuple(raw.get("validity_columns", ())),
                span_start=str(raw.get("span_start", "")),
                span_end=str(raw.get("span_end", "")),
                filter_column=str(raw.get("filter_column", "")),
                filter_values=tuple(raw.get("filter_values", ())),
            )
            adapter.related_templates[("node", str(role))] = template
            grammar.related_roles["node"] = tuple(sorted({
                *grammar.related_roles.get("node", ()),
                str(role),
            }))


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
            effective_noise = (
                0.0
                if family in {
                    "two_end",
                    "self_zero",
                    "ratio",
                    "windowed_ratio",
                    "cross_grain",
                }
                else nz
            )
            data = S.make_synthetic(n_entities=n_entities, n_snapshots=n_snapshots,
                                    noise=effective_noise, seed=seed, families=(family,))
            ds, grammar, _spec = prepare_columns(
                data.columns,
                data.matrix,
                inducer=inducer or make_inducer("subagent"),
                search_cfg=_small_search(seed, family),
                name=f"proxy_{family}",
                timestamps=data.timestamps,
            )
            _attach_proxy_context(ds, data)
            _enable_shape_capabilities(grammar, family, ds)
            res = run_prepared(
                ds,
                grammar,
                discovery_cfg=_fast_eval(seed, family),
                search_cfg=_small_search(seed, family),
            )
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
    return len([
        evaluation
        for evaluation in res.portfolio
        if isinstance(evaluation.rule.atom, A.Compare)
        and evaluation.rule.atom.op in ("~=", "==", "~∝")
    ])


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
        effective_noise = (
            0.0
            if family in {
                "two_end",
                "self_zero",
                "ratio",
                "windowed_ratio",
                "cross_grain",
            }
            else noise
        )
        data = S.make_synthetic(n_entities=4, n_snapshots=120, noise=effective_noise, seed=seed,
                                families=(family,))
        ds, grammar, _spec = prepare_columns(
            data.columns,
            data.matrix,
            inducer=make_inducer("subagent"),
            search_cfg=search,
            name=f"runtime_{family}",
            timestamps=data.timestamps,
        )
        _attach_proxy_context(ds, data)
        _enable_shape_capabilities(grammar, family, ds)
        res = run_prepared(ds, grammar, discovery_cfg=discovery, search_cfg=search)
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
        if isinstance(atom, A.BooleanDefinition):
            if isinstance(atom.predicate, A.Sustained):
                fams.add("sustained temporal definition")
            elif isinstance(atom.predicate, A.Conjunction):
                fams.add("conjunctive definition")
        elif isinstance(atom, A.CategoryDefinition):
            fams.add("categorical priority definition")
        elif atom.op == "!=":
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
    if not isinstance(rule.atom, A.Compare):
        return False
    return rule.atom.op in ("<=", ">=", "<", ">") and (
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
    from .regime import RegimeSpec

    null_suite = prepare_proxy_suite(
        RegimeSpec(),
        seed=seed,
        null_shapes=_PLANT_FAMILIES,
    )
    null_config = tuned["runtime_discovery"]
    return {
        "proxy_ok": bool(tuned["ok"]),
        "synthetic_recovery": {"ok": bool(tuned["ok"]), "families": tuned["family_ok"]},
        "tuned_threshold_tolerance": tuned["tuned_threshold_tolerance"],
        "runtime_recovery": tuned["runtime_recovery"],
        "null_equalities_accepted": null_equalities_at(
            null_suite.null,
            null_config,
            seed,
        ),
        "null_temporal_accepted": null_temporal_at(
            null_suite.temporal_null,
            null_config,
            seed,
        ),
        "null_definitions_accepted": null_definitions_at(
            null_suite.definition_null,
            null_config,
            seed,
        ),
        "portfolio_quality": pq,
    }
