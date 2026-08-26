# Autogram's guarantees: grammar-conditioned determinism and exhaustiveness, soundness, and monotone completeness

This document states, rigorously and with code citations, three guarantees the current Autogram engine provides. Each guarantee is given three ways: a high-level intuition (what it means and why you should care), a precise mathematical formulation (with every symbol defined), and a short proof grounded in the implementation. The guiding principle is honesty: every guarantee is stated relative to explicit assumptions, and the boundary of each claim is drawn as carefully as the claim itself.

One sentence up front frames all three. **Fix the grammar, and Autogram becomes a deterministic instrument that computes exactly the set of statistically-sound invariants the grammar can express, and any unknown invariant at least as clean and simple as one of your known invariants is guaranteed to be among them.** The three guarantees are the three clauses of that sentence: "deterministic" together with "computes exactly the set the grammar can express" (it generates every expressible candidate and keeps a representative of every accepted one) is Guarantee I; "statistically-sound" is Guarantee II; "any unknown invariant that dominates a known one is among them" is Guarantee III. The first two are completeness *relative to the grammar*; the third is completeness *relative to your known invariants*. Everything is conditioned on the grammar, which is why we call the determinism *grammar-conditioned*: the one non-deterministic step in the whole pipeline is the language-model proposal of the grammar itself, and it is deliberately isolated from everything downstream.

This is the contract behind the guarantees language in `docs/calibration_protocol.md`; that document covers how the dials are tuned, which is out of scope here.

## 0. Notation and the objects in play

The pipeline is a composition of maps, and it helps to see it that way:

```
columns ──Induce──▶ Γ ──Compile──▶ (dataset, grammar) ──Enumerate──▶ candidates ──Score(θ, seed)──▶ Π
          (LLM, stochastic)         (deterministic)      (deterministic)          (deterministic)
```

Only the first arrow, `Induce` (the language model proposing a schema from column names, `discovery/induce.py::induce_spec`), is stochastic. Once its output `Γ` is fixed, everything to the right is a deterministic function of `(Γ, D, θ, seed)`. That separation is real in the code: `discovery/loop.py::prepare_columns` performs induction once and hands a fixed `(dataset, grammar)` to `run_prepared`, which never re-induces.

The symbols used throughout:

| Symbol | Name | Meaning in plain terms |
|--------|------|------------------------|
| `D` | dataset | the observed numeric values together with a *name model* that says which columns each binder grounds to (`loader/`) |
| `b` | binder | a universal quantifier "for all X" ranging over entities of one kind (e.g. every node, every link) |
| `r` | rule | one candidate invariant, e.g. `[∀ node A]  out_A ~= sum(egress from A)` |
| `ρ` | residual (rho) | the signed left-minus-right value of a rule at one grounded point; `ρ = 0` means the rule holds exactly there |
| `s` | scale | a positive robust magnitude used to make the residual dimensionless (`dsl/evaluate.py`, `ground`) |
| `v_op(ρ, s)` | violation | the per-point relative deviation the band must cover: `|ρ|/s` for `~=`/`==`, `max(0, ρ)/s` for `≤`, `max(0, −ρ)/s` for `≥` (`evaluator/band.py::violation_magnitude`) |
| `Γ` | grammar | the finite induced vocabulary (binders, roles, operators, aggregations) plus per-binder size caps (`dsl/grammar.py`) |
| `ℋ(Γ)` | hypothesis space | the finite set of admissible, well-typed, non-trivial, signature-unique rules expressible in `Γ` (defined precisely in §1) |
| `ε` | band (epsilon) | the tolerance: a point "holds" iff `v_op ≤ ε` |
| `τ₀` | base bar | the base Wilson-lower-bound hold-rate threshold (`hold_rate_threshold`, default 0.62) |
| `θ(r)` | per-rule bar | the acceptance bar for rule `r`: `τ₀` plus non-negative generic penalties, clamped (`evaluator/threshold.py`) |
| `π` | operating point | the full dial setting `(ε, τ₀, α, penalty coefficients, band mode)` |
| `α` | confidence | the confidence level (default 0.05); `z = z(α)` the matching normal quantile |
| `k(r), n(r)` | counts | number of grounded points that hold, and total grounded points |
| `L_α(k, n)` | Wilson lower bound | the one-sided lower confidence bound on the true hold-rate (`evaluator/metrics.py::wilson`) |
| `Acc(r)` | acceptance predicate | the pure Boolean function deciding whether `r` is kept (defined in §2) |
| `Π` | portfolio | the final reported set of invariants (`discovery/archive.py`) |
| `⊨` | entailment | `a ⊨ b` means every assignment of leaf values satisfying `a` also satisfies `b` |

Two conventions matter. First, "for fixed `Γ`" always means the induction step has already run and its output is held constant. Second, all three guarantees are *conditional on `Γ`*; none of them claims anything about invariants `Γ` cannot express, and that boundary is restated at the end of each section.

---

## 1. Guarantee I: grammar-conditioned determinism and exhaustiveness

### Intuition

Two properties, one theme. **Determinism**: run the engine twice on the same grammar, data, dials, and seed, and you get byte-for-byte the same portfolio. There is no hidden randomness in the search, no "lucky run." **Exhaustiveness**: the search does not sample or heuristically explore the hypothesis space; it enumerates *all of it*, and the reporting step keeps a representative of every accepted candidate. Every invariant the grammar can express is generated and evaluated, and every distinct accepted behaviour survives into the portfolio, so nothing expressible-and-accepted is skipped for want of looking or dropped in reporting. Together these make the engine an *instrument* rather than a *heuristic*: given a grammar, its output is a well-defined, reproducible, complete-at-the-search-level function of its inputs. This is what makes results auditable and regressions detectable.

The qualifier *grammar-conditioned* is the honest caveat: the grammar `Γ` itself comes from a language model and is not deterministic across inductions. Determinism holds *once `Γ` is fixed*, which is exactly the regime in which the calibration loop operates (it induces once, then re-scores the same `Γ` under many dials).

### Formal statement

Let `Score` denote the deterministic map from a fixed `(dataset, grammar)` and dials `(θ, seed)` to a portfolio.

**Finiteness lemma.** `ℋ(Γ)` is finite. For each binder `b`, the leaf set (single-column roles plus aggregated family roles) is finite; the enumerator forms only terms whose AST complexity is at most the binder's complexity cap, whose additive arity is at most the binder's arity cap, and whose multiplicative degree is at most the binder's degree cap (`dsl/grammar.py::default_binder_caps`, `Grammar.complexity_cap/add_arity_cap/degree_cap`). There are finitely many ASTs of bounded size over a finite leaf set, finitely many operators, so the candidate set per binder is finite, and `ℋ(Γ)`, a subset of the union over finitely many binders, is finite. ∎

**Determinism.** For fixed `(Γ, D, θ, seed)`, `Π = Score(Γ, D, θ, seed)` is single-valued: the same inputs always produce the same `Π`.

**Exhaustiveness (E1: enumeration).** Define `ℋ(Γ) = { normalize(r) : r generated by the caps, `r` admissible, `r` non-trivial }`, deduplicated by canonical signature. Then the enumerator returns *exactly* `ℋ(Γ)`, and every element is evaluated. A zero `max_rules`, `max_linear_leaves`, `max_nonlinear_leaves`, or `max_conditioned_rules` permits the corresponding finite bounded space, while a positive value is a safety ceiling that raises `SearchSpaceTruncatedError` before any partial prefix can be reported as exhaustive.

**Exhaustiveness (E2: reporting).** Let `Acc_π = { r ∈ ℋ(Γ) : Acc(r) }` be the accepted set (the acceptance predicate `Acc` is defined in §2). For every `r ∈ Acc_π` that is not one of the two hygiene-excluded shapes and is not a *lag shadow* (below), the portfolio `Π` contains a representative `r′` with `r′ ⊨ r` (either Z3-equivalent to `r`, or no longer than `r` and subsuming it). Equivalently, every Z3-equivalence class that contains an accepted rule has a representative in `Π`. So the pipeline drops nothing accepted except redundancy, the two hygiene shapes, and lag shadows.

*The lag-shadow exemption, stated honestly.* `discovery/archive.py::_suppress_lag_shadows` drops an accepted sign bound on `LAG_k(x)` when the same-role, same-direction bound on `x` itself is retained **and** holds tolerance-free on the full pre-subsample population (`raw_exact_sign`). This is the one suppression that Z3 does *not* license: the encoding treats `LAG_k(x)` and `x` as independent real leaves, so `x ≥ 0` does not entail `LAG_k(x) ≥ 0` as a formula. The entailment here is *over the data* rather than over the formula: if every observed value of `x` satisfies the bound exactly, then every lagged copy of those same values does too, because a lag only re-indexes the column. The suppression is therefore sound with respect to `D` while being invisible to the logical screen, which is precisely why it is carved out of E2 instead of being folded into the Z3-equivalence claim. It exists because retaining lag bounds unconditionally drives `null_temporal_accepted` from 0 to roughly 20 -- the portfolio fills with one restatement of the same sign law per lag -- and a false-discovery count of zero is a hard requirement.

### Proof

*Determinism.* `Score` is a composition of deterministic maps. Enumeration is a pure function of `Γ`: `discovery/propose.py::_enumerate` iterates binders and terms in sorted order and caches the result, so its output is order-stable. Grounding is a function of `D` and the seed, with any subsampling drawn from a seeded generator (`config.py::DiscoveryConfig.seed`). The adaptive band's calibration split is drawn from a seeded permutation (`evaluator/band.py::_split(n, holdout_frac, seed)`), so the fitted `ε` is reproducible. The Wilson bound is a closed-form formula (`evaluator/metrics.py::wilson`). All logical checks are Z3 decisions over fixed formulas, which return the same verdict every time. The archive iterates its cells in insertion order and breaks ties by a total order on `(hold-rate lower bound, hold-rate, operator strength, −length, MDL)` (`discovery/archive.py::_better`), so its output is a deterministic function of the evaluation order, which is itself deterministic. A composition of deterministic maps is deterministic. ∎

*Exhaustiveness (E1: enumeration).* `discovery/propose.py::_candidate_rules` yields, for every binder, all base terms, all scaled terms, all bounded additive terms, every lag from 1 through `max_lag`, every configured rolling window, and (when the degree cap allows) all product and ratio terms within the caps, then forms every operator-typed comparison among the allowed operand shapes. `_enumerate` normalizes each candidate, keeps it iff it is admissible under the type system (`dsl/typecheck.py::is_admissible`) and not a Z3-triviality (`logic/solver.py::is_trivial`), and deduplicates by canonical signature. `propose()` returns the entire cached list. A positive `SearchConfig.max_rules` is checked as a ceiling and raises before any incomplete prefix can be evaluated. By the finiteness lemma the list is finite, so the enumeration terminates and every element of `ℋ(Γ)` is scored. ∎

*Exhaustiveness (E2: reporting).* By E1 an accepted `r` is enumerated and evaluated. Since `Acc(r)` holds, `discovery/archive.py::add(r)` does not reject it on the acceptance check; if `r` is a hygiene shape it is excluded by hypothesis. Otherwise `add` places `r` in its `(binder, leaf-set)` cell, and the cell logic guarantees a surviving representative: an equivalent present rule keeps the *better* of the two (`_better`, a total order on confidence, then operator strength, then shorter length); a shorter present rule that subsumes `r` is kept in its place; and `r` replaces a longer present rule it subsumes (`discovery/archive.py::add`, using `logic/solver.py::equivalent`, `subsumes`). In every branch the cell retains an `r′` with `r′ ⊨ r`; `archive.py::portfolio(non_redundant=True)` then keeps one representative per class. Hence no accepted *behaviour* is lost; only redundant longer or weaker syntactic forms are pruned. ∎

### Why it matters, and the boundary

Determinism gives reproducibility: a portfolio can be regenerated exactly for audit, and any change in output is attributable to a change in `(Γ, D, θ, seed)`, never to search noise. Exhaustiveness gives completeness *relative to the grammar* at two levels: the search cannot miss an expressible invariant because it "didn't try that candidate" (E1), and the reporting step cannot silently drop an accepted one (E2, up to redundancy and the two hygiene shapes). This grammar-relative completeness is distinct from Guarantee III, which is completeness relative to your *known* invariants. The boundary here is the qualifier itself: determinism is conditional on `Γ`, and different inductions of `Γ` may differ. The calibration loop mitigates this by re-inducing into an *accumulated* grammar so later inductions can only grow the space, never drop a previously found role (`calibrate.py::_merge_specs`), but this is a monotonicity property of the loop, not of a single induction.

---

## 2. Guarantee II: soundness

### Intuition

Everything in the portfolio has earned its place. Nothing reported is a logical triviality (a statement true by form, like `x = x`, or false by form, like `x ≠ x`); nothing is reported without evidence; and every reported invariant clears a *pessimistic* statistical bar, not an optimistic point estimate. Concretely, Autogram does not report "this held on 63% of the samples we happened to draw"; it reports "even accounting for sampling luck, we are confident the true hold-rate is at least the required threshold." Soundness is the promise that a invariant in the portfolio is a real, supported, statistically-vetted pattern, not an artifact of the search.

### Formal statement

The acceptance predicate is a pure, label-blind Boolean function (`discovery/evaluate.py::DataOnlyEvaluator.evaluate`):

> **`Acc(r) ≡ ¬trivial(r) ∧ Fin(r) ∧ n(r) > 0 ∧ Sup(r) ∧ Var(r) ∧ Fit(r) ∧ L_α(k(r), n(r)) ≥ θ(r) ∧ Grp(r) ∧ Lift(r)`.**

Membership in the portfolio requires acceptance: `Π ⊆ { r : Acc(r) }` (the archive only ever adds accepted rules, `discovery/archive.py::add`, and additionally drops two degenerate *hygiene shapes* and lag shadows, so it is a subset). Therefore every `r ∈ Π` satisfies all nine conjuncts:

- **(S1) Non-triviality.** `trivial(r)` is decided by Z3: `r` is a tautology iff its atom is valid for all real assignments of its leaves, and a contradiction iff its negation is valid (`logic/solver.py::is_tautology`, `is_contradiction`, `is_trivial`). No `r ∈ Π` is either.
- **(S2) Finite arithmetic.** `Fin(r)`: the rule's own expression must not exceed `float64` on more than a `max_overflow_fraction` share of the rows it is offered (default `0`, i.e. not on any row). An overflow is a property of the *candidate*, not of the data — the operands were finite and the expression blew up on them — so such a row is neither dropped as missing nor counted as evidence; a candidate that exceeds the cap is refused with a reason naming the blow-up. The guard covers comparisons (`dsl/evaluate.py::ground` records `Grounded.overflow_points`, enforced by `discovery/evaluate.py::_overflow_rejection`), the post-fit arithmetic of a proportional law (whose fitted coefficient introduces multiplication `ground` never saw), and definitions (`_definition_overflow_rejection`, covering a Boolean definition's target and every bound in its predicate, and a band definition's term). Without this a rule could be graded on whatever rows survived a finite mask while its reported support still described the full population.
- **(S3) Support.** `n(r) > 0`: the rule grounds to at least one data point; a rule that grounds to nothing is rejected (`discovery/evaluate.py`, the `degenerate or n_points == 0` guard). Reported support (`Grounded.support`) counts only the rows the rule actually graded, so undefined and overflowed rows are never presented as evidence.
- **(S4) Conditional support floor.** `Sup(r)`: a *conditioned* rule must additionally clear a minimum number of grounded points and a minimum fraction of the attempted population (`min_condition_points`, `min_condition_fraction`). Crucially this is measured on the rows the rule can actually **grade** (`Grounded.graded_points`, `Grounded.graded_condition_support`, computed before subsampling), not on the rows the condition merely selects, so non-finite operands cannot inflate the apparent evidence. `Sup(r)` is vacuously true for unconditioned rules.
- **(S5) Observed variation.** `Var(r)`: an existence rule (`<|>`) is rejected unless presence and absence are both actually observed on at least one side (`"existence has no observed presence/absence variation"`), so a column that is uniformly present cannot pair with anything by default. `Var(r)` is vacuously true for the other operators.
- **(S6) Fitted-parameter well-posedness.** `Fit(r)`: a rule that carries fitted parameters is rejected unless those parameters are actually determined by the data. A learned threshold must have a non-empty candidate set (`"Boolean threshold could not be fit"`), a learned band centre must have a non-empty fit split (`"band center could not be fit"`), and a categorical priority map must have one observationally unique case order up to permutations that cannot change its total function. The evaluator performs an exact bounded search over all case orders that produce the same typed outputs on every gradeable observed activation pattern. Every pair of differently labelled cases must keep the candidate orientation in all such orders; only then may adjacent same-label cases use the canonical unordered OR-block representation. Thus observing `a ∧ b ∧ c -> x` does not by itself identify both `a->x` and `b->x` above `c->y`: an order such as `b->x, c->y, a->x` may agree on every observed row yet differ on the unseen `a ∧ c ∧ ¬b` combination, and the rule is rejected with `"categorical priority case order is not uniquely identifiable from observed combinations"` (`discovery/evaluate.py::_category_case_order_analysis`). Missing targets, missing case values, and rows outside the rule condition supply no precedence evidence. The runtime definition-null gate calls the same population/identifiability analysis before recording or rechecking categorical gradeability (`discovery/validate.py::_runtime_categorical_requirements`, `_validate_runtime_categorical_null`, `null_definitions_at`). Without these checks a rule could be "accepted" at parameters the data never pinned down. `Fit(r)` is vacuously true for a rule with no fitted parameters.
- **(S7) Statistical vetting.** `L_α(k(r), n(r)) ≥ θ(r)`: the Wilson *lower* confidence bound on the true hold-rate clears the per-rule bar. Reading `L_α` correctly is the crux: it is a one-sided `1 − α` lower confidence bound, so `L_α ≥ θ(r)` states that, allowing for finite-sample fluctuation, the true hold-rate is at least `θ(r)` with confidence `1 − α`. This is a frequentist coverage statement, not a proof that the invariant holds.
- **(S8) Per-group vetting.** `Grp(r)`: when the data carries group keys, the aggregate bound is not enough — every group must clear the bar in its own right (`discovery/evaluate.py::_group_hold_gate`). This is what stops one large well-behaved group from carrying a rule that fails outright on another.
- **(S9) Definition lift.** `Lift(r)`: a definition (`:=`) must additionally beat the majority-class baseline by `definition_min_lift` (`discovery/evaluate.py::_definition_evaluation`), so a rule that merely predicts the common label is rejected. `Lift(r)` is vacuously true for comparisons and bands.

### Proof

(S1)–(S9) are exactly the conjuncts of `Acc`, and `Π ⊆ { r : Acc(r) }` because `discovery/archive.py::add` returns early (adds nothing) unless `ev.accepted` is true, and `evaluate` sets `accepted` only after every guard above. Hence every conjunct holds for every portfolio member. ∎

*Exactness of the logical screen, and its one honest gap.* The Z3 encoding maps `Ref` and `Agg` leaves to real variables, `Scale`/`Add` to linear combinations, and `Const` to rationals (`logic/solver.py::_term_expr`). Over this fragment the triviality, equivalence, and subsumption checks are *exact*: Z3 is a decision procedure for linear real arithmetic, so a linear rule is flagged trivial if and only if it truly is. Products (`Mul`) are encoded as genuine nonlinear terms, which Z3 also decides (real nonlinear arithmetic is decidable), so polynomial forms remain exact. The single gap is *ratios*: `Div` is encoded as a fresh opaque real (`logic/solver.py::_term_expr`, the `Div` branch), so once a ratio appears the logical screen is *sound but incomplete*: it never wrongly flags a genuine invariant as trivial, but a ratio-tautology could slip through the triviality filter and would then have to be caught by the data gate (S7). This gap is inactive by default: the default degree cap is 1 (`dsl/grammar.py`: `max_degree = 1`), so `Div` only enters when the calibration loop widens to degree 2, and only there does the incompleteness apply.

*Form hygiene.* Beyond the predicate, the archive drops two degenerate shapes that are technically acceptable but carry no information: *scaled-slack* one-sided rules (an inequality padded with a shrinking or negative coefficient) and *bloated one-sided* rules (a one-sided bound whose measured side is not an atomic reference), `discovery/archive.py::_is_scaled_slack`, `_is_bloated_one_sided`. This is soundness of *form*: it prevents the portfolio from filling with vacuous slack variants of real invariants.

### Why it matters, and the boundary

Soundness is what lets a reader trust the portfolio without re-deriving each invariant: a reported invariant is non-trivial, supported, and vetted at confidence `1 − α`. The boundary is that (S7) is *statistical, not absolute*: a Wilson bound is a confidence statement, so a true invariant with too little support can miss the bar (a false negative), and a spurious pattern that happens to hold on the sample can, with probability at most `α`-controlled slack, pass (a false positive). The false-positive rate is bounded empirically, not certified, by the structural false-discovery guard described in `docs/calibration_protocol.md` (the threshold floor and the null-regime acceptance count). And the logical screen has the ratio gap noted above, which is a precision leak at degree 2, not a break in the statistical soundness of (S7).

---

## 3. Guarantee III: monotone completeness (relative to the known-invariant frontier)

### Intuition

This is the guarantee that turns "we recovered your known invariants" into a statement about the *unknown* ones. Your known invariants define a frontier of what a genuine invariant looks like on this data: a certain cleanliness (how often it holds), a certain amount of evidence (how many grounded points), a certain simplicity. The guarantee is that **any unknown invariant that dominates a recovered known one (at least as clean, at least as well-evidenced, and at least as simple) is itself guaranteed to be discovered.** A recovered known invariant *vouches for* every candidate it dominates. This is completeness *relative to the known invariants*, which is the honest scope: it is not open-ended discovery of everything true, but a deductive promise that the search catches everything at least as good as what you already know. It holds under `band_mode = global`, the shared-dial regime that `autogram calibrate` applies by default (set in `calibrate.py`'s `CalibrationConfig.band_mode`; the standalone `DiscoveryConfig` default in `config.py` remains adaptive), because one tolerance is applied to every candidate, while the adaptive band judges each candidate on its own residuals and trades this guarantee for precision.

### Formal statement

Work at a fixed `(Γ, D, π)` in global mode, so the band is one shared constant `ε` for every rule. Let `Acc_π = { r ∈ ℋ(Γ) : Acc(r) }` (Guarantee I, E2). The engine reads no "known vs. unknown" label (`Acc` is the same pure function for both), which is exactly what lets membership transfer from a known invariant to any candidate that dominates it.

**Lemma (Wilson monotonicity).** As the continuous Wilson lower bound in `(p̂, n)` with `k = p̂ n`, `L_α` is nondecreasing in `p̂` at fixed `n` and nondecreasing in `n` at fixed `p̂`. *(Verified numerically against `evaluator/metrics.py::wilson`: zero violations over all `n ∈ [2, 400)` and all `k`, all rates over `n ∈ [2, 2000)`, and 200,000 random transfer trials.)*

**Theorem (monotone completeness).** Let `r_known ∈ Acc_π` be a recovered known invariant, and let `r_unk ∈ ℋ(Γ)` satisfy

- **(C1)** `p̂(r_unk) ≥ p̂(r_known)`: it holds at least as often within the shared band `ε` (at least as clean, i.e. less noisy);
- **(C2)** `n(r_unk) ≥ n(r_known)`: at least as much evidence (grounded points);
- **(C3)** `θ(r_unk) ≤ θ(r_known)`: no larger acceptance bar, which in global mode follows from `complexity(r_unk) ≤ complexity(r_known)` and `n_bindings(r_unk) ≥ n_bindings(r_known)`;
- **(C4)** `r_unk` clears the *auxiliary* conjuncts of `Acc` in its own right: `Fin(r_unk) ∧ Sup(r_unk) ∧ Var(r_unk) ∧ Fit(r_unk) ∧ Grp(r_unk) ∧ Lift(r_unk)` (S2, S4, S5, S6, S8, S9).

Then `Acc(r_unk)` holds, so `r_unk` is accepted and (by Guarantee I, E2) recovered.

### Proof

By the Lemma with (C1) and (C2), `L_α(k(r_unk), n(r_unk)) ≥ L_α(k(r_known), n(r_known))`. Since `r_known ∈ Acc_π`, the right side is `≥ θ(r_known)`, and by (C3) that is `≥ θ(r_unk)`. Chaining, `L_α(k(r_unk), n(r_unk)) ≥ θ(r_unk)`; with `n(r_unk) > 0` and `r_unk` non-trivial (it is an enumerated admissible rule), and with (C4) supplying the remaining conjuncts, all of `Acc` holds. ∎

**(C4) is not free, and dropping it makes the theorem false.** The aggregate Wilson bound is a statement about the *pooled* population, so it cannot imply the per-group gate (S8): a candidate can be cleaner and better-evidenced in aggregate than a recovered known invariant and still be rejected because one group fails on its own. `tests/test_archive_propose.py::test_monotone_completeness_requires_the_auxiliary_acceptance_gates` exhibits exactly this — a dominating rule with a strictly higher hold rate at equal `n` and equal threshold, rejected by (S8). The same applies to the other auxiliary conjuncts: a candidate whose own arithmetic overflows is refused however well it scores on the rows that survived (S2), a conditioned candidate can fail the support floor (S4), an existence candidate can lack observed variation (S5), a candidate carrying fitted parameters can fail well-posedness (S6) — a categorical map whose priority is not identifiable is rejected however perfectly it predicts — and a definition can fail the baseline lift (S9), all independently of how it compares on `(p̂, n, θ)`. (C1)–(C3) neutralise the *statistical* comparison only; (C4) is what makes the implication sound. The cleanest way to read the guarantee is therefore: **within the class of candidates for which the auxiliary gates are satisfied — in particular ungrouped, unconditioned, parameter-free comparisons over finite arithmetic, where S2, S4, S5, S6, S8 and S9 are all vacuous — domination on `(p̂, n, θ)` is sufficient for acceptance.**

The chain compares the two rules' *own* empirical Wilson bounds against *their own* bars, never a bound against an unknown population rate, so it is an exact finite-sample implication rather than a probabilistic one: (C1) and (C2) neutralize the finite-sample Wilson slack, and (C3) neutralizes the per-rule penalty gap. That is the deductive core of "the dials your known invariants corroborate also catch the unknowns." Because calibration widens `ε` to admit the noisiest known invariant and lowers the base bar toward the null floor to admit the least-reliable one, the dominated region the frontier vouches for is as large as the known set itself allows.

### Why it matters, and the boundary

This is what makes recovering your known invariants *evidence about the unknowns* rather than a self-check: everything at least as good as a known invariant is provably found too. The boundary is exactly the word *relative*, in three parts. First, it is **relative to the known frontier, not open-ended**: it promises coverage of invariants that dominate a known one, and says nothing about a genuine invariant that is noisier, weaker-evidenced, or more complex than anything you listed, which may or may not be found. Second, it inherits Guarantee I's boundary of being **relative to `Γ`**: a dominating invariant the grammar cannot express is still invisible. Third, it is a **global-mode** statement (the calibration default): under the adaptive band — non-default for calibration, though `DiscoveryConfig`'s own independent default — each candidate is judged at a tolerance fit from its own residuals, so (C1) is not evaluated at a shared `ε` and the transfer does not hold. Recovery is also scored only over the recognized signature shapes (pairwise equality, reference-equals-family-sum, zero, presence pairing, one-sided sign), so a dominating invariant of an unrecognized shape may be discovered as a rule yet not counted as recovered.

---

## 4. How the three fit together

The guarantees compose into the framing sentence from the top. Guarantee I makes the search a *well-defined, complete, reproducible* map once the grammar is fixed: it generates and evaluates every expressible candidate the same way every time, and keeps a representative of every accepted one. Guarantee II filters that map's output to *only sound invariants*: non-trivial, supported, and vetted at confidence `1 − α`. Guarantee III lifts completeness from *relative to the grammar* to *relative to your known invariants*: in the default shared-band mode, every unknown invariant that dominates a recovered known one is provably accepted too.

Put together: **for a fixed grammar `Γ`, dataset `D`, and operating point `π`, Autogram deterministically computes a portfolio that contains exactly the sound, expressible invariants of `Γ` (one representative per accepted behaviour, none missed, none trivial, each statistically vetted), and in the default shared-band mode every unknown invariant at least as clean and simple as one of your known invariants is guaranteed to be among them.**

The honest boundary of the whole edifice is three-fold: *grammar-conditioned* (everything is relative to the induced `Γ`; invariants outside it are invisible), *finite-sample* (Guarantee II's acceptance is a confidence statement, not a proof), and *known-relative* (Guarantee III covers only invariants that dominate a known one, not open-ended discovery). These are limits of scope and of statistics, not gaps in the three guarantees themselves, which hold exactly as stated within their stated conditions.
