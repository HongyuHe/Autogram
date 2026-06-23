"""Autogram: a proof-of-concept P3 AI-driven evolution/search engine for learning
soft, expressive network invariants.

The engine implements the design in
``docs/learning_soft_expressive_invariants.md`` (Section 10 is the primary spec):

* a typed, total, side-effect-free DSL whose AST is the evolution genotype (Sec. 6.6),
* a multi-objective evaluator with an explicit noise-fitting gate (Sec. 10.1),
* analytic (not searched) epsilon-band fitting (Sec. 10.4),
* a MAP-Elites + island evolutionary middle loop with Thompson-sampled budget (Sec. 10.4),
* individual-evolve-then-assemble catalog construction via greedy submodular
  mine-and-cover with entailment dedup (Sec. 10.3),
* two interchangeable LLM proposer backends (OpenAI API and a context-isolated
  Subagent backend) plus a deterministic name-semantic proposer for reproducible,
  API-free runs.

Nothing in this package modifies the datasets; all data access is read-only.
"""

__version__ = "0.1.0"
