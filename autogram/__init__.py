"""Autogram: guarantees-first discovery of unknown invariants.

The engine induces schema roles from observable column names using one of two backends
(``subagent`` or ``openai``), exhaustively enumerates a bounded typed DSL, uses Z3 for logical
truth/equivalence/subsumption, and measures only data hold-rate with a Wilson confidence interval.
"""

__version__ = "0.2.0"
