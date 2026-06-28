"""Autogram: open-ended discovery of unknown invariants on data it was never tuned for.

The engine judges candidate invariants from **data alone** -- no oracle, no clean frame, no
hand-built role vocabulary, no pre-tuned constants.  A schema is *induced* from column names, an
LLM/random proposer generates typed candidates inside it, and a data-only evaluator (self
calibrated band, name-permutation lift percentile, cross-split/temporal stability, MDL parsimony)
accepts and ranks them into a portfolio of discovered invariants.

Public surface:

* :func:`autogram.discovery.discover` -- run discovery on a named tabular dataset.
* :mod:`autogram.discovery.synth`     -- synthetic datasets with planted invariants.
* :mod:`autogram.discovery.validate`  -- adversarial sanity checks (plant-and-recover, null,
  tautology, rename invariance, ablations).
"""

__version__ = "0.2.0"
