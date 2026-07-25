"""gTIB Consumer Byte Completeness -- synthetic-data emulation framework.

This package generates realistic synthetic telemetry for the *gTIB Consumer Byte
Completeness Internal* SLO described in the project email thread
(``learning_rules_for_production.pdf``). No production data is used; every
quantitative choice is either anchored to a fact stated in that thread or is an
explicit, documented assumption exposed as a tunable knob (see ``config.py`` and
``README.md``).

The pipeline being emulated is::

    consumer traffic --> [Collector] --> (gRPC channel + buffer ~ queue) --> [Presenter] --> metrics

Two cumulative byte counters are measured per consumer:

    collector_input_counted   -- input bytes after initial filtering
    presenter_output_counted  -- output bytes accounted by the Presenter tasks

and the completeness ratio ``SUM(Output_Rate) / SUM(Input_Rate)`` is monitored.

The generator is built around one physical invariant that is baked into every
sample (see ``pipeline.py``)::

    cumulative_input(t) == cumulative_output_physical(t) + backlog(t) + cumulative_true_loss(t)

so that transient burst dips, post-burst catch-up overshoot, and *persistent*
true byte loss all emerge from a byte-conserving model rather than being drawn
by hand.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
