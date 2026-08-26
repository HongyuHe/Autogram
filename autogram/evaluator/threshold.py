"""Per-rule acceptance threshold (item: per-law precision gate).

The evaluator's single global ``hold_rate_threshold`` is a blunt instrument: it applies the same
Wilson-lower-bound bar to a tight, well-separated equality and to a complex, weakly-separated,
low-support form.  A spurious candidate that squeaks over one global bar is the main contributor to
a bloated portfolio.  ``rule_threshold`` turns the bar into a **per-rule** quantity by adding small
*precision penalties* computed from features already available at evaluation time -- so a rule that
looks structurally fragile must clear a higher hold-rate to be kept, while a clean strong law is
unaffected.

Design invariants:

* ``policy == "global"`` reproduces the flat ``hold_rate_threshold`` exactly (opt-out, and the
  penalties are additive so the per-rule bar is never *below* the global one).
* penalties key only on *generic*, dataset-agnostic features (strictness, complexity, support, the
  fitted band ``eps``) -- never on any specific invariant, so the Tuner may set them under the
  "generic knobs only" rule.
* separation/existence forms use a fixed tolerance rather than a fitted coverage band, so the
  band-separation penalty does not apply to them.
"""

from __future__ import annotations


def rule_threshold(cfg, *, op: str, strictness: str, complexity: int,
                   n_bindings: int, eps: float) -> float:
    """Return the per-rule Wilson-lower-bound bar this candidate must clear to be accepted.

    ``cfg`` is a :class:`~autogram.config.DiscoveryConfig`.  With ``cfg.threshold_policy ==
    "global"`` this is just ``cfg.hold_rate_threshold``.  Otherwise the base bar is raised by:

    * a **band-separation** penalty (adaptive band only): the closer the fitted ``eps`` sits to the
      global tolerance cap, the weaker the near-zero core, so the bar rises toward the cap.
    * a **complexity** penalty for forms larger than ``thr_base_complexity`` (bigger forms have more
      ways to hold spuriously).
    * a **low-support** penalty when a rule grounds on few bindings (fragile evidence).

    The result is clamped to ``[hold_rate_threshold, thr_ceiling]``.
    """
    base = float(cfg.hold_rate_threshold)
    if getattr(cfg, "threshold_policy", "global") == "global":
        return base

    thr = base
    # Band-separation penalty: only meaningful for coverage-band ops with a *fitted* eps.  A
    # deadzone protects genuine soft laws (whose band sits well below the cap); the penalty ramps
    # in only as the fitted eps approaches the global tolerance cap -- i.e. the "widen-to-accept"
    # regime where a spurious candidate's band had to stretch toward the ceiling.
    if (strictness not in ("existence", "separation")
            and getattr(cfg, "band_mode", "adaptive") == "adaptive"):
        cap = float(cfg.tolerance)
        frac = (
            min(1.0, max(0.0, (float(eps) / cap - 0.5) * 2.0))
            if cap > 0.0
            else 0.0
        )
        thr += float(cfg.thr_separation_penalty) * frac
    # Complexity penalty for forms above a baseline size.
    over = max(0, int(complexity) - int(cfg.thr_base_complexity))
    thr += float(cfg.thr_complexity_penalty) * over
    # Low-support penalty for fragile groundings.
    if int(n_bindings) < int(cfg.thr_min_bindings):
        thr += float(cfg.thr_low_support_penalty)

    # Clamp to [base, max(base, ceiling)] so penalties can only *raise* the bar and never push it
    # below a caller-set base (even when the base already exceeds the ceiling).
    return min(max(base, float(cfg.thr_ceiling)), thr)
