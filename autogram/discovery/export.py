"""Persist a discovered portfolio to a ``.pl`` rules file under ``./rules``.

Each accepted invariant is written as one line ``<unparsed rule>  # <STRICTNESS> <metrics>``,
followed by a provenance footer.  This restores the on-disk rule artifact the pre-redesign
pipeline produced under ``./rules`` -- but it is now emitted from the *data-only* discovery
portfolio (held-out coverage, name-permutation lift, stability, MDL gain), with no catalogue
and no oracle anywhere in the provenance.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional


def portfolio_to_pl(result, name: str, *, seed: int = 0, proposer: str = "random",
                    git: str = "unknown") -> str:
    """Render a :class:`~autogram.discovery.loop.DiscoveryResult` as ``.pl`` rule text."""
    lines = []
    for ev in result.portfolio:
        rule = ev.rule.unparse()
        meta = (f"# {ev.strictness.upper():<6s} "
                f"eps={ev.eps:.4f} cov={ev.coverage:.3f}[{ev.coverage_lo:.2f},{ev.coverage_hi:.2f}] "
                f"op_cov={ev.operating_cov:.3f} supp={ev.support:.2f} "
                f"lift={ev.lift:.4g} p={ev.lift_percentile:.4f} "
                f"stab={ev.stability_std:.3f} mdl={ev.mdl_gain:+.3f}")
        lines.append(f"{rule:<60s} {meta}")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    footer = [
        "",
        "# Autogram discovered invariants (data-only; no catalogue, no oracle)",
        f"# dataset    : {name}",
        f"# run time   : {ts} (UTC)",
        f"# git        : {git}",
        f"# proposer   : {proposer}  seed={seed}  rounds={result.rounds_run}  "
        f"reinductions={result.reinductions}",
        f"# portfolio  : {len(result.portfolio)} invariant(s); "
        "each line is `<invariant>  # <strictness> <metrics>`",
    ]
    return "\n".join(lines + footer) + "\n"


def write_rules_pl(result, name: str, out_dir: str = "rules", *, seed: int = 0,
                   proposer: str = "random", git: str = "unknown") -> Optional[str]:
    """Write the portfolio to ``<out_dir>/<name>_<UTCstamp>.pl``; return the path (or ``None``).

    Returns ``None`` without writing when the portfolio is empty (nothing was discovered).
    """
    if not result.portfolio:
        return None
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(out_dir, f"{name}_{ts}.pl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(portfolio_to_pl(result, name, seed=seed, proposer=proposer, git=git))
    return path
