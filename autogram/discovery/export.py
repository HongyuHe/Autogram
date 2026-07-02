"""Persist discovered rules to ``.dl`` files."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional


def portfolio_to_dl(result, name: str, *, seed: int = 0, proposer: str = "enumeration", git: str = "unknown") -> str:
    lines = []
    for ev in result.portfolio:
        rule = ev.rule.unparse()
        meta = (f"# {ev.strictness.upper():<10s} eps={ev.eps:.4g} "
                f"hold={ev.hold_rate:.3f}[{ev.hold_rate_lo:.2f},{ev.hold_rate_hi:.2f}] "
                f"supp={ev.support:.2f} mdl={ev.mdl_gain:+.3f}")
        lines.append(f"{rule:<64s} {meta}")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    footer = [
        "",
        "# Autogram discovered invariants (observed data only; no catalogue, no oracle)",
        f"# dataset    : {name}",
        f"# run time   : {ts} (UTC)",
        f"# git        : {git}",
        f"# proposer   : {proposer}  seed={seed}  rounds={result.rounds_run}",
        f"# statistic  : hold-rate with Wilson confidence interval; MDL is tie-break only",
        f"# portfolio  : {len(result.portfolio)} invariant(s)",
    ]
    diagnostics = list(getattr(result, "diagnostics", ()))
    if diagnostics:
        footer.append("# diagnostics:")
        footer.extend(f"# - {msg}" for msg in diagnostics[:20])
    return "\n".join(lines + footer) + "\n"


def write_rules_dl(result, name: str, out_dir: str = "rules", *, seed: int = 0,
                   proposer: str = "enumeration", git: str = "unknown") -> Optional[str]:
    if not result.portfolio:
        return None
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    path = os.path.join(out_dir, f"{safe}_{ts}.dl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(portfolio_to_dl(result, name, seed=seed, proposer=proposer, git=git))
    return path
