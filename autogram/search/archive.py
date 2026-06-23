"""MAP-Elites quality-diversity archive with island parallelism (design Sec. 10.3-10.4).

The archive is a grid of behaviour *cells* keyed by a candidate's descriptor
``(complexity, binder, tightness-bucket)`` (produced by the evaluator).  Each cell keeps
the single best-scoring *accepted* elite, so the search retains a diverse portfolio of
qualitatively different invariants instead of collapsing onto one global optimum -- this
is what lets one run recover invariants at several strictness levels at once (exact,
soft-structural, anti).  ``Islands`` holds several such archives that evolve
semi-independently and exchange elites by periodic migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..evaluator.evaluator import EvaluationResult


@dataclass
class Archive:
    """A MAP-Elites grid: one best-scoring elite per behaviour descriptor cell."""
    cells: Dict[tuple, EvaluationResult] = field(default_factory=dict)

    def add(self, res: EvaluationResult) -> bool:
        """Insert ``res`` if accepted and it beats its cell's incumbent.  Returns improved?"""
        if not res.accepted or not res.descriptor:
            return False
        cur = self.cells.get(res.descriptor)
        if cur is None or res.combined_score > cur.combined_score:
            self.cells[res.descriptor] = res
            return True
        return False

    def elites(self) -> List[EvaluationResult]:
        return list(self.cells.values())

    def best(self) -> Optional[EvaluationResult]:
        es = self.elites()
        return max(es, key=lambda r: r.combined_score) if es else None

    def signatures(self) -> set:
        return {r.rule.signature() for r in self.cells.values()}


@dataclass
class Islands:
    """A ring of independent archives that periodically migrate their best elites."""
    n: int

    def __post_init__(self) -> None:
        self.archives: List[Archive] = [Archive() for _ in range(self.n)]

    def add(self, idx: int, res: EvaluationResult) -> bool:
        return self.archives[idx].add(res)

    def migrate(self) -> None:
        """Send each island's global best to its ring neighbour."""
        bests = [a.best() for a in self.archives]
        for i, b in enumerate(bests):
            if b is not None:
                self.archives[(i + 1) % self.n].add(b)

    def all_elites(self) -> List[EvaluationResult]:
        """De-duplicated union of every island's elites, best score per signature."""
        best: Dict[str, EvaluationResult] = {}
        for a in self.archives:
            for r in a.elites():
                sig = r.rule.signature()
                if sig not in best or r.combined_score > best[sig].combined_score:
                    best[sig] = r
        return list(best.values())
