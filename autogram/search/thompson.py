"""Thompson-sampling budget allocation over islands (design Sec. 9.1, 10.4).

The AutoHarness-inspired view (Sec. 9.1): treat each island as an arm of a Bernoulli
bandit whose reward is "this evaluation improved the archive".  We keep a Beta(alpha,
beta) posterior per arm and, each step, sample a success rate from every posterior and
spend the next evaluation on the arm with the highest draw.  Islands that keep producing
new elites are explored more; stale islands are revisited just often enough to escape
local optima.  This is the explicit exploration-exploitation controller the doc proposes
as an alternative/complement to uniform round-robin over islands.

The sampler uses two Gamma draws to form a Beta variate, so it needs no SciPy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List


def _beta_sample(rng: random.Random, a: float, b: float) -> float:
    x = rng.gammavariate(a, 1.0)
    y = rng.gammavariate(b, 1.0)
    return x / (x + y) if (x + y) > 0 else 0.0


@dataclass
class ThompsonAllocator:
    """Beta-Bernoulli Thompson sampling over ``n`` island arms."""
    n: int
    prior_a: float = 1.0
    prior_b: float = 1.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.alpha: List[float] = [self.prior_a] * self.n
        self.beta: List[float] = [self.prior_b] * self.n

    def select(self) -> int:
        draws = [_beta_sample(self.rng, self.alpha[i], self.beta[i]) for i in range(self.n)]
        return max(range(self.n), key=lambda i: draws[i])

    def update(self, arm: int, reward: float) -> None:
        """Reward in [0,1]; 1 == the evaluation improved the archive."""
        r = max(0.0, min(1.0, reward))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r

    def rates(self) -> List[float]:
        return [self.alpha[i] / (self.alpha[i] + self.beta[i]) for i in range(self.n)]
