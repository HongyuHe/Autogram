"""Declarative proxy specification and generator (item 7).

A :class:`RegimeSpec` is the calibration tuner's editable list of ``(shape, regime)`` proxy
entries.  Each *shape* is a planted-relationship family the engine's grammar can express; each
*regime* is a small set of generation scalars (noise, strength, size).  A trusted generator
interprets the spec into synthetic datasets on **fresh** entities -- proxies are abstractions of
invariants (shape + regime), never copies of the user's real columns.  The tuner may add, adjust,
or deactivate entries; the null (no-relationship) proxy is always available and non-deletable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from . import synth as S

# Shapes the generator can plant (each is expressible in the bounded grammar).
KNOWN_SHAPES = (
    "row_sum", "col_sum", "two_end", "self_zero",
    "offset_pair", "agg_ref_balance", "presence_pair",
    "nonneg", "nonpos", "ratio", "proportional", "monotone", "windowed_ratio",
    "lag_bound", "sum_balance", "conditional_proportional",
    "conditional_pair",
    "conditional_positive", "conditional_zero",
    "cross_grain", "sustained", "conjunction", "categorical",
    "healthy_band",
)


@dataclass(frozen=True)
class ProxyEntry:
    """One declarative proxy: a shape planted at a given regime, on fresh synthetic entities."""
    shape: str
    noise: float = 0.02
    n_entities: int = 4
    n_snapshots: int = 160
    offset_hold_rate: float = 0.67
    presence_rate: float = 0.6
    temporal_window: int = 5
    active: bool = True


@dataclass
class RegimeSpec:
    """The tuner-editable proxy suite (declarative; no arbitrary code)."""
    entries: List[ProxyEntry] = field(default_factory=list)

    # -- tuner actions (declarative edits) -----------------------------------
    def add(self, shape: str, **regime) -> "RegimeSpec":
        if shape not in KNOWN_SHAPES:
            raise ValueError(f"unknown proxy shape {shape!r}; choose from {KNOWN_SHAPES}")
        self.entries.append(ProxyEntry(shape=shape, **regime))
        return self

    def adjust(self, shape: str, **regime) -> "RegimeSpec":
        self.entries = [replace(e, **regime) if e.shape == shape else e for e in self.entries]
        return self

    def deactivate(self, shape: str) -> "RegimeSpec":
        self.entries = [replace(e, active=False) if e.shape == shape else e for e in self.entries]
        return self

    def active_entries(self) -> List[ProxyEntry]:
        return [e for e in self.entries if e.active]


def default_regime() -> RegimeSpec:
    """The shipped proxy suite: one entry per known shape."""
    return RegimeSpec(entries=[ProxyEntry(s) for s in KNOWN_SHAPES])


def abstract_from_shapes(shapes, seed: int = 0) -> RegimeSpec:
    """Auto-coverage: build a proxy per supported shape present among the calibration invariants.

    ``shapes`` are the grammar shapes abstracted from the user's calibration-split invariants;
    unknown shapes are ignored (they signal a grammar-expressiveness gap, not a proxy gap).  Unlike
    earlier revisions this does **not** fall back to the full default suite when nothing maps: it
    returns an empty ``RegimeSpec`` so the caller (calibration) can fail loudly and tell the user to
    supply a custom regime, rather than silently fabricating proxies for shapes they never declared.
    """
    return RegimeSpec(entries=[ProxyEntry(s) for s in shapes if s in KNOWN_SHAPES])


def generate(entry: ProxyEntry, seed: int = 0):
    """Interpret one declarative entry into a synthetic dataset (fresh entities)."""
    exact_shapes = {
        "two_end",
        "ratio",
        "windowed_ratio",
        "cross_grain",
    }
    return S.make_synthetic(
        n_entities=entry.n_entities,
        n_snapshots=entry.n_snapshots,
        noise=0.0 if entry.shape in exact_shapes else entry.noise,
        seed=seed,
        families=(entry.shape,),
        offset_hold_rate=entry.offset_hold_rate,
        presence_rate=entry.presence_rate,
        temporal_window=entry.temporal_window,
    )


def generate_null(seed: int = 0, n_entities: int = 4, n_snapshots: int = 160):
    """The always-on false-discovery control proxy (no planted relationships)."""
    return S.make_null(n_entities=n_entities, n_snapshots=n_snapshots, seed=seed)
