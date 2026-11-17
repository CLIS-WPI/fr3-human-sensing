"""Deterministic seed derivation and ledger (the project plan.md section 19/20).

A seed alone is not a determinism guarantee. We derive every per-component seed
from one master seed and a stable label so that the same master seed reproduces
the same seeds on every machine, and we record the derived values in a ledger
that is written into the run manifest for provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

_MAX_SEED = 2**31 - 1


def derive_seed(master_seed: int, *labels: str) -> int:
    """Derive a stable, non-negative 31-bit seed from a master seed and labels.

    Uses SHA-256 over a length-prefixed encoding so distinct label tuples that
    would collide under string concatenation remain distinct.
    """
    h = hashlib.sha256()
    h.update(str(int(master_seed)).encode("utf-8"))
    for label in labels:
        raw = str(label).encode("utf-8")
        h.update(len(raw).to_bytes(4, "big"))
        h.update(raw)
    return int.from_bytes(h.digest()[:8], "big") % _MAX_SEED


@dataclass
class SeedLedger:
    """Records derived seeds for every stochastic component in a run."""

    master_seed: int
    seeds: dict[str, int] = field(default_factory=dict)

    def seed_for(self, *labels: str) -> int:
        key = ".".join(str(label) for label in labels)
        if key not in self.seeds:
            self.seeds[key] = derive_seed(self.master_seed, *labels)
        return self.seeds[key]

    def as_dict(self) -> dict[str, object]:
        return {"master_seed": self.master_seed, "derived": dict(sorted(self.seeds.items()))}


__all__ = ["derive_seed", "SeedLedger"]
