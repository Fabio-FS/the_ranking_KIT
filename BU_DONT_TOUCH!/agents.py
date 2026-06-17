from __future__ import annotations
import numpy as np
from .config import Config


class Agents:
    """
    All mutable agent state as flat numpy arrays.

    beliefs    float64 (N,)    — log-odds l_i
    seen       bool    (N, M)  — novelty memory: seen[i, c] iff agent i received claim c
    last_claim int32   (N,)    — claim agent i broadcasts this step
    """
    __slots__ = ("beliefs", "seen", "last_claim")

    def __init__(self, cfg: Config, rng: np.random.Generator) -> None:
        n, m = cfg.n, cfg.n_claims
        self.beliefs:    np.ndarray = rng.normal(0.0, cfg.belief_std, n).astype(np.float64)
        self.seen:       np.ndarray = np.zeros((n, m), dtype=bool)
        self.last_claim: np.ndarray = rng.integers(0, m, size=n, dtype=np.int32)
