from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Network
    n: int = 2_000          # number of agents
    k: int = 6              # WS lattice degree (must be even, >= 2)
    p_rewire: float = 0.1   # WS rewiring probability

    # Claim pool
    n_claims: int = 200     # distinct claims in the pool
    llr_std: float = 1.0    # LLR(c) ~ N(0, llr_std^2)

    # Agent initialisation
    belief_std: float = 0.5  # initial log-odds ~ N(0, belief_std^2)

    # Baseline weight slots (deformed by bias modules)
    gain: float = 1.0

    # Simulation
    n_steps: int = 300
    record_every: int = 1   # record metrics every k steps

    seed: Optional[int] = 42
