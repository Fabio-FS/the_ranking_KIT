from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    # Network
    n: int = 500          # number of agents
    k: int = 32              # WS lattice degree (must be even, >= 2)
    p_rewire: float = 0.01   # WS rewiring probability

    # Claim pool
    n_claims: int = 200     # distinct claims in the pool
    llr_std: float = 1.0    # LLR(c) ~ N(0, llr_std^2)
    claim_scheme: str = "gaussian"   # which builder in claims.SCHEMES
    llr_mag: float = 1.0             # magnitude for the "fixed" scheme
    
    
    # Claim pool — signal vs flooding split
    flood_fraction: float = 1.0     # fraction of n_claims that are zero-mean noise
    mu_signal: float = 1.0          # mean LLR of the signal claims
    sigma_signal: float = 1.0       # std of the signal claims
    sigma_flood: float = 1.0        # std of the flooding (zero-mean) claims

    n_disinfo: int = 0          # extra one-sided disinformation claims
    disinfo_mag: float = 1.0    # LLR of every disinfo claim (signed)
    
    # Agent initialisation
    belief_std: float = 0.5           # initial log-odds ~ N(0, belief_std^2)
    repertoire_seed_size: int = 5     # claims each agent knows at t=0

    # Baseline weight slots (deformed by bias modules)
    gain: float = 1.0

    # Emission
    emission_temp: float = 1.0        # beta in exp(beta * l_i * LLR(c)); 0 = uniform
    emission_scheme: str = "sign"     # "sign" (honest) | "magnitude" (sensationalism)
    
    # Simulation
    n_steps: int = 2000
    record_every: int = 5   # record metrics every k steps

    seed: Optional[int] = 42

    
    history_window: int = 5         # messages from last W steps are visible/readable
    n_surfaced: int = 1             # K: messages the ranker surfaces per agent per step
    n_tracked:  int = 50            # how many agents to track individually
    ranker:   str = "baseline"
    receiver: str = "neighbors"


    # Biases (process stage)
    biases: tuple[str, ...] = ("baseline",)
    confirmation_strength: float = 0.0   # 0 = off
    negativity_multiplier: float = 1.0   # 1 = off
    conservatism_strength: float = 0.0     # 0 = off
    illusory_truth_factor: float = 3.0   # total contribution of a repeated claim ≤ FACTOR * LLR


    # Likes
    like_slope: float = 1.0   # steepness of p_like = ceiling * tanh(slope * w_bias)
    
