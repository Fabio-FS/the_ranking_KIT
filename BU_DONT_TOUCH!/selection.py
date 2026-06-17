from __future__ import annotations
from typing import Callable
import numpy as np
from .agents import Agents
from .config import Config


# Type alias for all selection rules.
# (agents, neighbors_flat, neighbor_offsets, rng, cfg) -> received_claims (N,) int32
SelectionFn = Callable[
    [Agents, np.ndarray, np.ndarray, np.random.Generator, Config],
    np.ndarray,
]


def select_neighbor(
    agents: Agents,
    flat: np.ndarray,
    offsets: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
) -> np.ndarray:
    """
    Baseline network rule: each agent i samples one random neighbour j
    and receives the claim j last broadcast.

    Fully vectorised — one random draw per agent, two fancy-index lookups.
    Network topology shapes which claims reach which agents.
    """
    n = cfg.n
    degrees = offsets[1:] - offsets[:-1]                  # (N,)
    local = np.floor(rng.random(n) * degrees).astype(np.int32)
    local = np.minimum(local, degrees - 1)                # guard: WS always deg >= k
    chosen_nb = flat[offsets[:-1] + local]                # (N,) neighbour ids
    return agents.last_claim[chosen_nb]                   # (N,) claim ids


def select_external(
    agents: Agents,
    flat: np.ndarray,
    offsets: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
) -> np.ndarray:
    """
    No-network baseline: each agent samples a claim uniformly at random
    from the global pool, independent of topology.

    Use alongside select_neighbor to isolate the network's structural
    contribution to polarisation.
    """
    return rng.integers(0, cfg.n_claims, size=cfg.n, dtype=np.int32)
