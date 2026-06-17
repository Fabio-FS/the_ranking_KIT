from __future__ import annotations
import numpy as np
from .agents import Agents
from .config import Config


def step(
    agents: Agents,
    received: np.ndarray,   # (N,) claim id received by each agent this step
    llr: np.ndarray,        # (n_claims,) LLR value of each claim
    cfg: Config,
    *,
    w_compat:   np.ndarray | float = 1.0,
    w_source:   np.ndarray | float = 1.0,
    w_salience: np.ndarray | float = 1.0,
) -> None:
    """
    Vectorised in-place belief update for all N agents simultaneously.

        l_i <- l_i + gain * w_novelty_i * w_compat_i * w_source_i * w_salience_i * LLR(c_i)

    Baseline: gain=1, all weight args=1.
    Bias modules override individual weight arrays without touching this function.

    w_novelty is always computed internally from agents.seen (normative de-duplication).
    The other weight slots can be scalars (broadcast) or (N,) arrays.
    """
    idx = np.arange(cfg.n, dtype=np.int32)

    # Novelty: 1.0 on first exposure, 0.0 on repeat
    w_novelty = (~agents.seen[idx, received]).astype(np.float64)  # (N,)

    agents.beliefs += (
        cfg.gain
        * w_novelty
        * w_compat
        * w_source
        * w_salience
        * llr[received].astype(np.float64)
    )

    agents.seen[idx, received] = True
    agents.last_claim[:] = received
