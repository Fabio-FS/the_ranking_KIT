from __future__ import annotations
import numpy as np
from .agents import Agents
from .config import Config


def publish(agents: Agents, llr: np.ndarray, cfg: Config) -> None:
    """Write this step's N broadcasts into the ring column, one slot per agent."""
    n = cfg.n
    col = agents._write_col

    agents.own_claim[:n, col]  = agents.last_claim
    agents.own_weight[:n, col] = np.abs(llr[agents.last_claim]).astype(np.float32)
    agents.own_msgid[:n, col]  = agents._msg_counter + np.arange(n)
    agents.own_likes[:n, col]  = 0

    agents._msg_counter += n
    agents._write_col = (col + 1) % cfg.history_window