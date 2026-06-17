from __future__ import annotations
from typing import Callable
import numpy as np
from .agents import Agents
from .config import Config


# (agents, surfaced, nb_table, rng, cfg) -> received claim per agent, shape (N,)
ReceiverFn = Callable[[Agents, np.ndarray, np.ndarray, np.random.Generator, Config], np.ndarray]

RECEIVERS: dict[str, ReceiverFn] = {}


def register_receiver(name: str) -> Callable[[ReceiverFn], ReceiverFn]:
    def decorator(fn: ReceiverFn) -> ReceiverFn:
        RECEIVERS[name] = fn
        return fn
    return decorator


def record_read(agents: Agents, msgids: np.ndarray, cfg: Config) -> None:
    """Write the chosen message ids into each agent's read-ring (dedup memory)."""
    agents.read_ring[:, agents._read_col] = msgids
    agents._read_col = (agents._read_col + 1) % (2 * cfg.history_window)


@register_receiver("passive")
def receive_passive(agents: Agents, surfaced: np.ndarray, nb_table: np.ndarray,
                    rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """
    Reads the single surfaced message (K=1). No user-side input bias yet.
    Resolves the surfaced column to its claim and records the read.
    Stores the read message's ring slot on the agent for the like step.
    """
    n = cfg.n
    w = cfg.history_window
    col = surfaced[:, 0]
    rows = np.arange(n)

    cand_claim = agents.own_claim[nb_table].reshape(n, nb_table.shape[1] * w)
    cand_msgid = agents.own_msgid[nb_table].reshape(n, nb_table.shape[1] * w)

    received  = cand_claim[rows, col]
    chosen_id = cand_msgid[rows, col]
    record_read(agents, chosen_id, cfg)

    agents.read_neighbour = nb_table[rows, col // w]
    agents.read_col       = col % w

    return received.astype(np.int32)