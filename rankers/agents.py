from __future__ import annotations
import numpy as np
from .config import Config


class Agents:
    """
    All mutable agent state as flat numpy arrays.

    beliefs    float64 (N,)      - log-odds l_i
    seen       bool    (N, M)    - repertoire + novelty memory
    last_claim int32   (N,)      - claim agent i will broadcast next step

    Message window (Model A: each agent reads only neighbour messages from the
    last W steps, single weighted draw over the pooled neighbour messages):

    own_claim  int32   (N+1, W)  - ring of each agent's last W broadcast claims
    own_weight float32 (N+1, W)  - base weight of each (|LLR|); row N is dummy pad
    own_msgid  int64   (N+1, W)  - globally unique id of each message slot
    read_ring  int64   (N, 2W)   - ids of messages each agent has already read
    """
    __slots__ = ("beliefs", "seen", "last_claim", "_emit_buf",
                 "own_claim", "own_weight", "own_likes", "own_msgid", "read_ring",
                 "read_neighbour", "read_col",
                 "_write_col", "_read_col", "_msg_counter",
                 "liked_count", "user_likes")

    def __init__(self, cfg: Config, rng: np.random.Generator) -> None:
        n, m, w = cfg.n, cfg.n_claims, cfg.history_window
        self.beliefs = rng.normal(0.0, cfg.belief_std, n).astype(np.float64)
        self.seen = np.zeros((n, m), dtype=np.int32)

        seed_size = min(cfg.repertoire_seed_size, m)
        seeds = rng.integers(0, m, size=(n, seed_size), dtype=np.int32)
        self.seen[np.arange(n)[:, None], seeds] = True

        init_col = rng.integers(0, seed_size, size=n)
        self.last_claim = seeds[np.arange(n), init_col].astype(np.int32)

        self._emit_buf = np.empty((n, m), dtype=np.float32)

        # Row n is a dummy padding row (weight 0) for ragged neighbour gather.
        self.own_claim  = np.zeros((n + 1, w), dtype=np.int32)
        self.own_weight = np.zeros((n + 1, w), dtype=np.float32)
        self.own_likes  = np.zeros((n + 1, w), dtype=np.int32)
        self.own_msgid  = np.full((n + 1, w), -1, dtype=np.int64)
        self.read_ring  = np.full((n, 2 * w), -2, dtype=np.int64)

        self._write_col   = 0
        self._read_col    = 0
        self._msg_counter = 0

        self.liked_count = np.zeros((n + 1, n + 1), dtype=np.int32)  # [i, j]; row/col n = pad
        self.user_likes  = np.zeros(n + 1, dtype=np.int32)           # lifetime likes; index n = pad