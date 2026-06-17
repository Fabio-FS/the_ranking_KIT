from __future__ import annotations
import numpy as np
from .agents import Agents
from .config import Config
from .biases import BiasFn


def like_probability(w_bias: np.ndarray, already_seen: np.ndarray,
                     cfg: Config) -> np.ndarray:
    ceiling = np.where(already_seen, 0.1, 0.5)
    return ceiling * np.tanh(cfg.like_slope * w_bias)


def step(agents: Agents, received: np.ndarray, llr: np.ndarray, rng: np.random.Generator, cfg: Config, bias_fn: BiasFn) -> None:
    idx = np.arange(cfg.n, dtype=np.int32)

    already_seen = agents.seen[idx, received]
    illusory_truth_active = "illusory_truth" in cfg.biases
    repeat_weight = 1.0 if illusory_truth_active else 0.0
    w_novelty = np.where(already_seen, repeat_weight, 1.0)

    w_bias = bias_fn(agents, received, llr, cfg)

    agents.beliefs += (
        cfg.gain
        * w_novelty
        * w_bias
        * llr[received].astype(np.float64)
    )

    agents.seen[idx, received] += 1

    p_like = like_probability(w_bias, already_seen, cfg)
    liked = rng.random(cfg.n) < p_like



    real = agents.read_neighbour != cfg.n
    liked = liked & real

    likers = idx[liked]
    likees = agents.read_neighbour[liked]

    np.add.at(agents.own_likes, (likees, agents.read_col[liked]), 1)
    np.add.at(agents.liked_count, (likers, likees), 1)
    np.add.at(agents.user_likes, likees, 1)