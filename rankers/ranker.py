from __future__ import annotations
from typing import Callable
import numpy as np
from .agents import Agents
from .config import Config


# (agents, nb_table, llr, rng, cfg) -> surfaced candidate columns, shape (N, K)
RankerFn = Callable[[Agents, np.ndarray, np.ndarray, np.random.Generator, Config], np.ndarray]

RANKERS: dict[str, RankerFn] = {}


def register_ranker(name: str) -> Callable[[RankerFn], RankerFn]:
    def decorator(fn: RankerFn) -> RankerFn:
        RANKERS[name] = fn
        return fn
    return decorator


def gather_candidates(agents: Agents, nb_table: np.ndarray, cfg: Config):
    """
    Pool every neighbour message from the last W steps into (N, C) arrays.
    C = max_degree * W candidate slots per agent.
    Returns (cand_claim, cand_weight, cand_msgid, cand_likes, cand_sender), each (N, C).
    The pad row (index n) contributes weight-0 slots.
    """
    n, w = cfg.n, cfg.history_window
    d = nb_table.shape[1]
    c = d * w

    cand_claim  = agents.own_claim[nb_table].reshape(n, c)
    cand_weight = agents.own_weight[nb_table].reshape(n, c).copy()
    cand_msgid  = agents.own_msgid[nb_table].reshape(n, c)
    cand_likes  = agents.own_likes[nb_table].reshape(n, c)
    cand_sender = np.repeat(nb_table, w, axis=1)   # (N, d*W), sender id per slot
    return cand_claim, cand_weight, cand_msgid, cand_likes, cand_sender


def mask_already_read(cand_weight: np.ndarray, cand_msgid: np.ndarray,
                      agents: Agents) -> None:
    """Zero the weight of pad slots and any candidate already read (in place)."""
    cand_weight[cand_msgid < 0] = 0.0
    already = (cand_msgid[:, :, None] == agents.read_ring[:, None, :]).any(axis=2)
    cand_weight[already] = 0.0


def draw_without_replacement(cand_weight: np.ndarray, k: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Draw k candidate columns per agent, weighted, without replacement.
    Gumbel-top-k: argtop-k of (log weight + Gumbel noise). Returns (N, k) columns.
    Zero-weight candidates get -inf and are never picked while positives remain.
    """
    n = cand_weight.shape[0]
    logits = np.log(cand_weight, out=np.full_like(cand_weight, -np.inf),
                    where=cand_weight > 0.0)
    g = rng.random(cand_weight.shape, dtype=np.float32)
    np.clip(g, 1e-7, 1.0, out=g)
    keys = logits - np.log(-np.log(g))
    return np.argsort(keys, axis=1)[:, ::-1][:, :k]


# ── Rankers ───────────────────────────────────────────────────────────────────
@register_ranker("baseline")
def rank_baseline(agents, nb_table, llr, rng, cfg):
    _, cand_weight, cand_msgid, _, _ = gather_candidates(agents, nb_table, cfg)
    cand_weight[:] = 1.0
    mask_already_read(cand_weight, cand_msgid, agents)
    return draw_without_replacement(cand_weight, cfg.n_surfaced, rng)


@register_ranker("similarity")
def rank_similarity(agents, nb_table, llr, rng, cfg):
    cand_claim, cand_weight, cand_msgid, _, _ = gather_candidates(agents, nb_table, cfg)
    x = llr[cand_claim]
    distance = np.abs(agents.beliefs[:, None] - x)
    cand_weight = np.exp(-distance).astype(np.float32)
    mask_already_read(cand_weight, cand_msgid, agents)
    return draw_without_replacement(cand_weight, cfg.n_surfaced, rng)


@register_ranker("engagement")
def rank_engagement(agents, nb_table, llr, rng, cfg):
    _, cand_weight, cand_msgid, _, cand_sender = gather_candidates(agents, nb_table, cfg)
    rows = np.arange(cfg.n)[:, None]
    affinity = agents.liked_count[rows, cand_sender]
    cand_weight = (1.0 + affinity).astype(np.float32)
    mask_already_read(cand_weight, cand_msgid, agents)
    return draw_without_replacement(cand_weight, cfg.n_surfaced, rng)


@register_ranker("post_popularity")
def rank_post_popularity(agents, nb_table, llr, rng, cfg):
    _, cand_weight, cand_msgid, cand_likes, _ = gather_candidates(agents, nb_table, cfg)
    cand_weight = (1.0 + cand_likes).astype(np.float32)
    mask_already_read(cand_weight, cand_msgid, agents)
    return draw_without_replacement(cand_weight, cfg.n_surfaced, rng)


@register_ranker("user_popularity")
def rank_user_popularity(agents, nb_table, llr, rng, cfg):
    _, cand_weight, cand_msgid, _, cand_sender = gather_candidates(agents, nb_table, cfg)
    cand_weight = (1.0 + agents.user_likes[cand_sender]).astype(np.float32)
    mask_already_read(cand_weight, cand_msgid, agents)
    return draw_without_replacement(cand_weight, cfg.n_surfaced, rng)


@register_ranker("chronological")
def rank_chronological(agents, nb_table, llr, rng, cfg):
    _, cand_weight, cand_msgid, _, _ = gather_candidates(agents, nb_table, cfg)
    keys = cand_msgid.astype(np.float64)
    keys[cand_weight <= 0.0] = -np.inf
    already = (cand_msgid[:, :, None] == agents.read_ring[:, None, :]).any(axis=2)
    keys[already] = -np.inf
    return np.argsort(keys, axis=1)[:, ::-1][:, :cfg.n_surfaced]