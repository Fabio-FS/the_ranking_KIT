from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .config import Config


# ── Metrics ───────────────────────────────────────────────────────────────────

def binned_variance(hist, bins):
    centers = (bins[:-1] + bins[1:]) / 2
    mean = np.sum(centers * hist) / np.sum(hist)
    var = np.sum(np.power(centers - mean, 2) * hist) / np.sum(hist)
    max_var = np.power(centers[-1], 2)  # all mass at ±extreme → variance = extreme²
    return var / max_var

def compute_metrics(beliefs: np.ndarray) -> dict[str, float]:
    opinion = 1.0 / (1.0 + np.exp(-beliefs))   # σ(belief): perceived probability
    return {
        "mean":       float(beliefs.mean()),
        "std":        float(beliefs.std()),
        "variance":   float(beliefs.var()),
        "opinion":    float(opinion.mean()),    # average opinion across agents
        "polarization": float(4 * opinion.var())
    }


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SimResult:
    history:             dict[str, np.ndarray]       # metric -> shape (n_records,)
    final_beliefs:       np.ndarray                  # shape (N,)
    elapsed_s:           float
    cfg:                 Config
    tracked_agents:      Optional[np.ndarray] = None # shape (K,)       — agent indices
    repertoire_history:  Optional[np.ndarray] = None # shape (n_records, K) — seen count per step
    belief_trajectories: Optional[np.ndarray] = None # shape (n_records, K) — belief per tracked agent per step
    info_counts:         Optional[np.ndarray] = None # shape (N,)        — total unique claims seen, final
    graph               = None                       # igraph.Graph for network viz

def neighbor_homophily_series(result):
    """
    Edge homophily on probabilities O = σ(belief).
    Per-edge mean of  (1/2 - |O_i - O_j|) * 2  =  1 - 2|O_i - O_j|  ∈ [-1, +1].
      +1 = neighbours identical (touching only same-belief agents)
       0 = neighbours differ by 0.5 on average
      -1 = checkerboard (every edge maximally opposed)
    Returns (n_records,).
    """
    edges = np.array(result.graph.get_edgelist())      # (E, 2)
    i, j = edges[:, 0], edges[:, 1]
    O = 1.0 / (1.0 + np.exp(-result.full_belief_traj))  # (n_records, N)
    agree = 1.0 - 2.0 * np.abs(O[:, i] - O[:, j])       # (n_records, E)
    return agree.mean(axis=1)                           # (n_records,)