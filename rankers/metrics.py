from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from .config import Config


# ── Metrics ───────────────────────────────────────────────────────────────────

def bimodality_coeff(x: np.ndarray) -> float:
    """
    Sarle's bimodality coefficient B = (skew^2 + 1) / raw_kurtosis.
    B > 5/9 ~ 0.555 is the conventional threshold for bimodality.
    A uniform distribution sits at exactly 5/9; a normal distribution at 1/3.
    """
    mu  = x.mean()
    d   = x - mu
    var = np.mean(d * d)
    if var < 1e-12:
        return 0.0
    skew = np.mean(d ** 3) / var ** 1.5
    kurt = np.mean(d ** 4) / var ** 2   # raw kurtosis (normal = 3)
    return float((skew ** 2 + 1.0) / kurt)

def binned_variance(hist, bins):
    centers = (bins[:-1] + bins[1:]) / 2
    mean = np.sum(centers * hist) / np.sum(hist)
    var = np.sum(np.power(centers - mean, 2) * hist) / np.sum(hist)
    max_var = np.power(centers[-1], 2)  # all mass at ±extreme → variance = extreme²
    return var / max_var

def compute_metrics(beliefs: np.ndarray) -> dict[str, float]:
    return {
        "mean":       float(beliefs.mean()),
        "std":        float(beliefs.std()),
        "variance":   float(beliefs.var()),
        "bimodality": bimodality_coeff(beliefs),
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

