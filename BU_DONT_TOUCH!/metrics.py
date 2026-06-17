from __future__ import annotations
from dataclasses import dataclass
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
    history:       dict[str, np.ndarray]  # metric -> shape (n_records,)
    final_beliefs: np.ndarray             # shape (N,)
    elapsed_s:     float
    cfg:           Config
