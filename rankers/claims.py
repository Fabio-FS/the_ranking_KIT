from __future__ import annotations
import numpy as np
from .config import Config

def build_claims(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Draw LLR values for the claim pool, split into two populations:
      signal   claims ~ N(mu_signal, sigma_signal^2)   — a real, directed signal
      flooding claims ~ N(0,         sigma_flood^2)     — zero-mean noise

    flood_fraction sets how many of the n_claims are flooding.
    """
    n_flood = int(cfg.flood_fraction * cfg.n_claims)
    n_signal = cfg.n_claims - n_flood

    signal = rng.normal(cfg.mu_signal, cfg.sigma_signal, size=n_signal)
    flood = rng.normal(0.0, cfg.sigma_flood, size=n_flood)

    llr = np.concatenate([signal, flood])
    return llr.astype(np.float32)
