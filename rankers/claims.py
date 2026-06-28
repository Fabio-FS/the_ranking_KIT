from __future__ import annotations
import numpy as np
from .config import Config


def build_gaussian(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Signal vs flooding split.
      signal   claims ~ N(mu_signal, sigma_signal^2)
      flooding claims ~ N(0,         sigma_flood^2)
    flood_fraction sets how many of the n_claims are flooding.
    """
    n_flood = int(cfg.flood_fraction * cfg.n_claims)
    n_signal = cfg.n_claims - n_flood

    signal = rng.normal(cfg.mu_signal, cfg.sigma_signal, size=n_signal)
    flood = rng.normal(0.0, cfg.sigma_flood, size=n_flood)

    return np.concatenate([signal, flood])


def build_fixed(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Fixed-magnitude symmetric pool: half at +llr_mag, half at -llr_mag.
    Zero-mean, single magnitude.
    """
    half = cfg.n_claims // 2
    pos = np.full(half, cfg.llr_mag)
    neg = np.full(cfg.n_claims - half, -cfg.llr_mag)
    return np.concatenate([pos, neg])

def build_disinfo(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Legitimate symmetric pool plus a one-sided disinformation block.
      legitimate: half at +llr_mag, half at -llr_mag (zero-mean ground truth)
      disinfo:    n_disinfo extra claims all at +disinfo_mag
    """
    half = cfg.n_claims // 2
    pos = np.full(half, cfg.llr_mag)
    neg = np.full(cfg.n_claims - half, -cfg.llr_mag)
    disinfo = np.full(cfg.n_disinfo, cfg.disinfo_mag)
    return np.concatenate([pos, neg, disinfo])

SCHEMES = {
    "gaussian": build_gaussian,
    "fixed": build_fixed,
    "disinfo": build_disinfo
}


def build_claims(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the claim-pool builder named by cfg.claim_scheme."""
    llr = SCHEMES[cfg.claim_scheme](cfg, rng)
    return llr.astype(np.float32)

