from __future__ import annotations
import numpy as np
from .config import Config


def build_claims(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Draw LLR values for the claim pool.
    llr[c] ~ N(0, llr_std^2), float32.
    Positive = evidence for the hypothesis; negative = against.
    """
    return rng.normal(0.0, cfg.llr_std, size=cfg.n_claims).astype(np.float32)
