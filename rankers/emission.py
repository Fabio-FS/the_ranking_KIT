"""
emission.py — stochastic argument emission.

Each agent samples one claim from its repertoire (agents.seen), weighted
toward claims whose LLR agrees in sign with its current belief.

Weight: w(c, i) ∝ exp(emission_temp * l_i * LLR(c))  for c in repertoire(i)

Two backends, auto-selected at import time:

  Numba  (if installed) — inverse-CDF per-agent kernel, parallel across agents,
                          no (N,M) temporary array, only N random draws per step.
                          First call pays ~1–2 s JIT warmup (cached to disk after).

  NumPy  (fallback)     — Gumbel-max with pre-allocated float32 workspace.
                          N×M random draws per step; ~2-3× slower than Numba.
"""

from __future__ import annotations
import math
import numpy as np
from .agents import Agents
from .config import Config


# ── Optional Numba backend ────────────────────────────────────────────────────

try:
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _numba_kernel(
        beliefs: np.ndarray,   # float64 (N,)
        llr:     np.ndarray,   # float32 (M,)
        seen:    np.ndarray,   # bool    (N, M)
        temp:    float,
        u:       np.ndarray,   # float64 (N,) — ONE draw per agent
        out:     np.ndarray,   # int32   (N,)
    ) -> None:
        """
        Inverse-CDF sampling with only N random draws (u shape (N,)).
        Two passes over the inner loop per agent:
          pass 1 — sum exp-weights over the repertoire
          pass 2 — walk the CDF until the threshold is crossed
        Saves ~200× on RNG vs Gumbel-max while staying exact.
        """
        N, M = seen.shape
        for i in numba.prange(N):
            b = beliefs[i]
            # Pass 1: total weight
            w_total = 0.0
            for c in range(M):
                if seen[i, c]:
                    w_total += math.exp(temp * b * float(llr[c]))
            # Pass 2: inverse CDF
            threshold = float(u[i]) * w_total
            cumsum = 0.0
            out[i] = 0
            for c in range(M):
                if seen[i, c]:
                    cumsum += math.exp(temp * b * float(llr[c]))
                    if cumsum >= threshold:
                        out[i] = c
                        break

    _HAS_NUMBA = True

except ImportError:
    _HAS_NUMBA = False


# ── Public function ───────────────────────────────────────────────────────────

def emit(
    agents: Agents,
    llr: np.ndarray,          # (M,) float32
    rng: np.random.Generator,
    cfg: Config,
) -> None:
    """
    Sample one claim per agent from its repertoire, weighted by belief
    alignment, and write the result into agents.last_claim.

    Called after step() so updated beliefs and the freshly received claim
    are both visible in the repertoire.
    """
    if _HAS_NUMBA:
        _emit_numba(agents, llr, rng, cfg)
    else:
        _emit_numpy(agents, llr, rng, cfg)


# ── NumPy path (Gumbel-max) ───────────────────────────────────────────────────

def _emit_numpy(
    agents: Agents,
    llr: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
) -> None:
    N, M = cfg.n, cfg.n_claims
    logits = agents._emit_buf   # pre-allocated (N, M) float32

    # logits[i, c] = emission_temp * belief_i * LLR(c)
    np.multiply(
        agents.beliefs.astype(np.float32)[:, None],
        (cfg.emission_temp * llr)[None, :],
        out=logits,
    )

    # Mask unseen: -inf + any finite value = -inf (IEEE 754)
    logits[~agents.seen] = -np.inf

    # Gumbel(0,1) = -log(-log(u)),  u ~ Uniform(0,1)
    g = rng.random((N, M), dtype=np.float32)
    np.clip(g, 1e-7, 1.0, out=g)
    np.log(g, out=g)
    np.negative(g, out=g)
    np.log(g, out=g)
    np.negative(g, out=g)

    logits += g
    agents.last_claim[:] = logits.argmax(axis=1).astype(np.int32)


# ── Numba path (inverse-CDF, N draws) ────────────────────────────────────────

def _emit_numba(
    agents: Agents,
    llr: np.ndarray,
    rng: np.random.Generator,
    cfg: Config,
) -> None:
    u = rng.random(cfg.n)   # float64 (N,) — one draw per agent
    _numba_kernel(
        agents.beliefs,
        llr,
        agents.seen,
        float(cfg.emission_temp),
        u,
        agents.last_claim,
    )
