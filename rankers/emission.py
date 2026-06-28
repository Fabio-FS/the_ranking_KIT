# emission.py
"""
emission.py — stochastic argument emission.

Each agent samples one claim from its repertoire (agents.seen), weighted
toward claims whose LLR agrees in sign with its current belief.

Emission schemes (cfg.emission_scheme), registered below:

  "sign"      (default substrate) — honest emission:
                  w(c, i) ∝ exp(beta * sign(l_i * LLR(c)))
              Magnitude drops out; every sign-agreeing claim is equally
              likely. beta = emission_temp tunes agree-vs-disagree strength.
              Use a large finite beta for the saturated (uniform-over-
              agreeing) limit; the exponent is bounded to ±beta.

  "magnitude" (sensationalism bias) — pushes agents toward broadcasting
              their most extreme aligned content:
                  w(c, i) ∝ exp(beta * l_i * LLR(c))
              Use a large finite beta to approach the argmax limit.

Two stochastic backends per scheme, auto-selected at import time:

  Numba  (if installed) — inverse-CDF per-agent kernel, parallel across agents,
                          no (N,M) temporary array, only N random draws per step.
                          First call pays ~1–2 s JIT warmup (cached to disk after).

  NumPy  (fallback)     — Gumbel-max with pre-allocated float32 workspace.
                          N×M random draws per step; ~2-3× slower than Numba.
"""

from __future__ import annotations
import numpy as np
from .agents import Agents
from .config import Config


# ── Scheme score functions ────────────────────────────────────────────────────
# Each returns the per-claim logit exponent (before beta scaling) for the
# numpy path. score(belief, llr) broadcast over (N, M).

def _score_sign(beliefs, llr):
    return np.sign(beliefs[:, None] * llr[None, :])

def _score_magnitude(beliefs, llr):
    return beliefs[:, None] * llr[None, :]


# ── Optional Numba backend ────────────────────────────────────────────────────

try:
    import numba
    import math

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _numba_kernel_sign(beliefs, llr, seen, temp, u, out):
        N, M = seen.shape
        for i in numba.prange(N):
            b = beliefs[i]
            w_total = 0.0
            for c in range(M):
                if seen[i, c]:
                    s = b * float(llr[c])
                    sign = (s > 0.0) - (s < 0.0)
                    w_total += math.exp(temp * sign)
            threshold = float(u[i]) * w_total
            cumsum = 0.0
            out[i] = 0
            for c in range(M):
                if seen[i, c]:
                    s = b * float(llr[c])
                    sign = (s > 0.0) - (s < 0.0)
                    cumsum += math.exp(temp * sign)
                    if cumsum >= threshold:
                        out[i] = c
                        break

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _numba_kernel_magnitude(beliefs, llr, seen, temp, u, out):
        N, M = seen.shape
        for i in numba.prange(N):
            b = beliefs[i]
            w_total = 0.0
            for c in range(M):
                if seen[i, c]:
                    w_total += math.exp(temp * b * float(llr[c]))
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


# ── Scheme registry ───────────────────────────────────────────────────────────
# Each scheme maps to (numba_kernel, numpy_score_fn).

EMISSION_SCHEMES = {
    "sign":      (_numba_kernel_sign if _HAS_NUMBA else None,      _score_sign),
    "magnitude": (_numba_kernel_magnitude if _HAS_NUMBA else None, _score_magnitude),
}


# ── Public function ───────────────────────────────────────────────────────────

def emit(agents: Agents, llr: np.ndarray, rng: np.random.Generator, cfg: Config) -> None:
    """
    Sample one claim per agent from its repertoire under cfg.emission_scheme,
    writing the result into agents.last_claim.

    Called after step() so updated beliefs and the freshly received claim
    are both visible in the repertoire.
    """
    if _HAS_NUMBA:
        _emit_numba(agents, llr, rng, cfg)
    else:
        _emit_numpy(agents, llr, rng, cfg)


# ── NumPy path (Gumbel-max) ───────────────────────────────────────────────────

def _emit_numpy(agents: Agents, llr: np.ndarray, rng: np.random.Generator, cfg: Config) -> None:
    N, M = cfg.n, cfg.n_claims
    _, score_fn = EMISSION_SCHEMES[cfg.emission_scheme]
    logits = agents._emit_buf   # pre-allocated (N, M) float32

    logits[:] = (cfg.emission_temp * score_fn(agents.beliefs, llr)).astype(np.float32)
    logits[~agents.seen] = -np.inf

    g = rng.random((N, M), dtype=np.float32)
    np.clip(g, 1e-7, 1.0, out=g)
    np.log(g, out=g)
    np.negative(g, out=g)
    np.log(g, out=g)
    np.negative(g, out=g)

    logits += g
    agents.last_claim[:] = logits.argmax(axis=1).astype(np.int32)


# ── Numba path (inverse-CDF, N draws) ────────────────────────────────────────

def _emit_numba(agents: Agents, llr: np.ndarray, rng: np.random.Generator, cfg: Config) -> None:
    kernel, _ = EMISSION_SCHEMES[cfg.emission_scheme]
    u = rng.random(cfg.n)
    kernel(agents.beliefs, llr, agents.seen, float(cfg.emission_temp), u, agents.last_claim)