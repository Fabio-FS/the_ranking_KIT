"""
simulate.py — main entry point for the Bayesian WS simulation.

Typical notebook usage
----------------------
from rankers.simulate import run, run_replicates
from rankers.config import Config
from rankers.selection import select_neighbor, select_external

result  = run(Config(n=2_000, n_steps=300))
agg     = run_replicates(Config(), n_reps=30, parallel=False)
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

import numpy as np

from .config    import Config
from .network   import build_network
from .claims    import build_claims
from .agents    import Agents
from .selection import SelectionFn, select_neighbor
from .update    import step
from .metrics   import compute_metrics, SimResult


def run(
    cfg: Config,
    selection: SelectionFn = select_neighbor,
) -> SimResult:
    """
    Run one replicate of the Bayesian simulation.

    Parameters
    ----------
    cfg       : Config   — full experimental specification
    selection : SelectionFn — which claim-selection rule to use
                             (default: select_neighbor, the WS-network rule)

    Returns
    -------
    SimResult with .history (metric trajectories), .final_beliefs, .elapsed_s
    """
    rng = np.random.default_rng(cfg.seed)

    _, flat, offsets = build_network(cfg)
    llr    = build_claims(cfg, rng)
    agents = Agents(cfg, rng)

    hist: dict[str, list[float]] = {k: [] for k in ("mean", "std", "variance", "bimodality")}
    t0 = time.perf_counter()

    for t in range(cfg.n_steps):
        received = selection(agents, flat, offsets, rng, cfg)
        step(agents, received, llr, cfg)
        if t % cfg.record_every == 0:
            m = compute_metrics(agents.beliefs)
            for k, v in m.items():
                hist[k].append(v)

    return SimResult(
        history       = {k: np.asarray(v) for k, v in hist.items()},
        final_beliefs = agents.beliefs.copy(),
        elapsed_s     = time.perf_counter() - t0,
        cfg           = cfg,
    )


def run_replicates(
    cfg: Config,
    n_reps: int = 30,
    selection: SelectionFn = select_neighbor,
    parallel: bool = False,
) -> dict:
    """
    Run n_reps independent replicates and return aggregated trajectories.

    Parameters
    ----------
    cfg       : Config
    n_reps    : int          — number of independent seeds
    selection : SelectionFn
    parallel  : bool         — use ProcessPoolExecutor (useful for n_reps >= 20)

    Returns
    -------
    {
      "mean":          dict[metric, ndarray (T,)]   replicate mean per time-step
      "std":           dict[metric, ndarray (T,)]   replicate std per time-step
      "final_beliefs": ndarray (n_reps, N)
      "elapsed_s":     list[float]
    }
    """
    base = cfg.seed if cfg.seed is not None else 0

    def _one(rep: int) -> SimResult:
        return run(replace(cfg, seed=base + rep), selection=selection)

    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(_one, range(n_reps)))
    else:
        results = [_one(r) for r in range(n_reps)]

    keys    = list(results[0].history.keys())
    stacked = {k: np.stack([r.history[k] for r in results]) for k in keys}  # (R, T)

    return {
        "mean":          {k: stacked[k].mean(0) for k in keys},
        "std":           {k: stacked[k].std(0)  for k in keys},
        "final_beliefs": np.stack([r.final_beliefs for r in results]),
        "elapsed_s":     [r.elapsed_s for r in results],
    }
