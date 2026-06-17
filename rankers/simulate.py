"""
simulate.py — main entry point for the Bayesian WS simulation.

Typical notebook usage
----------------------
from rankers.simulate import run, run_replicates
from rankers.config import Config

result  = run(Config(n=2_000, n_steps=300))
agg     = run_replicates(Config(), n_reps=30, parallel=False)
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from .config    import Config
from .network   import build_network, build_neighbor_table
from .claims    import build_claims
from .agents    import Agents
from .update    import step
from .emission  import emit
from .history   import publish
from .metrics   import compute_metrics, SimResult
from .ranker    import RANKERS
from .receiver  import RECEIVERS
from .biases import compose_biases, BIASES

def run(cfg: Config) -> SimResult:
    """
    Run one replicate of the Bayesian simulation.

    The ranker (cfg.ranker) and receiver (cfg.receiver) are looked up by name
    from their registries.

    Returns
    -------
    SimResult with .history (metric trajectories), .final_beliefs, .elapsed_s
    """
    rng = np.random.default_rng(cfg.seed)

    g, flat, offsets = build_network(cfg)
    nb_table = build_neighbor_table(flat, offsets, cfg.n)
    llr    = build_claims(cfg, rng)
    agents = Agents(cfg, rng)

    hist: dict[str, list[float]] = {k: [] for k in ("mean", "std", "variance", "bimodality")}
    t0 = time.perf_counter()

    ranker   = RANKERS[cfg.ranker]
    receiver = RECEIVERS[cfg.receiver]

    # Pick n_tracked random agents to follow throughout the simulation
    tracked = rng.choice(cfg.n, size=cfg.n_tracked, replace=False).astype(np.int32)
    belief_traj: list[np.ndarray] = []   # one (n_tracked,) snapshot per recorded step
    rep_traj:    list[np.ndarray] = []   # one (n_tracked,) seen-count snapshot per recorded step
    
    bias_fn  = compose_biases(cfg.biases)
    publish(agents, llr, cfg)
    for t in range(cfg.n_steps):
        surfaced = ranker(agents, nb_table, llr, rng, cfg)         # platform: surface K posts
        received = receiver(agents, surfaced, nb_table, rng, cfg)  # user: read one
        step(agents, received, llr, rng, cfg, bias_fn)             # process: update model
        emit(agents, llr, rng, cfg)                                # emit: pick next post
        publish(agents, llr, cfg)
        if t % cfg.record_every == 0:
            m = compute_metrics(agents.beliefs)
            for k, v in m.items():
                hist[k].append(v)
            belief_traj.append(agents.beliefs[tracked].copy())
            rep_traj.append(agents.seen[tracked].sum(axis=1).copy())

    result = SimResult(
        history             = {k: np.asarray(v) for k, v in hist.items()},
        final_beliefs       = agents.beliefs.copy(),
        elapsed_s           = time.perf_counter() - t0,
        cfg                 = cfg,
        tracked_agents      = tracked,
        belief_trajectories = np.stack(belief_traj),    # (n_records, n_tracked)
        repertoire_history  = np.stack(rep_traj),        # (n_records, n_tracked)
        info_counts         = agents.seen.sum(axis=1),   # (N,) final unique claims seen
    )
    result.graph = g
    return result


def run_replicates(
    cfg: Config,
    n_reps: int = 30,
    parallel: bool = False,
) -> dict:
    """
    Run n_reps independent replicates and return aggregated trajectories.

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
        return run(replace(cfg, seed=base + rep))

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