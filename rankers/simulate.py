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
import multiprocessing as mp

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
    agents = Agents(cfg, rng, llr)

# simulate.py — inside run(), replace the hist initialization line
    hist: dict[str, list[float]] = {k: [] for k in ("mean", "std", "variance", "opinion", "polarization")}
    t0 = time.perf_counter()

    ranker   = RANKERS[cfg.ranker]
    receiver = RECEIVERS[cfg.receiver]

    # Pick n_tracked random agents to follow throughout the simulation
    tracked = rng.choice(cfg.n, size=cfg.n_tracked, replace=False).astype(np.int32)
    belief_traj: list[np.ndarray] = []   # one (n_tracked,) snapshot per recorded step
    rep_traj:    list[np.ndarray] = []   # one (n_tracked,) seen-count snapshot per recorded step
    
    bias_fn  = compose_biases(cfg.biases)
    publish(agents, llr, cfg)


    # simulate.py  — near the other trajectory buffers, before the loop
    n_records = (cfg.n_steps + cfg.record_every - 1) // cfg.record_every
    full_traj = np.empty((n_records, cfg.n), dtype=np.float64)   # full-population snapshots
    rec = 0
    for t in range(cfg.n_steps):
        surfaced = ranker(agents, nb_table, llr, rng, cfg)         # platform: surface K posts
        received = receiver(agents, surfaced, nb_table, rng, cfg)  # user: read one
        step(agents, received, llr, rng, cfg, bias_fn)             # process: update model
        emit(agents, llr, rng, cfg)                                # emit: pick next post
        publish(agents, llr, cfg)
        # simulate.py  — inside the loop, the recording block
        if t % cfg.record_every == 0:
            m = compute_metrics(agents.beliefs)
            for k, v in m.items():
                hist[k].append(v)
            belief_traj.append(agents.beliefs[tracked].copy())
            rep_traj.append((agents.seen[tracked] > 0).sum(axis=1).copy())
            full_traj[rec] = agents.beliefs           # no .append, no .copy (row write copies)
            rec += 1

    result = SimResult(
        history             = {k: np.asarray(v) for k, v in hist.items()},
        final_beliefs       = agents.beliefs.copy(),
        elapsed_s           = time.perf_counter() - t0,
        cfg                 = cfg,
        tracked_agents      = tracked,
        belief_trajectories = np.stack(belief_traj),    # (n_records, n_tracked)
        repertoire_history  = np.stack(rep_traj),        # (n_records, n_tracked)
        info_counts         = (agents.seen > 0).sum(axis=1)   # (N,) final unique claims seen
    )
# simulate.py  — after the loop, where result.graph is attached
    result.graph = g
    result.full_belief_traj = full_traj              # (n_records, N), preallocated
    return result


def _run_one(args: tuple[Config, int]) -> SimResult:
    """Module-level (picklable) replicate worker: run one seed."""
    cfg, seed = args
    return run(replace(cfg, seed=seed))



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
    tasks = [(cfg, base + rep) for rep in range(n_reps)]

    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx) as pool:
            results = list(pool.map(_run_one, tasks))
    else:
        results = [_run_one(t) for t in tasks]

    keys    = list(results[0].history.keys())
    stacked = {k: np.stack([r.history[k] for r in results]) for k in keys}  # (R, T)

    return {
        "mean":          {k: stacked[k].mean(0) for k in keys},
        "std":           {k: stacked[k].std(0)  for k in keys},
        "final_beliefs": np.stack([r.final_beliefs for r in results]),
        "elapsed_s":     [r.elapsed_s for r in results],
    }



def run_replicates_and_save_all_trajectories(cfg, n_reps=30, parallel=False):
    base = cfg.seed if cfg.seed is not None else 0
    tasks = [(cfg, base + rep) for rep in range(n_reps)]

    if parallel:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(mp_context=ctx) as pool:
            results = list(pool.map(_run_one, tasks))
    else:
        results = [_run_one(t) for t in tasks]

    keys = list(results[0].history.keys())
    stacked = {k: np.stack([r.history[k] for r in results]) for k in keys}  # (R, T)

    return {
        "mean":          {k: stacked[k].mean(0) for k in keys},
        "std":           {k: stacked[k].std(0)  for k in keys},
        "final_beliefs": np.stack([r.final_beliefs for r in results]),
        "trajectories":  np.stack([r.full_belief_traj for r in results]),  # (R, n_records, N)
        "elapsed_s":     [r.elapsed_s for r in results],
    }



## --- Sweeps over a parameter

def run_beta_sweep(base, betas, n_reps, parallel=True):
    sweep = {}
    for beta in betas:
        print(f"Running β = {beta} ...")
        cfg_beta = replace(base, emission_temp=beta)
        sweep[beta] = run_replicates(cfg_beta, n_reps=n_reps, parallel=parallel)
    return sweep

def run_matrix_sweep(base, bias_configs, ranker_names, n_reps=10, parallel=False):
    """
    Run all (bias, ranker) combinations.
    Returns dict[bias_name][ranker_name] -> run_replicates() aggregate dict.
    """
    results = {}
    n_total = len(bias_configs) * len(ranker_names)
    done = 0
    for bias_name, bias_overrides in bias_configs:
        results[bias_name] = {}
        for ranker in ranker_names:
            done += 1
            print(f"[{done}/{n_total}]  bias={bias_name:15s}  ranker={ranker} ...")
            cfg = replace(base, biases=(bias_name,), ranker=ranker, **bias_overrides)
            results[bias_name][ranker] = run_replicates(cfg, n_reps=n_reps, parallel=parallel)
    return results


def run_ndisinfo_sweep(base, n_disinfo_values, n_reps, parallel=True):
    sweep = {}
    for nd in n_disinfo_values:
        print(f"Running n_disinfo = {nd} ...")
        cfg_nd = replace(base, disinfo_mag=-1.0, n_disinfo=nd)
        sweep[nd] = run_replicates_and_save_all_trajectories(cfg_nd, n_reps=n_reps, parallel=parallel)
    return sweep

