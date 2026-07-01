#!/usr/bin/env python3
"""
Run replicates for a single emission_temp (beta) value.

Invoked by the Slurm array:
    python run_beta.py <task_id> <output_dir>

task_id 0     → beta = 0.0
task_id 1-20  → 20 values log-spaced in [1, 1024] (base-2)
"""
import sys
import os
import pickle
import numpy as np
from dataclasses import replace

from rankers.config import Config
from rankers.simulate import run_replicates

BETAS = [0.0] + list(np.logspace(0, 10, 20, base=2))  # 21 total

BASE = Config(
    n=500,
    k=8,
    p_rewire=0.01,
    n_claims=200,
    claim_scheme="fixed",
    llr_mag=1.0,
    repertoire_seed_size=5,
    belief_std=0.5,
    history_window=1,
    n_surfaced=1,
    ranker="baseline",
    receiver="neighbors",
    biases=("baseline",),
    n_steps=20000,
    record_every=20,
    n_tracked=1,
    seed=404,
    emission_scheme="sign",
)

N_REPS = 100


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_beta.py <task_id> <output_dir>")
        sys.exit(1)

    task_id = int(sys.argv[1])
    output_dir = sys.argv[2]

    beta = BETAS[task_id]
    cfg = replace(BASE, emission_temp=beta)

    print(f"[task {task_id:02d}] β = {beta:.4f}  n_reps = {N_REPS}", flush=True)
    result = run_replicates(cfg, n_reps=N_REPS, parallel=False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"result_beta_{task_id:02d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"beta": beta, "task_id": task_id, "result": result}, f)
    print(f"[task {task_id:02d}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
