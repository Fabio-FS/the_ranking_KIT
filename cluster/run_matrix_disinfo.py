#!/usr/bin/env python3
"""
Run replicates for a single (bias, ranker) combination — disinformation pool.

Identical to run_matrix.py except the claim pool contains 8 disinformation
posts at LLR = -1.0 (one-sided negative push).

Invoked by the Slurm array:
    python run_matrix_disinfo.py <task_id> <output_dir>

task_id maps to one of 30 (bias, ranker) pairs:
    task_id = bias_idx * n_rankers + ranker_idx
"""
import sys
import os
import pickle
import numpy as np
from dataclasses import replace

from rankers.config import Config
from rankers.simulate import run_replicates

BIAS_CONFIGS = [
    ("baseline",       dict()),
    ("confirmation",   dict(confirmation_strength=1.0)),
    ("negativity",     dict(negativity_multiplier=2.0)),
    ("illusory_truth", dict()),
    ("conservatism",   dict(conservatism_strength=1.0)),
]

RANKER_NAMES = [
    "baseline",
    "similarity",
    "engagement",
    "post_popularity",
    "user_popularity",
    "chronological",
]

N_CONDITIONS = len(BIAS_CONFIGS) * len(RANKER_NAMES)  # 30

BASE = Config(
    n=500,
    k=8,
    p_rewire=0.01,
    n_claims=200,
    claim_scheme="disinfo",
    llr_mag=1.0,
    n_disinfo=8,
    disinfo_mag=-1.0,
    repertoire_seed_size=5,
    belief_std=0.5,
    history_window=5,
    n_surfaced=1,
    ranker="baseline",
    receiver="neighbors",
    biases=("baseline",),
    emission_temp=1,
    n_steps=8000,
    record_every=20,
    n_tracked=1,
    seed=41,
)

N_REPS = 100


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_matrix_disinfo.py <task_id> <output_dir>")
        sys.exit(1)

    task_id = int(sys.argv[1])
    output_dir = sys.argv[2]

    n_rankers = len(RANKER_NAMES)
    bias_idx   = task_id // n_rankers
    ranker_idx = task_id %  n_rankers

    bias_name, bias_overrides = BIAS_CONFIGS[bias_idx]
    ranker_name = RANKER_NAMES[ranker_idx]

    cfg = replace(BASE, biases=(bias_name,), ranker=ranker_name, **bias_overrides)

    print(f"[task {task_id:02d}] bias={bias_name:<15s} ranker={ranker_name}  n_reps={N_REPS}", flush=True)
    result = run_replicates(cfg, n_reps=N_REPS, parallel=False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"result_{task_id:02d}_{bias_name}_{ranker_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "task_id":     task_id,
            "bias_name":   bias_name,
            "ranker_name": ranker_name,
            "result":      result,
        }, f)
    print(f"[task {task_id:02d}] Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
