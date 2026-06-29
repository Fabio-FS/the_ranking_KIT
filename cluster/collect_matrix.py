#!/usr/bin/env python3
"""
Collect per-condition pickle files into a single nested dict.

Usage:
    python collect_matrix.py <results_dir>

Reads:  <results_dir>/result_NN_<bias>_<ranker>.pkl  (one per task)
Writes: <results_dir>/matrix_sweep.pkl
        → dict[bias_name][ranker_name] = run_replicates() aggregate dict
          (same structure as run_matrix_sweep() output)
"""
import sys
import glob
import pickle


def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_matrix.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    files = sorted(glob.glob(f"{results_dir}/result_*.pkl"))

    if not files:
        print(f"No result_*.pkl files found in {results_dir}")
        sys.exit(1)

    matrix = {}
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        bias   = data["bias_name"]
        ranker = data["ranker_name"]
        matrix.setdefault(bias, {})[ranker] = data["result"]
        print(f"  [{data['task_id']:02d}] bias={bias:<15s} ranker={ranker:<18s} ← {path}")

    out_path = f"{results_dir}/matrix_sweep.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(matrix, f)

    n = sum(len(v) for v in matrix.values())
    print(f"\nCollected {n} conditions → {out_path}")


if __name__ == "__main__":
    main()
