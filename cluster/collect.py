#!/usr/bin/env python3
"""
Collect per-beta pickle files into a single sweep dict.

Usage:
    python collect.py <results_dir>

Reads:  <results_dir>/result_beta_NN.pkl  (one per task)
Writes: <results_dir>/beta_sweep.pkl      (same structure as run_beta_sweep())
"""
import sys
import glob
import pickle


def main():
    if len(sys.argv) != 2:
        print("Usage: python collect.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    files = sorted(glob.glob(f"{results_dir}/result_beta_*.pkl"))

    if not files:
        print(f"No result_beta_*.pkl files found in {results_dir}")
        sys.exit(1)

    sweep = {}
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        beta = data["beta"]
        sweep[beta] = data["result"]
        print(f"  β = {beta:10.4f}  ← {path}")

    out_path = f"{results_dir}/beta_sweep.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(sweep, f)

    print(f"\nCollected {len(sweep)} beta values → {out_path}")


if __name__ == "__main__":
    main()
