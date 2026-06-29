#!/bin/bash
# ── Matrix-sweep Slurm array job ─────────────────────────────────────────────
# 30 tasks: one per (bias, ranker) combination (5 biases × 6 rankers)
#
# Usage (from the workspace root on the cluster login node):
#   sbatch cluster/submit_matrix_sweep.sh
#
# Before first run:
#   mkdir -p logs results/matrix_sweep
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=matrix_sweep
#SBATCH --partition=cpu
#SBATCH --array=0-29
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matrix_sweep_%A_%a.out
#SBATCH --error=logs/matrix_sweep_%A_%a.err

WORKSPACE=/pfs/work9/workspace/scratch/ka_eq2170-Rankers

PROJECT=$WORKSPACE/Rankers
RESULTS=$WORKSPACE/results/matrix_sweep
SIF=$WORKSPACE/rankers.sif

mkdir -p "$RESULTS"

echo "=== task $SLURM_ARRAY_TASK_ID  node $(hostname)  $(date) ==="

singularity exec \
    --bind "$WORKSPACE:/workspace" \
    --env  PYTHONPATH=/workspace/Rankers \
    "$SIF" \
    python /workspace/Rankers/cluster/run_matrix.py \
        "$SLURM_ARRAY_TASK_ID" \
        /workspace/results/matrix_sweep

echo "=== done $(date) ==="
