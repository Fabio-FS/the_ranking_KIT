#!/bin/bash
# ── Beta-sweep Slurm array job ────────────────────────────────────────────────
# 21 tasks: task 0 = β=0, tasks 1-20 = log-spaced β in [1, 1024]
#
# Usage (from the workspace root on the cluster login node):
#   sbatch cluster/submit_beta_sweep.sh
#
# Before first run:
#   mkdir -p logs results/beta_sweep
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=beta_sweep
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/beta_sweep_%A_%a.out
#SBATCH --error=logs/beta_sweep_%A_%a.err

# ── paths ─────────────────────────────────────────────────────────────────────
# Set WORKSPACE to wherever the project lives on the cluster.
# Recommended layout inside the workspace:
#   $WORKSPACE/Rankers/          ← git repo (this project)
#   $WORKSPACE/rankers.sif       ← Singularity image
#   $WORKSPACE/results/          ← outputs
#   $WORKSPACE/logs/             ← Slurm logs (create before submitting)
WORKSPACE=/pfs/work9/workspace/scratch/ka_eq2170-Rankers

PROJECT=$WORKSPACE/Rankers
RESULTS=$WORKSPACE/results/beta_sweep
SIF=$WORKSPACE/rankers.sif

mkdir -p "$RESULTS"

echo "=== task $SLURM_ARRAY_TASK_ID  node $(hostname)  $(date) ==="

singularity exec \
    --bind "$WORKSPACE:/workspace" \
    --env  PYTHONPATH=/workspace/Rankers \
    "$SIF" \
    python /workspace/Rankers/cluster/run_beta.py \
        "$SLURM_ARRAY_TASK_ID" \
        /workspace/results/beta_sweep

echo "=== done $(date) ==="
