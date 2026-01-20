#!/bin/bash
#SBATCH --job-name=epsilon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-140          # Adjust based on total combinations
#SBATCH --time=12:00:00       # 24 hours per job
#SBATCH --cpus-per-task=1     # Single core (dummy parallelization)
#SBATCH --mem=16G              # Memory per job
#SBATCH --partition=cpuonly  # Adjust to your cluster

# Print job info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

# Path to your container
CONTAINER_PATH=~/ranking_env.sif

# Get experiment directory (current directory when submitting)
EXPERIMENT_DIR=$(pwd)

# Run the job using container
singularity exec $CONTAINER_PATH python ../../scripts/run_single_job.py $SLURM_ARRAY_TASK_ID $EXPERIMENT_DIR

# Print completion info
echo "=================================================="
echo "End time: $(date)"
echo "=================================================="