#!/bin/bash
#SBATCH --job-name=epsilon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-18          # Adjust based on total combinations
#SBATCH --time=1:00:00       # 24 hours per job
#SBATCH --cpus-per-task=1     # Single core (dummy parallelization)
#SBATCH --mem=8G              # Memory per job
#SBATCH --partition=standard  # Adjust to your cluster

# Print job info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

# Activate conda environment (adjust path)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name

# Get experiment directory (current directory when submitting)
EXPERIMENT_DIR=$(pwd)

# Run the job
python ../../scripts/run_single_job.py $SLURM_ARRAY_TASK_ID $EXPERIMENT_DIR

# Print completion info
echo "=================================================="
echo "End time: $(date)"
echo "=================================================="