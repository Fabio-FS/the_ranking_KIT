#!/bin/bash
#SBATCH --job-name=epsilon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-10           # 11 jobs
#SBATCH --time=4:00:00       
#SBATCH --cpus-per-task=10     # 10 parallel combinations
#SBATCH --mem=32G              # Increased for 10 parallel runs
#SBATCH --partition=cpuonly   

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

CONTAINER_PATH=~/ranking_env.sif
EXPERIMENT_DIR=$(pwd)
PROJECT_ROOT=$(cd ../../ && pwd)

mkdir -p $EXPERIMENT_DIR/logs

export SINGULARITYENV_PYTHONPATH=$PROJECT_ROOT

singularity exec --bind $PROJECT_ROOT:$PROJECT_ROOT $CONTAINER_PATH \
    python $EXPERIMENT_DIR/run_single_job.py $SLURM_ARRAY_TASK_ID $EXPERIMENT_DIR

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="