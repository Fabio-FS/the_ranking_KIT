#!/bin/bash
#SBATCH --job-name=epsilon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-1          
#SBATCH --time=0:30:00       
#SBATCH --cpus-per-task=1     
#SBATCH --mem=4G              
#SBATCH --partition=dev_cpuonly   

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "=================================================="

CONTAINER_PATH=~/ranking_env.sif
EXPERIMENT_DIR=$(pwd)
PROJECT_ROOT=$(cd ../../ && pwd)

# Set PYTHONPATH for inside the container using SINGULARITYENV_ prefix
export SINGULARITYENV_PYTHONPATH=$PROJECT_ROOT

# Run with container
singularity exec $CONTAINER_PATH python run_single_job.py $SLURM_ARRAY_TASK_ID $EXPERIMENT_DIR

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="