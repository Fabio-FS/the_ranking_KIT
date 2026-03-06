#!/bin/bash
#SBATCH --job-name=epsilon_sweep
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-139          # 3 epsilons × 6 rankers = 18 jobs
#SBATCH --time=24:00:00       
#SBATCH --cpus-per-task=1     
#SBATCH --mem=16G              
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

# Create logs directory
mkdir -p $EXPERIMENT_DIR/logs

# Set PYTHONPATH for inside the container
export SINGULARITYENV_PYTHONPATH=$PROJECT_ROOT

# Run with container - ADD --bind to mount project root!
singularity exec --bind $PROJECT_ROOT:$PROJECT_ROOT $CONTAINER_PATH \
    python $EXPERIMENT_DIR/run_single_job.py $SLURM_ARRAY_TASK_ID $EXPERIMENT_DIR

echo "=================================================="
echo "End time: $(date)"
echo "=================================================="