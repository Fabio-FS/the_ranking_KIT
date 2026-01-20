import sys
import os
import time

project_root = '/hkfs/work/workspace/scratch/eq2170-Rankers_OD/the_ranking_KIT'
sys.path.insert(0, project_root)

# DEBUG: Print everything
print("="*50)
print(f"Project root: {project_root}")
print(f"Does it exist? {os.path.exists(project_root)}")
print(f"Contents: {os.listdir(project_root)}")
print(f"Does src/ exist? {os.path.exists(os.path.join(project_root, 'src'))}")
print(f"sys.path after insert: {sys.path[:3]}")
print("="*50)

# Try importing step by step
try:
    import src
    print(f"✓ Found src at: {src.__file__}")
except ImportError as e:
    print(f"✗ Cannot import src: {e}")

try:
    import src.simulation
    print(f"✓ Found src.simulation")
except ImportError as e:
    print(f"✗ Cannot import src.simulation: {e}")


# Get absolute path to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels to project root
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add to Python path
sys.path.insert(0, project_root)

# Debug print (remove after testing)
print(f"Script dir: {script_dir}")
print(f"Project root: {project_root}")
print(f"Project root contents: {os.listdir(project_root)}")

import json
import itertools
import numpy as np


from src.simulation import run_replicas
from src.analysis import save_results


def load_config(exp_dir):
    """Load base config and parameter grid"""
    with open(f'{exp_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    with open(f'{exp_dir}/param_grid.json', 'r') as f:
        param_grid = json.load(f)
    
    return config, param_grid


def generate_combinations(param_grid):
    """
    Generate all parameter combinations, respecting conditional rules.
    
    Returns:
        list of dicts with format {'OD.epsilon': 0.2, 'Ranker.rule': 'Engagement', ...}
    """
    grid = param_grid['grid']
    conditional_params = param_grid['rules']['conditional_params']
    
    # Base parameters that always vary
    epsilons = grid['OD.epsilon']
    rankers = grid['Ranker.rule']
    
    combinations = []
    
    for epsilon in epsilons:
        for ranker in rankers:
            base_combo = {
                'OD.epsilon': epsilon,
                'Ranker.rule': ranker
            }
            
            # Add conditional parameters based on ranker type
            if ranker == 'Engagement':
                for alpha in grid['Ranker.alpha']:
                    combo = base_combo.copy()
                    combo['Ranker.alpha'] = alpha
                    combinations.append(combo)
            
            elif ranker == 'Narrative':
                for target in grid['Ranker.target_opinion']:
                    combo = base_combo.copy()
                    combo['Ranker.target_opinion'] = target
                    combinations.append(combo)
            
            else:  # Random, Closest, Diverse_Engagement
                combinations.append(base_combo)
    
    return combinations


def update_config(config, params):
    """
    Update config dict with parameter combination.
    
    Args:
        config: base config dict
        params: dict with keys like 'OD.epsilon', 'Ranker.rule'
    """
    updated = config.copy()
    
    for key, value in params.items():
        parts = key.split('.')
        
        if len(parts) == 2:
            section, param = parts
            updated[section][param] = value
        else:
            updated[key] = value
    
    return updated


def make_filename(params):
    """
    Create descriptive filename from parameters.
    
    Example: eps0.20_Random.npz
             eps0.15_Engagement_alpha2.0.npz
             eps0.25_Narrative_target0.8.npz
    """
    epsilon = params['OD.epsilon']
    ranker = params['Ranker.rule']
    
    filename = f"eps{epsilon:.2f}_{ranker}"
    
    if ranker == 'Engagement':
        alpha = params['Ranker.alpha']
        filename += f"_alpha{alpha}"
    elif ranker == 'Narrative':
        target = params['Ranker.target_opinion']
        filename += f"_target{target}"
    
    return filename


def run_job(job_id, exp_dir, n_replicas=100):
    """
    Run simulation for specific job_id.
    
    Args:
        job_id: integer index into parameter combinations
        exp_dir: path to experiment directory
        n_replicas: number of replicas to run
    """
    # Load configurations
    config, param_grid = load_config(exp_dir)
    
    # Generate all combinations
    combinations = generate_combinations(param_grid)
    
    print(f"Total combinations: {len(combinations)}")
    
    if job_id >= len(combinations):
        print(f"ERROR: job_id {job_id} exceeds combinations ({len(combinations)})")
        sys.exit(1)
    
    # Get parameters for this job
    params = combinations[job_id]
    
    print(f"\nJob {job_id} parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")
    
    # Update config with these parameters
    info = update_config(config, params)
    
    # Run simulation
    print(f"\nRunning {n_replicas} replicas...")
    results = run_replicas(info, n_replicas=n_replicas, n_save_trajectories=5)
    
    # Save results
    filename = make_filename(params)
    results_dir = f"{exp_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = f"{results_dir}/{filename}"
    save_results(results, filepath)
    
    print(f"\nJob {job_id} completed!")
    print(f"Saved to: {filepath}.npz")


if __name__ == "__main__":
    start_time = time.time()
    
    if len(sys.argv) < 2:
        print("Usage: python run_single_job.py <job_id> [experiment_dir]")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    exp_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    run_job(job_id, exp_dir)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"TOTAL EXECUTION TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*50}")