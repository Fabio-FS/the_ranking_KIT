import sys
import os
import time
import json
import numpy as np

# Get paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add project root to path
sys.path.insert(0, project_root)

# Import after path setup
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
    
    Uses the conditional_params from param_grid to determine which
    parameters to vary for each ranker type.
    """
    grid = param_grid['grid']
    conditional_params = param_grid['rules']['conditional_params']
    
    epsilons = grid['OD.epsilon']
    rankers = grid['Ranker.rule']
    
    combinations = []
    
    for epsilon in epsilons:
        for ranker in rankers:
            base_combo = {
                'OD.epsilon': epsilon,
                'Ranker.rule': ranker
            }
            
            # Check if this ranker has conditional parameters
            if ranker in conditional_params:
                # Get the list of parameters this ranker needs
                params_to_vary = conditional_params[ranker]
                
                # Build all combinations of these parameters
                param_values = []
                param_names = []
                
                for param_name in params_to_vary:
                    param_key = f'Ranker.{param_name}'
                    if param_key in grid:
                        param_values.append(grid[param_key])
                        param_names.append(param_key)
                
                # Generate all combinations
                import itertools
                for values in itertools.product(*param_values):
                    combo = base_combo.copy()
                    for param_name, value in zip(param_names, values):
                        combo[param_name] = value
                    combinations.append(combo)
            else:
                # No conditional parameters, just use base combo
                combinations.append(base_combo)
    
    return combinations


def update_config(config, params):
    """Update config dict with parameter combination"""
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
    """Create descriptive filename from parameters"""
    epsilon = params['OD.epsilon']
    ranker = params['Ranker.rule']
    
    filename = f"eps{epsilon:.4f}_{ranker}"
    
    if 'Ranker.alpha' in params:
        alpha = params['Ranker.alpha']
        filename += f"_alpha{alpha}"
    if 'Ranker.target_opinion' in params:
        target = params['Ranker.target_opinion']
        filename += f"_target{target}"
    
    return filename


def run_job(job_id, exp_dir, n_replicas=100):
    """Run simulation for specific job_id"""
    config, param_grid = load_config(exp_dir)
    combinations = generate_combinations(param_grid)
    
    print(f"\nTotal combinations: {len(combinations)}")
    print(f"Running job_id: {job_id}")
    
    if job_id >= len(combinations):
        print(f"ERROR: job_id {job_id} exceeds combinations ({len(combinations)})")
        sys.exit(1)
    
    params = combinations[job_id]
    
    print(f"\nJob {job_id} parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")
    
    info = update_config(config, params)
    
    print(f"\nRunning {n_replicas} replicas...")
    results = run_replicas(info, n_replicas=n_replicas, n_save_trajectories=5)
    
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
    
    print(f"Starting job {job_id} in directory {exp_dir}")
    
    run_job(job_id, exp_dir)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"TOTAL EXECUTION TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*50}")