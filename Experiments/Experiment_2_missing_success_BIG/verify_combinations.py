#!/usr/bin/env python3
"""
Verify that parameter combinations are generated correctly.
Run this locally before submitting to cluster.
"""
import json
import itertools

def generate_combinations(param_grid):
    """Generate all parameter combinations"""
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
            
            if ranker in conditional_params:
                params_to_vary = conditional_params[ranker]
                
                param_values = []
                param_names = []
                
                for param_name in params_to_vary:
                    param_key = f'Ranker.{param_name}'
                    if param_key in grid:
                        param_values.append(grid[param_key])
                        param_names.append(param_key)
                
                for values in itertools.product(*param_values):
                    combo = base_combo.copy()
                    for param_name, value in zip(param_names, values):
                        combo[param_name] = value
                    combinations.append(combo)
            else:
                combinations.append(base_combo)
    
    return combinations


# Load param_grid
with open('param_grid.json', 'r') as f:
    param_grid = json.load(f)

# Generate combinations
combinations = generate_combinations(param_grid)

# Print summary
print(f"Total combinations: {len(combinations)}")
print(f"\nExpected: {len(param_grid['grid']['OD.epsilon'])} epsilons × {len(param_grid['grid']['Ranker.alpha'])} alphas = {len(param_grid['grid']['OD.epsilon']) * len(param_grid['grid']['Ranker.alpha'])}")
print(f"Generated: {len(combinations)}")
print()

# Show first and last few
print("First 5 combinations:")
for i, combo in enumerate(combinations[:5]):
    print(f"  {i}: eps={combo['OD.epsilon']:.4f}, rule={combo['Ranker.rule']}, alpha={combo.get('Ranker.alpha', 'N/A')}")

print("\nLast 5 combinations:")
for i, combo in enumerate(combinations[-5:], start=len(combinations)-5):
    print(f"  {i}: eps={combo['OD.epsilon']:.4f}, rule={combo['Ranker.rule']}, alpha={combo.get('Ranker.alpha', 'N/A')}")

print(f"\n✓ Array should be: #SBATCH --array=0-{len(combinations)-1}")