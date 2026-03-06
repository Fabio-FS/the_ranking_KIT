import json
import os
from itertools import product

RANKER_CONFIGS = [
    {'rule': 'Random'},
    {'rule': 'Closest'},
    {'rule': 'Narrative',       'target_opinion': 0.0},
    {'rule': 'Narrative',       'target_opinion': 0.25},
    {'rule': 'Narrative',       'target_opinion': 0.5},
    {'rule': 'Engagement',      'alpha': 0.5},
    {'rule': 'Engagement',      'alpha': 1.0},
    {'rule': 'Engagement',      'alpha': 2.0},
    {'rule': 'User_Success',    'alpha': 0.5},
    {'rule': 'User_Success',    'alpha': 1.0},
    {'rule': 'User_Success',    'alpha': 2.0},
    {'rule': 'Personalization', 'alpha': 0.5},
    {'rule': 'Personalization', 'alpha': 1.0},
    {'rule': 'Personalization', 'alpha': 2.0},
]

EPSILONS     = [0.1, 0.2, 0.3, 0.4]
NOISE_VALUES = [0.0, 0.1]
KPOST_VALUES = [1, 5, 10]
N_REPLICAS   = 100
MU           = 0.2

def ranker_label(r):
    if r['rule'] == 'Narrative':
        return f"Narrative_t{r['target_opinion']}"
    if r['rule'] in ('Engagement', 'User_Success', 'Personalization'):
        return f"{r['rule']}_a{r['alpha']}"
    return r['rule']

jobs = []

# BCM and BCM_HK: one job per (model, ranker, epsilon)
for model, ranker, epsilon in product(['BCM', 'BCM_HK'], RANKER_CONFIGS, EPSILONS):
    jobs.append({
        'model': model,
        'ranker': ranker,
        'epsilon': epsilon,
        'mu': MU,
        'noise_values': NOISE_VALUES,
        'k_values': KPOST_VALUES,
        'n_replicas': N_REPLICAS,
    })

# BCM_negative: one job per (ranker, epsilon_1)
EPSILON2_VALUES = [0.5, 0.7]
MU2_VALUES      = [0.1, 0.05]

for ranker, epsilon_1 in product(RANKER_CONFIGS, EPSILONS):
    jobs.append({
        'model': 'BCM_negative',
        'ranker': ranker,
        'epsilon_1': epsilon_1,
        'epsilon_2_values': EPSILON2_VALUES,
        'mu': MU,
        'noise_values': NOISE_VALUES,
        'k_values': KPOST_VALUES,
        'n_replicas': N_REPLICAS,
    })

# BCM_asymmetric: one job per (ranker, epsilon)
for ranker, epsilon in product(RANKER_CONFIGS, EPSILONS):
    jobs.append({
        'model': 'BCM_asymmetric',
        'ranker': ranker,
        'epsilon': epsilon,
        'mu_1': MU,
        'mu_2_values': MU2_VALUES,
        'noise_values': NOISE_VALUES,
        'k_values': KPOST_VALUES,
        'n_replicas': N_REPLICAS,
    })

os.makedirs('job_configs', exist_ok=True)
for i, job in enumerate(jobs):
    with open(f'job_configs/job_{i:04d}.json', 'w') as f:
        json.dump(job, f, indent=2)

print(f"Generated {len(jobs)} job configs")