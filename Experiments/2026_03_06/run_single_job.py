import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
from itertools import product
from src.simulation import simulate, run_replicas
from src.analysis import save_results

def build_info(job, noise, k_posts, **extra_od):
    od = {'model': job['model'], 'mu': job['mu'], **extra_od}
    return {
        "General":            {"seed": 42},
        "Graph":              {"type": "ER", "n": 1000, "p": 20/999},
        "OD":                 od,
        "Ranker":             job['ranker'],
        "Simulation_details": {"n_steps": 2000, "convergence_window": 50, "convergence_delta": 1e-4},
        "k_posts":            k_posts,
        "post_history":       50,
        "noise":              noise,
    }

def result_filename(experiment_dir, job, noise, k_posts, **extra_od):
    from src.analysis import save_results
    ranker = job['ranker']
    rule = ranker['rule']
    if rule == 'Narrative':
        ranker_str = f"Narrative_t{ranker['target_opinion']}"
    elif rule in ('Engagement', 'User_Success', 'Personalization'):
        ranker_str = f"{rule}_a{ranker['alpha']}"
    else:
        ranker_str = rule

    od_str = "_".join(f"{k}{v}" for k, v in extra_od.items())
    return os.path.join(experiment_dir, 'results',
                        f"{job['model']}_{ranker_str}_{od_str}_noise{noise}_k{k_posts}")


task_id      = int(sys.argv[1])
experiment_dir = sys.argv[2]

with open(f'{experiment_dir}/job_configs/job_{task_id:04d}.json') as f:
    job = json.load(f)

os.makedirs(os.path.join(experiment_dir, 'results'), exist_ok=True)

model = job['model']

if model in ('BCM', 'BCM_HK'):
    epsilon = job['epsilon']
    for noise, k in product(job['noise_values'], job['k_values']):
        fname = result_filename(experiment_dir, job, noise, k, epsilon=epsilon)
        info  = build_info(job, noise, k, epsilon=epsilon)
        results = run_replicas(info, n_replicas=job['n_replicas'])
        save_results(results, fname)

elif model == 'BCM_negative':
    epsilon_1 = job['epsilon_1']
    for epsilon_2, noise, k in product(job['epsilon_2_values'], job['noise_values'], job['k_values']):
        fname = result_filename(experiment_dir, job, noise, k, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
        info  = build_info(job, noise, k, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
        results = run_replicas(info, n_replicas=job['n_replicas'])
        save_results(results, fname)

elif model == 'BCM_asymmetric':
    epsilon = job['epsilon']
    for mu_2, noise, k in product(job['mu_2_values'], job['noise_values'], job['k_values']):
        fname = result_filename(experiment_dir, job, noise, k, epsilon=epsilon, mu_2=mu_2)
        info  = build_info(job, noise, k, epsilon=epsilon, mu_2=mu_2)
        results = run_replicas(info, n_replicas=job['n_replicas'])
        save_results(results, fname)