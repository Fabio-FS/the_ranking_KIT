from time import time

import igraph as ig
import numpy as np
from src.graphs import build_graph
from src.metrics import compute_timestep_metrics_light, compute_homophily
import src.models.BCM
import src.models.BCM_HK
import src.models.BCM_asymmetric
import src.models.BCM_negative

import src.rankers.random, src.rankers.closest, src.rankers.diverse_engagement
import src.rankers.engagement, src.rankers.narrative, src.rankers.evil
import src.rankers.user_success, src.rankers.personalization
from src.rankers.ranker_utilities import get_valid_posts


INIT_OD = {
    'BCM':           src.models.BCM.initialize,
    'BCM_HK':        src.models.BCM_HK.initialize,
    'BCM_asymmetric': src.models.BCM_asymmetric.initialize,
    'BCM_negative':  src.models.BCM_negative.initialize,
}
UPDATE = {
    'BCM':           src.models.BCM.update,
    'BCM_HK':        src.models.BCM_HK.update,
    'BCM_asymmetric': src.models.BCM_asymmetric.update,
    'BCM_negative':  src.models.BCM_negative.update,
}
CHECK_CONVERGENCE = {
    'BCM':           src.models.BCM.check_convergence,
    'BCM_HK':        src.models.BCM_HK.check_convergence,
    'BCM_asymmetric': src.models.BCM_asymmetric.check_convergence,
    'BCM_negative':  src.models.BCM_negative.check_convergence,
}

INIT_RANKER = {
    'Random': src.rankers.random.initialize,
    'Closest': src.rankers.closest.initialize,
    'Diverse_Engagement': src.rankers.diverse_engagement.initialize,
    'Engagement': src.rankers.engagement.initialize,
    'Narrative': src.rankers.narrative.initialize,
    "Evil": src.rankers.evil.initialize,
    'User_Success': src.rankers.user_success.initialize,
    'Personalization': src.rankers.personalization.initialize,
}
RANKER = {
    'Random': src.rankers.random.rank,
    'Closest': src.rankers.closest.rank,
    'Diverse_Engagement': src.rankers.diverse_engagement.rank,
    'Engagement': src.rankers.engagement.rank,
    'Narrative': src.rankers.narrative.rank,
    "Evil": src.rankers.evil.rank,
    'User_Success': src.rankers.user_success.rank,
    'Personalization': src.rankers.personalization.rank,
}
OPTIMIZER = False


def compute_exp_checkpoints(n_steps, n_checkpoints=200):
    return np.unique(np.geomspace(1, n_steps, n_checkpoints).astype(int)) - 1


def subsample_at_checkpoints(array, convergence_step, checkpoints):
    result = np.zeros(len(checkpoints))
    last_value = array[convergence_step]
    for i, cp in enumerate(checkpoints):
        result[i] = array[cp] if cp <= convergence_step else last_value
    return result

def apply_noise(selected_posts, noise, G, post_seen_gen):
    if noise == 0.0:
        return selected_posts

    selected_authors, selected_times = selected_posts
    n_users, k = selected_authors.shape

    for user_i in range(n_users):
        for slot in range(k):
            if selected_authors[user_i, slot] == -1:
                continue
            if np.random.rand() < noise:
                valid_authors, valid_times, n_available = get_valid_posts(G, user_i, post_seen_gen)
                if n_available == 0:
                    continue
                idx = np.random.randint(n_available)
                selected_authors[user_i, slot] = valid_authors[idx]
                selected_times[user_i, slot] = valid_times[idx]

    return selected_authors, selected_times

def simulate(info, seed=42, homophily_steps=None):
    """
    homophily_steps: set of step indices where homophily is computed.
                     If None, compute at every step.
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    if OPTIMIZER:
        import time
        tic = time.time()

    G = build_graph(info)

    if OPTIMIZER:
        print("Graph built in {:.2f} seconds".format(time.time() - tic))
        tic = time.time()

    init_fn        = INIT_OD[info['OD']["model"]]
    init_ranker_fn = INIT_RANKER[info['Ranker']["rule"]]
    update_fn      = UPDATE[info['OD']["model"]]
    ranker_fn      = RANKER[info['Ranker']["rule"]]
    convergence_fn = CHECK_CONVERGENCE[info['OD']["model"]]
    noise = info['Ranker'].get('noise', 0.0)

    post_opinions, post_likes, post_seen_by = init_fn(G, info)
    init_ranker_fn(G, info)

    if OPTIMIZER:
        print("Initialization completed in {:.2f} seconds".format(time.time() - tic))
        tic = time.time()

    n_steps  = info["Simulation_details"].get('n_steps', 100)
    W        = info["Simulation_details"].get('convergence_window', 50)
    delta    = info["Simulation_details"].get('convergence_delta', 1e-4)
    n_users  = G.vcount()

    RES_pol          = np.zeros(n_steps)
    RES_filter_bubble = np.zeros(n_steps)
    RES_homophily    = np.zeros(n_steps)
    RES_opinions     = np.zeros((n_steps, n_users))

    convergence_step = n_steps - 1

    if OPTIMIZER:
        print("Starting simulation loop... at time {:.2f} seconds".format(time.time() - tic))
        tic = time.time()

    for step in range(n_steps):
        G['current_step'] = step

        selected_posts = ranker_fn(G, info, post_opinions, post_likes, post_seen_by)
        selected_posts = apply_noise(selected_posts, noise, G, post_seen_by)
        fb = update_fn(G, info, selected_posts, post_opinions, post_likes, post_seen_by)

        #selected_posts = ranker_fn(G, info, post_opinions, post_likes, post_seen_by)
        #fb = update_fn(G, info, selected_posts, post_opinions, post_likes, post_seen_by)

        current_opinions = np.array(G.vs['opinion'])

        RES_pol[step]          = np.var(current_opinions)
        RES_filter_bubble[step] = fb
        RES_opinions[step]     = current_opinions

        if homophily_steps is None or step in homophily_steps:
            RES_homophily[step] = compute_homophily(G)
        else:
            RES_homophily[step] = RES_homophily[step - 1]

        if step > 2 * W and convergence_fn(RES_pol, step, W, delta):
            convergence_step = step
            break

        if OPTIMIZER:
            if step % 10 == 0:
                print(f"Step {step}/{n_steps} completed in {time.time() - tic:.2f} seconds")
                tic = time.time()

    RES_pol[convergence_step + 1:]           = RES_pol[convergence_step]
    RES_filter_bubble[convergence_step + 1:] = RES_filter_bubble[convergence_step]
    RES_homophily[convergence_step + 1:]     = RES_homophily[convergence_step]
    RES_opinions[convergence_step + 1:]      = RES_opinions[convergence_step]

    n_users = G.vcount()
    final_likes = G['agent_cumulative_likes'].copy() if G['track_cumulative_likes'] else np.zeros(n_users)

    return {
        'G':               G,
        'pol':             RES_pol,
        'filter_bubble':   RES_filter_bubble,
        'homophily':       RES_homophily,
        'opinions':        RES_opinions,
        'convergence_step': convergence_step,
        'final_opinions':  np.array(G.vs['opinion']),
        'final_likes':     final_likes,
    }


def run_replicas(info, n_replicas=100):

    seed    = info["General"].get("seed", 42)
    n_steps = info["Simulation_details"].get('n_steps', 100)
    n_users = info["Graph"]["n"]

    checkpoints    = compute_exp_checkpoints(n_steps)
    homophily_steps = set(checkpoints)
    n_checkpoints  = len(checkpoints)

    all_pol              = np.zeros((n_replicas, n_checkpoints))
    all_filter_bubble    = np.zeros((n_replicas, n_checkpoints))
    all_homophily        = np.zeros((n_replicas, n_checkpoints))
    all_convergence_steps = np.zeros(n_replicas, dtype=np.int32)
    all_final_opinions   = np.zeros((n_replicas, n_users))
    all_final_likes      = np.zeros((n_replicas, n_users))

    full_pol_rep0      = None
    full_opinions_rep0 = None

    print(f"Running {n_replicas} replicas...")
    for rep in range(n_replicas):
        seed_rep = seed + rep
        print(f"  Replica {rep+1}/{n_replicas}")

        result = simulate(info, seed=seed_rep, homophily_steps=homophily_steps)
        conv   = result['convergence_step']

        all_convergence_steps[rep] = conv
        all_pol[rep]           = subsample_at_checkpoints(result['pol'], conv, checkpoints)
        all_filter_bubble[rep] = subsample_at_checkpoints(result['filter_bubble'], conv, checkpoints)
        all_homophily[rep]     = subsample_at_checkpoints(result['homophily'], conv, checkpoints)
        all_final_opinions[rep] = result['final_opinions']
        all_final_likes[rep]   = result['final_likes']

        if rep == 0:
            full_pol_rep0      = result['pol'][:conv + 1]
            full_opinions_rep0 = result['opinions'][:conv + 1]

    print(f"Completed {n_replicas} replicas!")
    print(f"Convergence steps: min={all_convergence_steps.min()}, max={all_convergence_steps.max()}, mean={all_convergence_steps.mean():.0f}")

    return {
        'info':             info,
        'n_replicas':       n_replicas,
        'checkpoints':      checkpoints,
        'pol':              all_pol,
        'filter_bubble':    all_filter_bubble,
        'homophily':        all_homophily,
        'convergence_steps': all_convergence_steps,
        'final_opinions':   all_final_opinions,
        'final_likes':      all_final_likes,
        'pol_rep0':         full_pol_rep0,
        'opinions_rep0':    full_opinions_rep0,
    }