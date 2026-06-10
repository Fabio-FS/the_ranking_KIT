import numpy as np
from src.models.model_utilities import (
    initialize_tracking, initialize_buffers,
    update_likes, mark_seen, advance_time,
    compute_filter_bubble, check_convergence
)

# Dynamic-mu BCM variant.
# Each agent maintains an EWMA estimate of the mean and variance of posts they've recently seen.
# Their per-step plasticity mu_i is an increasing function of that recent-input variance:
#   coherent feed -> low variance -> low mu_i -> certain agent, updates little
#   chaotic feed  -> high variance -> high mu_i -> uncertain agent, updates a lot
# Likes are still gated by epsilon (engagement signal for rankers),
# but opinion updates happen on every valid post seen, with strength mu_i.


def initialize(G, info):
    n_users = G.vcount()
    G.vs['opinion'] = np.random.rand(n_users)

    # Like threshold (engagement signal only; does NOT gate opinion updates).
    G['epsilon'] = info["OD"].get('epsilon', 0.2)

    # Dynamic-mu parameters.
    G['mu_min']  = info["OD"].get('mu_min',  0.01)
    G['mu_max']  = info["OD"].get('mu_max',  0.2)
    G['var_ref'] = info["OD"].get('var_ref', 0.05)
    G['alpha']   = info["OD"].get('alpha',   0.1)

    # Per-agent EWMA state.
    # mean_i starts at the agent's own initial opinion ("the world looks like me").
    # var_i starts at var_ref so initial mu_i = (mu_min + mu_max) / 2 for every agent.
    G.vs['ewma_mean'] = np.array(G.vs['opinion'])
    G.vs['ewma_var']  = np.full(n_users, G['var_ref'])
    G.vs['mu']        = _compute_mu_from_var(np.full(n_users, G['var_ref']),
                                             G['mu_min'], G['mu_max'], G['var_ref'])

    initialize_tracking(G, info, n_users)
    return initialize_buffers(G, info, n_users)


def _compute_mu_from_var(var_arr, mu_min, mu_max, var_ref):
    # var_i / (var_i + var_ref) maps [0, inf) -> [0, 1).
    # mu_i = mu_min when var_i = 0, approaches mu_max as var_i grows.
    return mu_min + (mu_max - mu_min) * var_arr / (var_arr + var_ref)


def _update_ewma_and_mu(G, valid, post_ops):
    # O(1)-per-agent update of EWMA mean and variance, then recompute mu_i.
    # Only agents with a valid post in this slot get updated.
    alpha = G['alpha']
    ewma_mean = np.array(G.vs['ewma_mean'])
    ewma_var  = np.array(G.vs['ewma_var'])

    new_mean = (1 - alpha) * ewma_mean + alpha * post_ops
    deviation = post_ops - new_mean
    new_var  = (1 - alpha) * ewma_var + alpha * deviation * deviation

    # Only overwrite for agents who actually saw a post this slot.
    ewma_mean = np.where(valid, new_mean, ewma_mean)
    ewma_var  = np.where(valid, new_var,  ewma_var)

    G.vs['ewma_mean'] = ewma_mean
    G.vs['ewma_var']  = ewma_var
    G.vs['mu']        = _compute_mu_from_var(ewma_var, G['mu_min'], G['mu_max'], G['var_ref'])


def _read_and_evaluate_posts(G, n_users, post_slot, selected_authors, selected_times,
                              post_opinions, post_seen_gen, post_likes, current_opinions):
    authors = selected_authors[:, post_slot]
    times = selected_times[:, post_slot]
    valid = authors >= 0

    post_ops = np.where(valid, post_opinions[authors, times], current_opinions)
    opinion_diff = post_ops - current_opinions
    abs_diff = np.abs(opinion_diff)
    within_epsilon = abs_diff < G['epsilon']
    like_mask = within_epsilon & valid

    slot_abs_diff = np.sum(abs_diff[valid])
    slot_count = np.sum(valid)

    # Likes are emitted for within-epsilon posts (engagement signal for rankers).
    if np.any(like_mask):
        update_likes(G, like_mask, authors, times, post_likes, n_users)

    # Update EWMA and recompute mu_i BEFORE applying the opinion update,
    # so the post the agent just saw shapes how strongly they respond to it.
    _update_ewma_and_mu(G, valid, post_ops)
    mu_arr = np.array(G.vs['mu'])

    # Opinion update: every VALID post moves the agent (no epsilon gate),
    # with per-agent strength mu_i.
    current_opinions += mu_arr * opinion_diff * valid

    if np.any(valid):
        mark_seen(G, valid, times, post_seen_gen, n_users)

    return slot_abs_diff, slot_count


def update(G, info, selected_posts, post_opinions, post_likes, post_seen_gen):
    selected_authors, selected_times = selected_posts
    n_users = G.vcount()
    k = selected_authors.shape[1]
    current_opinions = np.array(G.vs['opinion'])

    total_abs_diff = 0.0
    total_count = 0

    for post_slot in range(k):
        slot_abs_diff, slot_count = _read_and_evaluate_posts(
            G, n_users, post_slot, selected_authors, selected_times,
            post_opinions, post_seen_gen, post_likes, current_opinions)
        total_abs_diff += slot_abs_diff
        total_count += slot_count

    G.vs['opinion'] = current_opinions
    advance_time(G, post_opinions, post_likes, current_opinions)
    return compute_filter_bubble(total_abs_diff, total_count)