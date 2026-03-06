import numpy as np
from src.models.model_utilities import (
    initialize_tracking, initialize_buffers,
    update_likes, mark_seen, advance_time,
    compute_filter_bubble, check_convergence
)

# Asymmetric BCM (HK variant) — biased assimilation.
# Agents update toward the average of within-epsilon posts, but with different step sizes
# depending on whether the post is on the same side of 0.5 as the agent (mu_1)
# or the opposite side (mu_2). Typically mu_1 > mu_2.
# Reference: Dandekar, Goel & Lee (2013), PNAS.


def initialize(G, info):
    G.vs['opinion'] = np.random.rand(G.vcount())
    G['epsilon'] = info["OD"].get('epsilon', 0.2)
    G['mu_1'] = info["OD"].get('mu_1', 0.2)
    G['mu_2'] = info["OD"].get('mu_2', 0.05)
    n_users = G.vcount()
    initialize_tracking(G, info, n_users)
    return initialize_buffers(G, info, n_users)


def update(G, info, selected_posts, post_opinions, post_likes, post_seen_gen):
    selected_authors, selected_times = selected_posts
    n_users = G.vcount()
    k = selected_authors.shape[1]
    current_opinions = np.array(G.vs['opinion'])
    epsilon = G['epsilon']
    mu_1 = G['mu_1']
    mu_2 = G['mu_2']

    same_side_sum = np.zeros(n_users)
    same_side_count = np.zeros(n_users, dtype=np.int32)
    cross_side_sum = np.zeros(n_users)
    cross_side_count = np.zeros(n_users, dtype=np.int32)
    total_abs_diff = 0.0
    total_count = 0

    for post_slot in range(k):
        authors = selected_authors[:, post_slot]
        times = selected_times[:, post_slot]
        valid = authors >= 0

        post_ops = post_opinions[authors, times]
        abs_diff = np.abs(post_ops - current_opinions)
        within_epsilon = abs_diff < epsilon
        like_mask = within_epsilon & valid

        total_abs_diff += np.sum(abs_diff[valid])
        total_count += np.sum(valid)

        same_side = (current_opinions >= 0.5) == (post_ops >= 0.5)
        same_side_sum += np.where(like_mask & same_side, post_ops, 0.0)
        same_side_count += (like_mask & same_side).astype(np.int32)
        cross_side_sum += np.where(like_mask & ~same_side, post_ops, 0.0)
        cross_side_count += (like_mask & ~same_side).astype(np.int32)

        if np.any(like_mask):
            update_likes(G, like_mask, authors, times, post_likes, n_users)

        if np.any(valid):
            mark_seen(G, valid, times, post_seen_gen, n_users)

    has_same = same_side_count > 0
    has_cross = cross_side_count > 0
    avg_same = np.where(has_same, same_side_sum / np.maximum(same_side_count, 1), current_opinions)
    avg_cross = np.where(has_cross, cross_side_sum / np.maximum(cross_side_count, 1), current_opinions)
    current_opinions += mu_1 * (avg_same - current_opinions) * has_same
    current_opinions += mu_2 * (avg_cross - current_opinions) * has_cross

    G.vs['opinion'] = current_opinions
    advance_time(G, post_opinions, post_likes, current_opinions)
    return compute_filter_bubble(total_abs_diff, total_count)