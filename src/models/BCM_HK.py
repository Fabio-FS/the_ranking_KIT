import numpy as np
from src.models.model_utilities import (
    initialize_tracking, initialize_buffers,
    update_likes, mark_seen, advance_time,
    compute_filter_bubble, check_convergence
)

# HK (Hegselmann-Krause) variant of BCM.
# Agents update once toward the average opinion of all within-epsilon posts, rather than sequentially post by post.


def initialize(G, info):
    G.vs['opinion'] = np.random.rand(G.vcount())
    G['epsilon'] = info["OD"].get('epsilon', 0.2)
    G['mu'] = info["OD"].get('mu', 0.1)
    n_users = G.vcount()
    initialize_tracking(G, info, n_users)
    return initialize_buffers(G, info, n_users)


def update(G, info, selected_posts, post_opinions, post_likes, post_seen_gen):
    selected_authors, selected_times = selected_posts
    n_users = G.vcount()
    k = selected_authors.shape[1]
    current_opinions = np.array(G.vs['opinion'])
    epsilon = G['epsilon']
    mu = G['mu']

    opinion_sum = np.zeros(n_users)
    within_epsilon_count = np.zeros(n_users, dtype=np.int32)
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

        opinion_sum += np.where(like_mask, post_ops, 0.0)
        within_epsilon_count += like_mask.astype(np.int32)

        if np.any(like_mask):
            update_likes(G, like_mask, authors, times, post_likes, n_users)

        if np.any(valid):
            mark_seen(G, valid, times, post_seen_gen, n_users)

    has_influence = within_epsilon_count > 0
    avg_opinion = np.where(has_influence, opinion_sum / np.maximum(within_epsilon_count, 1), current_opinions)
    current_opinions += mu * (avg_opinion - current_opinions) * has_influence

    G.vs['opinion'] = current_opinions
    advance_time(G, post_opinions, post_likes, current_opinions)
    return compute_filter_bubble(total_abs_diff, total_count)