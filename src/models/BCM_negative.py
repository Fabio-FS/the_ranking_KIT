import numpy as np
from src.models.model_utilities import (
    initialize_tracking, initialize_buffers,
    update_likes, mark_seen, advance_time,
    compute_filter_bubble, check_convergence
)

# Negative influence BCM (DW variant) — assimilation-contrast model.
# Agents move toward posts within epsilon_1 (assimilation),
# and away from posts beyond epsilon_2 (repulsion/contrast).
# Posts in the dead zone [epsilon_1, epsilon_2] have no effect.
# References: Jager & Amblard (2005); Deffuant et al. (2002);
#             psychological grounding: Sherif & Hovland (1961) Social Judgment Theory.


def initialize(G, info):
    G.vs['opinion'] = np.random.rand(G.vcount())
    G['epsilon'] = info["OD"].get('epsilon_1', 0.2)
    G['epsilon2'] = info["OD"].get('epsilon_2', 0.6)
    G['mu'] = info["OD"].get('mu', 0.1)
    n_users = G.vcount()
    initialize_tracking(G, info, n_users)
    return initialize_buffers(G, info, n_users)


def _read_and_evaluate_posts(G, n_users, post_slot, selected_authors, selected_times,
                              post_opinions, post_seen_gen, post_likes, current_opinions):
    authors = selected_authors[:, post_slot]
    times = selected_times[:, post_slot]
    valid = authors >= 0

    post_ops = np.where(valid, post_opinions[authors, times], current_opinions)
    opinion_diff = post_ops - current_opinions
    abs_diff = np.abs(opinion_diff)

    assimilate = (abs_diff < G['epsilon']) & valid
    repulse    = (abs_diff > G['epsilon2']) & valid

    slot_abs_diff = np.sum(abs_diff[valid])
    slot_count = np.sum(valid)

    if np.any(assimilate):
        update_likes(G, assimilate, authors, times, post_likes, n_users)

    current_opinions += G['mu'] * opinion_diff * assimilate
    current_opinions -= G['mu'] * opinion_diff * repulse
    np.clip(current_opinions, 0.0, 1.0, out=current_opinions)

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