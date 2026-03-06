import numpy as np


def initialize_ranker(G, info):
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)


def get_valid_posts(G, user_i, post_seen_gen):
    na = G["neighbor_authors"][user_i]
    nt = G["neighbor_times"][user_i]

    if len(na) == 0:
        return None, None, 0

    is_unseen = post_seen_gen[na, nt] < G['post_write_gen'][nt]
    valid_authors = na[is_unseen]
    valid_times = nt[is_unseen]
    n_available = len(valid_authors)

    return valid_authors, valid_times, n_available


def topk_closest(distances, n_available, n_selected):
    if n_available <= n_selected:
        return np.arange(n_available)
    return np.argpartition(distances, n_selected)[:n_selected]


def weighted_sample(weights, n_available, n_selected):
    if n_available <= n_selected:
        return np.arange(n_available)
    scores = np.random.uniform(size=n_available) ** (1.0 / weights)
    return np.argpartition(scores, -n_selected)[-n_selected:]

"""
Person inventing this algorithm. Used for Engagement and user success
Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random sampling with a reservoir. Information Processing Letters, 97(5), 181–185.
"Posts are sampled without replacement using weighted reservoir sampling (Efraimidis & Spirakis, 2006)"
"""