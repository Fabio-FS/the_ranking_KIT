import numpy as np
from src.rankers.ranker_utilities import initialize_ranker, get_valid_posts, weighted_sample


def initialize(G, info):
    initialize_ranker(G, info)
    G["alpha"] = info["Ranker"].get("alpha", 1.0)


def rank(G, info, post_opinions, post_likes, post_seen_gen):
    n_users = G['N']
    k = G["k_posts"]
    alpha = G["alpha"]
    user_author_likes = G['user_author_likes']

    selected_authors = np.full((n_users, k), -1, dtype=np.int32)
    selected_times = np.full((n_users, k), -1, dtype=np.int32)

    for user_i in range(n_users):
        valid_authors, valid_times, n_available = get_valid_posts(G, user_i, post_seen_gen)
        if n_available == 0:
            continue

        weights = np.power(user_author_likes[user_i, valid_authors] + 1, alpha)
        n_selected = min(k, n_available)
        chosen = weighted_sample(weights, n_available, n_selected)
        selected_authors[user_i, :n_selected] = valid_authors[chosen]
        selected_times[user_i, :n_selected] = valid_times[chosen]

    return selected_authors, selected_times