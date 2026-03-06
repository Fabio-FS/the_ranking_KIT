import numpy as np
from src.rankers.ranker_utilities import initialize_ranker, get_valid_posts


def initialize(G, info):
    initialize_ranker(G, info)


def rank(G, info, post_opinions, post_likes, post_seen_gen):
    n_users = G['N']
    k = G["k_posts"]

    selected_authors = np.full((n_users, k), -1, dtype=np.int32)
    selected_times = np.full((n_users, k), -1, dtype=np.int32)

    for user_i in range(n_users):
        valid_authors, valid_times, n_available = get_valid_posts(G, user_i, post_seen_gen)
        if n_available == 0:
            continue

        if n_available <= k:
            selected_authors[user_i, :n_available] = valid_authors
            selected_times[user_i, :n_available] = valid_times
        else:
            chosen = np.random.permutation(n_available)[:k]
            selected_authors[user_i, :k] = valid_authors[chosen]
            selected_times[user_i, :k] = valid_times[chosen]

    return selected_authors, selected_times