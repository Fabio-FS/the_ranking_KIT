import numpy as np
from src.rankers.ranker_utilities import initialize_ranker, get_valid_posts


def initialize(G, info):
    initialize_ranker(G, info)


def rank(G, info, post_opinions, post_likes, post_seen_gen):
    n_users = G['N']
    k = G["k_posts"]
    epsilon = G['epsilon']
    current_opinions = np.array(G.vs['opinion'])

    selected_authors = np.full((n_users, k), -1, dtype=np.int32)
    selected_times = np.full((n_users, k), -1, dtype=np.int32)

    for user_i in range(n_users):
        valid_authors, valid_times, n_available = get_valid_posts(G, user_i, post_seen_gen)
        if n_available == 0:
            continue

        within_epsilon = np.abs(post_opinions[valid_authors, valid_times] - current_opinions[user_i]) < epsilon
        epsilon_authors = valid_authors[within_epsilon]
        epsilon_times = valid_times[within_epsilon]
        n_within_epsilon = len(epsilon_authors)

        if n_within_epsilon >= k:
            chosen = np.random.permutation(n_within_epsilon)[:k]
            selected_authors[user_i, :k] = epsilon_authors[chosen]
            selected_times[user_i, :k] = epsilon_times[chosen]
        else:
            selected_authors[user_i, :n_within_epsilon] = epsilon_authors
            selected_times[user_i, :n_within_epsilon] = epsilon_times

            outside_authors = valid_authors[~within_epsilon]
            outside_times = valid_times[~within_epsilon]
            n_outside = len(outside_authors)
            n_fill = min(k - n_within_epsilon, n_outside)

            if n_fill > 0:
                chosen = np.random.permutation(n_outside)[:n_fill]
                selected_authors[user_i, n_within_epsilon:n_within_epsilon + n_fill] = outside_authors[chosen]
                selected_times[user_i, n_within_epsilon:n_within_epsilon + n_fill] = outside_times[chosen]

    return selected_authors, selected_times