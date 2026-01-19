import numpy as np
import igraph as ig


def initialize(G, info):
    """
    Initialize the maximize engagement ranker.
    
    This ranker has learned user behavior: it knows users only engage with
    posts within epsilon of their opinion (for BCM). It maximizes engagement
    by only showing posts users are likely to interact with.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)


def rank(G, info, post_opinions, post_likes, post_seen_by):
    """
    Maximize engagement ranker: show only posts within epsilon, sample uniformly.
    
    This is the "smart algorithm" that has perfect knowledge of user behavior.
    It knows users only engage with posts where |post_opinion - user_opinion| < epsilon,
    so it filters to only those posts. If fewer than k posts satisfy this,
    it fills remaining slots with random unseen neighbor posts.
    
    This models realistic recommendation systems that optimize for engagement
    by learning which content users interact with.
    
    Args:
        G: igraph Graph with neighbor_matrix and epsilon parameter
        info: Configuration dict
        post_opinions: Array (n_users, history_size) of post opinion values
        post_likes: Array (n_users, history_size) of like counts per post
        post_seen_by: Array (n_users, history_size, n_users) tracking which users saw which posts
        
    Returns:
        tuple: (selected_authors, selected_times) where each is shape (n_users, k)
    """
    n_users = G['N']
    k = G["k_posts"]
    history_size = G['post_history']
    epsilon = G['epsilon']
    
    selected_authors = np.full((n_users, k), -1, dtype=np.int32)
    selected_times = np.full((n_users, k), -1, dtype=np.int32)
    
    all_authors = np.repeat(np.arange(n_users), history_size)
    all_times = np.tile(np.arange(history_size), n_users)
    
    is_neighbor = G['neighbor_matrix'][:, all_authors]
    is_seen = post_seen_by[all_authors, all_times, :].T
    valid_posts = is_neighbor & ~is_seen
    
    current_opinions = np.array(G.vs['opinion'])
    
    for user_i in range(n_users):
        valid_idx = np.flatnonzero(valid_posts[user_i])
        n_available = len(valid_idx)
        
        if n_available == 0:
            continue
        
        unseen_opinions = post_opinions[all_authors[valid_idx], all_times[valid_idx]]
        within_epsilon = np.abs(unseen_opinions - current_opinions[user_i]) < epsilon
        
        epsilon_idx = valid_idx[within_epsilon]
        n_within_epsilon = len(epsilon_idx)
        
        if n_within_epsilon >= k:
            if n_within_epsilon == k:
                selected_idx = epsilon_idx
            else:
                random_order = np.random.permutation(n_within_epsilon)[:k]
                selected_idx = epsilon_idx[random_order]
            n_selected = k
        else:
            selected_idx = np.zeros(k, dtype=np.int32)
            selected_idx[:n_within_epsilon] = epsilon_idx
            
            remaining = k - n_within_epsilon
            outside_idx = valid_idx[~within_epsilon]
            n_outside = len(outside_idx)
            
            if n_outside > 0:
                n_fill = min(remaining, n_outside)
                if n_outside <= n_fill:
                    selected_idx[n_within_epsilon:n_within_epsilon+n_fill] = outside_idx
                else:
                    random_order = np.random.permutation(n_outside)[:n_fill]
                    selected_idx[n_within_epsilon:n_within_epsilon+n_fill] = outside_idx[random_order]
                n_selected = n_within_epsilon + n_fill
            else:
                n_selected = n_within_epsilon
        
        selected_authors[user_i, :n_selected] = all_authors[selected_idx[:n_selected]]
        selected_times[user_i, :n_selected] = all_times[selected_idx[:n_selected]]
    
    return selected_authors, selected_times