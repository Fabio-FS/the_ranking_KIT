import numpy as np
import igraph as ig


def initialize(G, info):
    """
    Initialize the closest opinion ranker.
    
    This ranker creates filter bubbles by showing users posts with opinions
    closest to their own, modeling echo chambers and opinion homophily.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)


def rank(G, info, post_opinions, post_likes, post_seen_by):
    """
    Closest opinion ranker: show posts with opinions nearest to user's opinion.
    
    This models filter bubbles and echo chambers where users primarily see
    content that aligns with their existing views. It selects the k posts
    with opinions closest to the user's current opinion.
    
    Args:
        G: igraph Graph with neighbor_matrix precomputed
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
        distances = np.abs(unseen_opinions - current_opinions[user_i])
        
        n_selected = min(k, n_available)
        closest_indices = np.argsort(distances)[:n_selected]
        
        selected_authors[user_i, :n_selected] = all_authors[valid_idx[closest_indices]]
        selected_times[user_i, :n_selected] = all_times[valid_idx[closest_indices]]
    
    return selected_authors, selected_times