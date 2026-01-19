import numpy as np
import igraph as ig


def initialize(G, info):
    """
    Initialize the engagement-based ranker.
    
    This ranker weights posts by their popularity: posts with more likes
    are more likely to be shown. The relationship is controlled by alpha parameter.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)
    G["alpha"] = info["Ranker"].get("alpha", 1.0)


def rank(G, info, post_opinions, post_likes, post_seen_by):
    """
    Engagement-based ranker: select posts with probability proportional to (likes + 1)^alpha.
    
    Posts with more likes are more likely to be shown. The alpha parameter controls
    how strongly engagement is weighted:
    - alpha = 0: uniform random (ignores likes)
    - alpha = 1: linear preference for popular posts
    - alpha > 1: winner-take-all dynamics (viral posts dominate)
    
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
    alpha = G["alpha"]
    
    selected_authors = np.full((n_users, k), -1, dtype=np.int32)
    selected_times = np.full((n_users, k), -1, dtype=np.int32)
    
    all_authors = np.repeat(np.arange(n_users), history_size)
    all_times = np.tile(np.arange(history_size), n_users)
    
    is_neighbor = G['neighbor_matrix'][:, all_authors]
    is_seen = post_seen_by[all_authors, all_times, :].T
    valid_posts = is_neighbor & ~is_seen
    
    for user_i in range(n_users):
        valid_idx = np.flatnonzero(valid_posts[user_i])
        n_available = len(valid_idx)
        
        if n_available == 0:
            continue
        
        unseen_likes = post_likes[all_authors[valid_idx], all_times[valid_idx]]
        weights = np.power(unseen_likes + 1, alpha)
        probabilities = weights / np.sum(weights)
        
        n_selected = min(k, n_available)
        selected_indices = np.random.choice(n_available, size=n_selected, replace=False, p=probabilities)
        
        selected_authors[user_i, :n_selected] = all_authors[valid_idx[selected_indices]]
        selected_times[user_i, :n_selected] = all_times[valid_idx[selected_indices]]
    
    return selected_authors, selected_times