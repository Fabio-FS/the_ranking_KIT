import numpy as np
import igraph as ig


def initialize(G, info):
    """
    Initialize the narrative pushing ranker.
    
    This ranker has algorithmic bias: it systematically shows posts closer
    to a target opinion value, modeling platforms that push a specific narrative.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary with target_opinion parameter
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)
    G["target_opinion"] = info["Ranker"].get("target_opinion", 0.5)


def rank(G, info, post_opinions, post_likes, post_seen_by):
    """
    Narrative pushing ranker: show posts closest to a target opinion value.
    
    This models algorithmic bias where the platform systematically promotes
    content aligned with a specific narrative or ideology. The target opinion
    could represent left-wing (e.g., 0.2), centrist (0.5), or right-wing (0.8) bias.
    
    Args:
        G: igraph Graph with neighbor_matrix precomputed
        info: Configuration dict with target_opinion
        post_opinions: Array (n_users, history_size) of post opinion values
        post_likes: Array (n_users, history_size) of like counts per post
        post_seen_by: Array (n_users, history_size, n_users) tracking which users saw which posts
        
    Returns:
        tuple: (selected_authors, selected_times) where each is shape (n_users, k)
    """
    n_users = G['N']
    k = G["k_posts"]
    history_size = G['post_history']
    target = G["target_opinion"]
    
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
        
        unseen_opinions = post_opinions[all_authors[valid_idx], all_times[valid_idx]]
        distances = np.abs(unseen_opinions - target)
        
        n_selected = min(k, n_available)
        closest_to_target = np.argsort(distances)[:n_selected]
        
        selected_authors[user_i, :n_selected] = all_authors[valid_idx[closest_to_target]]
        selected_times[user_i, :n_selected] = all_times[valid_idx[closest_to_target]]
    
    return selected_authors, selected_times