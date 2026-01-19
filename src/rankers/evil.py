import numpy as np
import igraph as ig


def initialize(G, info):
    """
    Initialize the evil ranker.
    
    This ranker combines algorithmic bias with knowledge of user behavior:
    it knows users only engage with posts within epsilon, so it filters to those,
    then strategically shows posts that pull users toward a target narrative.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary with target_opinion parameter
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)
    G["target_opinion"] = info["Ranker"].get("target_opinion", 0.5)


def rank(G, info, post_opinions, post_likes, post_seen_by):
    """
    Evil ranker: filter to posts within epsilon, then show ones closest to target.
    
    This models a sophisticated recommendation system that:
    1. Knows user behavior (only shows posts within epsilon to ensure engagement)
    2. Has algorithmic bias (strategically selects which posts to maximize narrative push)
    
    The ranker is "evil" because it exploits knowledge of user psychology to
    manipulate opinions toward a target while maintaining high engagement.
    
    Args:
        G: igraph Graph with neighbor_matrix and epsilon precomputed
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
    epsilon = G['epsilon']
    target = G["target_opinion"]
    
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
        
        # Filter to posts within epsilon (will get engagement)
        within_epsilon = np.abs(unseen_opinions - current_opinions[user_i]) < epsilon
        epsilon_idx = valid_idx[within_epsilon]
        n_within_epsilon = len(epsilon_idx)
        
        if n_within_epsilon == 0:
            continue
        
        # Among posts within epsilon, select those closest to target narrative
        epsilon_opinions = post_opinions[all_authors[epsilon_idx], all_times[epsilon_idx]]
        distances_to_target = np.abs(epsilon_opinions - target)
        
        n_selected = min(k, n_within_epsilon)
        closest_to_target = np.argsort(distances_to_target)[:n_selected]
        
        selected_authors[user_i, :n_selected] = all_authors[epsilon_idx[closest_to_target]]
        selected_times[user_i, :n_selected] = all_times[epsilon_idx[closest_to_target]]
    
    return selected_authors, selected_times