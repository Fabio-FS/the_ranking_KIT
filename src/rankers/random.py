import numpy as np
import igraph as ig



def initialize(G, info):
    """
    Initialize the random ranker by storing graph metadata.
    
    This ranker doesn't need complex initialization - just stores
    the number of users for convenience in the rank function.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)



def rank(G, info, post_opinions, post_likes, post_seen_by):
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
    
    for user_i in range(n_users):
        valid_idx = np.flatnonzero(valid_posts[user_i])
        n_available = len(valid_idx)
        
        if n_available == 0:
            continue
        
        if n_available <= k:
            selected_idx = valid_idx
        else:
            random_order = np.random.permutation(n_available)[:k]
            selected_idx = valid_idx[random_order]
        
        n_selected = len(selected_idx)
        selected_authors[user_i, :n_selected] = all_authors[selected_idx]
        selected_times[user_i, :n_selected] = all_times[selected_idx]
    
    return selected_authors, selected_times