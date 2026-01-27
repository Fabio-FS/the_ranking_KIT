# src/rankers/user_success.py
import numpy as np
import igraph as ig


def initialize(G, info):
    """
    Initialize the user success ranker.
    
    This ranker prioritizes posts from successful agents. Posts are weighted
    by their author's cumulative likes across all time, creating rich-get-richer
    dynamics at the agent level rather than post level.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary
    """
    G["N"] = G.vcount()
    G["k_posts"] = info.get("k_posts", 1)
    G["alpha"] = info["Ranker"].get("alpha", 1.0)


def rank(G, info, post_opinions, post_likes, post_seen_by):
    """
    User success ranker: select posts weighted by author's total cumulative likes.
    
    Posts are weighted by (author_cumulative_likes + 1)^alpha, where cumulative
    likes include all likes the author has ever received, not just on recent posts.
    This creates agent-level inequality where successful users dominate visibility.
    
    Args:
        G: igraph Graph with neighbor_matrix and agent_cumulative_likes
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
    
    agent_cumulative_likes = G['agent_cumulative_likes']
    
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
        
        post_authors = all_authors[valid_idx]
        author_success = agent_cumulative_likes[post_authors]
        weights = np.power(author_success + 1, alpha)
        probabilities = weights / np.sum(weights)
        
        n_selected = min(k, n_available)
        selected_indices = np.random.choice(n_available, size=n_selected, replace=False, p=probabilities)
        
        selected_authors[user_i, :n_selected] = all_authors[valid_idx[selected_indices]]
        selected_times[user_i, :n_selected] = all_times[valid_idx[selected_indices]]
    
    return selected_authors, selected_times