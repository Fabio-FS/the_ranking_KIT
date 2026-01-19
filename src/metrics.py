import numpy as np
import igraph as ig

def compute_timestep_metrics(G, selected_posts, post_opinions, post_likes, post_seen_by):
    """
    Compute metrics for this timestep.
    
    Returns:
        tuple: (filter_bubble, gini_success, gini_reach, homophily)
    """
    selected_authors, selected_times = selected_posts
    n_users = G['N']
    k = selected_authors.shape[1]
    current_opinions = np.array(G.vs['opinion'])
    
    # 1. Filter bubble strength: <|o_i - o_post|>
    selected_authors_flat = selected_authors.flatten()
    selected_times_flat = selected_times.flatten()
    valid_mask = selected_authors_flat >= 0
    
    if np.any(valid_mask):
        user_indices = np.repeat(np.arange(n_users), k)[valid_mask]
        post_ops = post_opinions[selected_authors_flat[valid_mask], selected_times_flat[valid_mask]]
        user_ops = current_opinions[user_indices]
        filter_bubble = 1 - np.mean(np.abs(user_ops - post_ops))
    else:
        filter_bubble = 0.0
    
    # 2. Gini of success (cumulative likes so far)
    total_likes = post_likes.flatten()
    total_likes = total_likes[total_likes > 0]
    gini_success = compute_gini(total_likes) if len(total_likes) > 1 else 0.0
    
    # 3. Gini of reach (cumulative views so far)
    total_views = post_seen_by.sum(axis=2).flatten()
    total_views = total_views[total_views > 0]
    gini_reach = compute_gini(total_views) if len(total_views) > 1 else 0.0
    
    # 4. Homophily: <1 - |O_i - O_j|>_links
    neighbor_matrix = G['neighbor_matrix']
    opinion_diffs = np.abs(current_opinions[:, None] - current_opinions[None, :])
    homophily_values = (1 - opinion_diffs)[neighbor_matrix]
    homophily = np.mean(homophily_values)
    
    return filter_bubble, gini_success, gini_reach, homophily

def compute_gini(values):
    """
    Compute Gini coefficient for array of values.
    
    Gini = 0: perfect equality
    Gini = 1: perfect inequality
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n


def compute_final_metrics(G, post_opinions, post_likes, histogram_1d, histogram_2d, like_bins, opinion_bins):
    """
    Finalize histograms by adding posts still in the buffer.
    """
    # Add surviving posts (still in buffer) to histograms
    surviving_opinions = post_opinions.flatten()
    surviving_likes = post_likes.flatten()
    
    like_bin_indices = np.digitize(surviving_likes, bins=like_bins)
    opinion_bin_indices = np.digitize(surviving_opinions, bins=opinion_bins) - 1
    opinion_bin_indices = np.clip(opinion_bin_indices, 0, len(opinion_bins) - 2)
    
    # Update histograms with survivors
    for like_idx in like_bin_indices:
        histogram_1d[like_idx] += 1
    
    for op_idx, like_idx in zip(opinion_bin_indices, like_bin_indices):
        histogram_2d[op_idx, like_idx] += 1
    
    return {
        'success_histogram_1d': histogram_1d,
        'success_histogram_2d': histogram_2d
    }



def record_dying_posts(current_time, post_opinions, post_likes, histogram_1d, histogram_2d, like_bins, opinion_bins):
    """
    Record posts about to be overwritten into histograms.
    
    When the circular buffer wraps around, posts at current_time are about to die.
    Record their final (opinion, likes) values.
    """
    dying_opinions = post_opinions[:, current_time]
    dying_likes = post_likes[:, current_time]
    
    # Bin likes into histogram bins
    like_bin_indices = np.digitize(dying_likes, bins=like_bins)
    
    # Bin opinions into histogram bins
    opinion_bin_indices = np.digitize(dying_opinions, bins=opinion_bins) - 1
    opinion_bin_indices = np.clip(opinion_bin_indices, 0, len(opinion_bins) - 2)
    
    # Update 1D histogram
    for like_idx in like_bin_indices:
        histogram_1d[like_idx] += 1
    
    # Update 2D histogram
    for op_idx, like_idx in zip(opinion_bin_indices, like_bin_indices):
        histogram_2d[op_idx, like_idx] += 1