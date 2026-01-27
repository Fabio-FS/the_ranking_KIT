import igraph as ig
import numpy as np
from src.graphs import build_graph
from src.metrics import compute_timestep_metrics, record_dying_posts, compute_final_metrics
# import models:
import src.models.BCM

# import rankers:
import src.rankers.random, src.rankers.closest, src.rankers.diverse_engagement 
import src.rankers.engagement, src.rankers.narrative, src.rankers.evil
import src.rankers.user_success


INIT_OD = {
    'BCM': src.models.BCM.initialize
}
UPDATE = {
    'BCM': src.models.BCM.update
}


INIT_RANKER = {
    'Random': src.rankers.random.initialize,
    'Closest': src.rankers.closest.initialize,
    'Diverse_Engagement': src.rankers.diverse_engagement.initialize,
    'Engagement': src.rankers.engagement.initialize,
    'Narrative': src.rankers.narrative.initialize,
    "Evil": src.rankers.evil.initialize,
    'User_Success': src.rankers.user_success.initialize
}
RANKER = {
    'Random': src.rankers.random.rank,
    'Closest': src.rankers.closest.rank,
    'Diverse_Engagement': src.rankers.diverse_engagement.rank,
    'Engagement': src.rankers.engagement.rank,
    'Narrative': src.rankers.narrative.rank,
    "Evil": src.rankers.evil.rank,
    'User_Success': src.rankers.user_success.rank
}




def run_replicas(info, n_replicas=100, n_save_trajectories=5):
    """
    Run multiple simulation replicas and aggregate results.
    
    Args:
        info: Configuration dictionary
        n_replicas: Number of replicas to run
        n_save_trajectories: Number of replicas to save full opinion trajectories (memory intensive)
        
    Returns:
        dict with aggregated results across replicas
    """
    # Get dimensions from first run (quick dry run)
    print(f"Running {n_replicas} replicas...")
    first_run = simulate(info)
    n_steps = len(first_run['mean'])
    n_users = first_run['G'].vcount()
    n_bins_1d = len(first_run['histogram_1d'])
    n_opinion_bins, n_like_bins = first_run['histogram_2d'].shape
    
    # Preallocate storage arrays
    all_mean = np.zeros((n_replicas, n_steps))
    all_pol = np.zeros((n_replicas, n_steps))
    all_filter_bubble = np.zeros((n_replicas, n_steps))
    all_gini_success = np.zeros((n_replicas, n_steps))
    all_gini_reach = np.zeros((n_replicas, n_steps))
    all_homophily = np.zeros((n_replicas, n_steps))
    all_histogram_1d = np.zeros((n_replicas, n_bins_1d))
    all_histogram_2d = np.zeros((n_replicas, n_opinion_bins, n_like_bins))
    
    # Only save trajectories for first few replicas
    all_opinions = np.zeros((n_save_trajectories, n_steps, n_users))
    
    # Store first run (already computed)
    all_mean[0] = first_run['mean']
    all_pol[0] = first_run['pol']
    all_filter_bubble[0] = first_run['filter_bubble']
    all_gini_success[0] = first_run['gini_success']
    all_gini_reach[0] = first_run['gini_reach']
    all_homophily[0] = first_run['homophily']
    all_histogram_1d[0] = first_run['histogram_1d']
    all_histogram_2d[0] = first_run['histogram_2d']
    all_opinions[0] = first_run['opinions']
    
    # Run remaining replicas
    for rep in range(1, n_replicas):
        print(f"  Replica {rep}/{n_replicas}")
        
        result = simulate(info)
        
        all_mean[rep] = result['mean']
        all_pol[rep] = result['pol']
        all_filter_bubble[rep] = result['filter_bubble']
        all_gini_success[rep] = result['gini_success']
        all_gini_reach[rep] = result['gini_reach']
        all_homophily[rep] = result['homophily']
        all_histogram_1d[rep] = result['histogram_1d']
        all_histogram_2d[rep] = result['histogram_2d']
        
        # Only save trajectories for first n_save_trajectories replicas
        if rep < n_save_trajectories:
            all_opinions[rep] = result['opinions']
    
    print(f"Completed {n_replicas} replicas!")
    
    return {
        'info': info,
        'n_replicas': n_replicas,
        'mean': all_mean,
        'pol': all_pol,
        'filter_bubble': all_filter_bubble,
        'gini_success': all_gini_success,
        'gini_reach': all_gini_reach,
        'homophily': all_homophily,
        'histogram_1d': all_histogram_1d,
        'histogram_2d': all_histogram_2d,
        'opinions': all_opinions,  # Only first n_save_trajectories
        'n_saved_trajectories': n_save_trajectories
    }






def simulate(info):
    # Initialize the network (with neighbor_matrix)
    G = build_graph(info)

    # Get functions from dictionaries
    init_fn = INIT_OD[info['OD']["model"]]
    init_ranker_fn = INIT_RANKER[info['Ranker']["rule"]]
    update_fn = UPDATE[info['OD']["model"]]
    ranker_fn = RANKER[info['Ranker']["rule"]]

    # Initialize OD model (returns post arrays)
    post_opinions, post_likes, post_seen_by = init_fn(G, info)
    init_ranker_fn(G, info)

    # Metrics storage
    n_steps = info["Simulation_details"].get('n_steps', 100)
    n_users = G.vcount()  
  
    RES_mean = np.zeros(n_steps)
    RES_pol = np.zeros(n_steps)
    RES_opinions = np.zeros((n_steps, n_users))
    RES_filter_bubble = np.zeros(n_steps)
    RES_gini_success = np.zeros(n_steps)
    RES_gini_reach = np.zeros(n_steps)
    RES_homophily = np.zeros(n_steps)

    like_bins = [0, 1, 2, 5, 10, 20, 50, 100]
    opinion_bins = np.linspace(0, 1, 11)

    RES_histogram_1d = np.zeros(len(like_bins) + 1, dtype=np.int32)  # One extra bin for [100, inf)
    RES_histogram_2d = np.zeros((len(opinion_bins) - 1, len(like_bins) + 1), dtype=np.int32)


    for step in range(n_steps):
        G['current_step'] = step
        
        # Record posts that are about to be overwritten
        current_time = G['current_time_idx']
        if step >= G["post_history"]:  # Only record after first wrap-around
            record_dying_posts(current_time, post_opinions, post_likes, 
                            RES_histogram_1d, RES_histogram_2d, like_bins, opinion_bins)
        
        # Select posts for each agent (ranker)
        selected_posts = ranker_fn(G, info, post_opinions, post_likes, post_seen_by)
        
        # Read, like, update opinions, generate new posts (OD model)
        update_fn(G, info, selected_posts, post_opinions, post_likes, post_seen_by)
        
        # Compute timestep metrics
        fb, gini_s, gini_r, homophily = compute_timestep_metrics(
            G, selected_posts, post_opinions, post_likes, post_seen_by
        )
        
        # Store metrics
        current_opinions = np.array(G.vs['opinion'])
        RES_mean[step] = np.mean(current_opinions)
        RES_pol[step] = np.var(current_opinions)
        RES_opinions[step] = current_opinions
        RES_filter_bubble[step] = fb
        RES_gini_success[step] = gini_s
        RES_gini_reach[step] = gini_r
        RES_homophily[step] = homophily
    
    final_metrics = compute_final_metrics(G, post_opinions, post_likes, 
                                     RES_histogram_1d, RES_histogram_2d, 
                                     like_bins, opinion_bins)
    
    return {
        'G': G,
        'mean': RES_mean,
        'pol': RES_pol,
        'opinions': RES_opinions,
        'filter_bubble': RES_filter_bubble,
        'gini_success': RES_gini_success,
        'gini_reach': RES_gini_reach,
        'homophily': RES_homophily,
        'histogram_1d': final_metrics['success_histogram_1d'],
        'histogram_2d': final_metrics['success_histogram_2d'],
        'post_likes': post_likes
    }