# src/models/BCM.py
import numpy as np
import igraph as ig

def initialize(G, info):
    """
    Initialize the Bounded Confidence Model (BCM).
    
    BCM is an opinion dynamics model where agents only update their opinions
    when exposed to opinions within a confidence bound (epsilon). The model includes:
    - Random initial opinions in [0,1]
    - Epsilon: confidence bound (agents ignore opinions > epsilon away)
    - Mu: convergence rate (how much agents move toward acceptable opinions)
    
    This function also initializes the post storage system where each user
    maintains a circular buffer of posts over time.
    
    Args:
        G: igraph Graph object
        info: Configuration dictionary with OD parameters and post_history
        
    Returns:
        tuple: (post_opinions, post_likes, post_seen_by)
            - post_opinions: (n_users, history_size) array of opinion values per post
            - post_likes: (n_users, history_size) array of like counts per post
            - post_seen_by: (n_users, history_size, n_users) boolean array tracking views
    """
    # Initialize each user's opinion randomly in [0, 1]
    G.vs['opinion'] = np.array(np.random.rand(G.vcount()))
    
    # Store BCM parameters as graph attributes for easy access
    G['epsilon'] = info["OD"].get('epsilon', 0.2)  # Confidence bound: only consider opinions within ±epsilon
    G['mu'] = info["OD"].get('mu', 0.1)  # Convergence rate: step size toward acceptable opinions
    
    # Neighbor relationships are already precomputed in build_graph (neighbor_matrix)
    
    # Set up post storage system
    n_users = G.vcount()
    history_size = info.get('post_history', 50)  # Circular buffer size for posts
    
    # Initialize post arrays:
    # - Each user has history_size post slots (circular buffer)
    # - Posts are identified by (author_id, time_index) coordinates
    post_opinions = np.zeros((n_users, history_size))  # Opinion value of each post
    post_likes = np.zeros((n_users, history_size), dtype=np.int32)  # Like count per post
    post_seen_by = np.zeros((n_users, history_size, n_users), dtype=bool)  # [author, time, viewer] = has viewer seen this post?
    
    # Initialize all post slots with each user's starting opinion
    # This represents each user having made history_size identical posts at initialization
    current_opinions = np.array(G.vs['opinion'])
    for i in range(n_users):
        post_opinions[i, :] = current_opinions[i]
    
    # Store metadata for circular buffer management
    G['post_history'] = history_size  # Buffer size
    G['current_time_idx'] = 0  # Next slot to write to (wraps around)
    
    return post_opinions, post_likes, post_seen_by


def _read_and_evaluate_posts(G, n_users, post_slot, selected_authors, selected_times, post_opinions, post_seen_by, post_likes, current_opinions):
    """
    Process one post slot for all users simultaneously.
    
    This is the core BCM update logic:
    1. Each user reads the post in their post_slot
    2. If post opinion is within epsilon, user likes it and moves toward it
    3. Mark post as seen
    
    All updates happen in-place on the arrays passed in.
    
    Args:
        G: Graph with BCM parameters (epsilon, mu)
        n_users: Number of users in network
        post_slot: Which post slot to process (0 to k-1)
        selected_authors: (n_users, k) array of post author IDs
        selected_times: (n_users, k) array of post time indices
        post_opinions: (n_users, history_size) array of post opinion values
        post_seen_by: (n_users, history_size, n_users) tracking array
        post_likes: (n_users, history_size) like counts (modified in-place)
        current_opinions: (n_users,) array of current opinions (modified in-place)
    """
    # Extract the post each user is reading at this slot
    authors = selected_authors[:, post_slot]  # Author of post shown to each user
    times = selected_times[:, post_slot]  # Time index of post shown to each user
    
    # Get the opinion value of each post being read
    valid_posts = authors >= 0
    
    # Only process valid posts
    post_ops = np.where(valid_posts, 
                       post_opinions[authors, times],
                       current_opinions)  # Dummy for invalid posts
    
    # Calculate opinion difference: positive = post is more extreme, negative = post is less extreme
    opinion_diff = post_ops - current_opinions
    
    # BCM rule: only interact with posts within confidence bound (epsilon)
    within_epsilon = np.abs(opinion_diff) < G['epsilon']
    
    # Award likes to posts within epsilon
    # Multiple users might like the same post, so we use bincount to aggregate
    like_mask = within_epsilon & valid_posts
    if np.any(like_mask):
        like_authors = authors[like_mask]  # Authors of posts being liked
        like_times = times[like_mask]  # Time indices of posts being liked
        
        # Convert 2D coordinates (author, time) to flat indices for bincount
        flat_indices = like_authors * post_likes.shape[1] + like_times
        
        # Count likes per post using bincount (handles duplicates automatically)
        like_counts = np.bincount(flat_indices, minlength=post_likes.size)
        
        # Add likes to post_likes array (in-place modification)
        post_likes += like_counts.reshape(post_likes.shape)
    
    # Update opinions: move toward posts within epsilon by convergence rate mu
    # Only opinions where within_epsilon=True are updated (boolean mask multiplication)
    current_opinions += G['mu'] * opinion_diff * within_epsilon * like_mask
    
    # Mark posts as seen by their respective viewers
    # Each user i sees the post (authors[i], times[i])
    if np.any(valid_posts): 
        seen_indices = np.arange(n_users)[valid_posts]  # Viewer IDs [0, 1, 2, ..., n_users-1]
        post_seen_by[authors[valid_posts], times[valid_posts], seen_indices] = True


def update(G, info, selected_posts, post_opinions, post_likes, post_seen_by):
    """
    Execute one time step of BCM dynamics with social media posts.
    
    The update follows this sequence:
    1. Each user reads their k selected posts sequentially (one at a time)
    2. For each post: like if within epsilon, update opinion incrementally
    3. After reading all posts: generate a new post with updated opinion
    4. Advance the circular buffer pointer
    
    Sequential post reading is crucial: reading post 1 changes your opinion,
    which affects how you react to post 2. This models real social media
    browsing where opinions evolve as you scroll.
    
    Args:
        G: Graph with BCM parameters and current opinions
        info: Configuration dictionary
        selected_posts: Tuple (selected_authors, selected_times) from ranker
        post_opinions: (n_users, history_size) array (modified in-place)
        post_likes: (n_users, history_size) array (modified in-place)
        post_seen_by: (n_users, history_size, n_users) array (modified in-place)
    """
    # Unpack selected posts from ranker
    selected_authors, selected_times = selected_posts
    
    # Get simulation parameters
    n_users = G.vcount()
    k = selected_authors.shape[1]  # Number of posts shown to each user this step
    epsilon = G['epsilon']  # Confidence bound
    mu = G['mu']  # Convergence rate
    
    # Load current opinions into working array (will be modified)
    current_opinions = np.array(G.vs['opinion'])
    
    # Process each post slot sequentially
    # This loop represents users scrolling through k posts one at a time
    # Opinion updates from post j affect how user reacts to post j+1
    for post_slot in range(k):
        _read_and_evaluate_posts(G, n_users, post_slot, selected_authors, selected_times, 
                                post_opinions, post_seen_by, post_likes, current_opinions)
    
    # Save updated opinions back to graph
    G.vs['opinion'] = current_opinions
    
    # Generate new posts with updated opinions
    # Each user makes one post per time step with their current opinion
    current_time = G['current_time_idx']  # Next slot in circular buffer
    post_opinions[:, current_time] = current_opinions  # Post contains current opinion
    post_likes[:, current_time] = 0  # New post starts with 0 likes
    post_seen_by[:, current_time, :] = False  # No one has seen the new post yet
    
    # Advance circular buffer pointer (wraps around at history_size)
    G['current_time_idx'] = (current_time + 1) % G['post_history']