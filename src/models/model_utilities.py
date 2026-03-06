import numpy as np


def initialize_tracking(G, info, n_users):
    ranker_rule = info['Ranker']['rule']
    G['track_cumulative_likes'] = ranker_rule in ('Engagement', 'User_Success')
    G['track_user_author_likes'] = ranker_rule == 'Personalization'

    if G['track_cumulative_likes']:
        G['agent_cumulative_likes'] = np.zeros(n_users, dtype=np.int32)

    if G['track_user_author_likes']:
        G['user_author_likes'] = np.zeros((n_users, n_users), dtype=np.int32)


def initialize_buffers(G, info, n_users):
    history_size = info.get('post_history', 50)

    post_opinions = np.zeros((n_users, history_size))
    post_likes = np.zeros((n_users, history_size), dtype=np.int32)
    post_seen_gen = np.zeros((n_users, history_size), dtype=np.int32)
    post_write_gen = np.zeros(history_size, dtype=np.int32)

    current_opinions = np.array(G.vs['opinion'])
    for i in range(n_users):
        post_opinions[i, :] = current_opinions[i]

    G['post_history'] = history_size
    G['current_time_idx'] = 0
    G['post_write_gen'] = post_write_gen

    return post_opinions, post_likes, post_seen_gen


def update_likes(G, like_mask, authors, times, post_likes, n_users):
    like_authors = authors[like_mask]
    like_times = times[like_mask]
    flat_indices = like_authors * post_likes.shape[1] + like_times
    like_counts = np.bincount(flat_indices, minlength=post_likes.size)
    post_likes += like_counts.reshape(post_likes.shape)

    if G['track_cumulative_likes']:
        author_like_counts = np.bincount(like_authors, minlength=n_users)
        G['agent_cumulative_likes'] += author_like_counts

    if G['track_user_author_likes']:
        like_users = np.where(like_mask)[0]
        np.add.at(G['user_author_likes'], (like_users, like_authors), 1)


def mark_seen(G, valid, times, post_seen_gen, n_users):
    seen_users = np.arange(n_users)[valid]
    seen_times = times[valid]
    post_seen_gen[seen_users, seen_times] = G['post_write_gen'][seen_times]


def advance_time(G, post_opinions, post_likes, current_opinions):
    current_time = G['current_time_idx']
    post_opinions[:, current_time] = current_opinions
    post_likes[:, current_time] = 0
    G['post_write_gen'][current_time] += 1
    G['current_time_idx'] = (current_time + 1) % G['post_history']


def compute_filter_bubble(total_abs_diff, total_count):
    return 1 - (total_abs_diff / total_count) if total_count > 0 else 0.0


def check_convergence(RES_pol, step, W, delta):
    return np.std(RES_pol[step - W : step]) < delta