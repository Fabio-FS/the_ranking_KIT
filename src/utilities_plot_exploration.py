import numpy as np
import matplotlib.pyplot as plt

def plot_simulation(result, info, title=None):
    conv = result['convergence_step']
    T = conv + 1  # plot only up to convergence

    opinions = result['opinions'][:T]          # (T, n_users)
    pol      = result['pol'][:T]
    hom      = result['homophily'][:T]
    mean     = result['log_ewma_mean'][:T]     # (T, n_users)
    var      = result['log_ewma_var'][:T]      # (T, n_users)

    n_users = opinions.shape[1]
    steps = np.arange(T)

    # color each agent by its initial opinion
    initial_opinions = opinions[0]
    colors = plt.cm.coolwarm(initial_opinions)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    ax = axes[0, 0]
    for i in range(n_users):
        ax.plot(steps, opinions[:, i], color=colors[i], alpha=0.4, linewidth=0.6)
    ax.set_title('opinions (proxy for pol)')
    ax.set_xlabel('step')
    ax.set_ylabel('opinion')

    ax = axes[0, 1]
    ax.plot(steps, hom, color='black')
    ax.set_title('homophily')
    ax.set_xlabel('step')
    ax.set_ylabel('homophily')

    ax = axes[1, 0]
    for i in range(n_users):
        ax.plot(steps, mean[:, i], color=colors[i], alpha=0.4, linewidth=0.6)
    ax.set_title('ewma_mean per agent')
    ax.set_xlabel('step')
    ax.set_ylabel('ewma_mean')

    ax = axes[1, 1]
    for i in range(n_users):
        ax.plot(steps, var[:, i], color=colors[i], alpha=0.4, linewidth=0.6)
    ax.set_title('ewma_var per agent')
    ax.set_xlabel('step')
    ax.set_ylabel('ewma_var')

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()