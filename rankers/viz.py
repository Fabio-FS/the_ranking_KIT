"""
viz.py — plotting utilities for SimResult.

Four functions:
  plot_belief_trajectories  — opinion over time for each tracked agent
  plot_info_trajectories    — unique claims seen over time for each tracked agent
  plot_network_beliefs      — network, nodes coloured by final belief (log-odds)
  plot_network_info         — network, nodes coloured by total unique claims seen
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import igraph as ig
from .metrics import SimResult


def plot_belief_trajectories(result: SimResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Line plot: one trajectory per tracked agent, x = step, y = perceived probability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    steps = np.arange(result.belief_trajectories.shape[0]) * result.cfg.record_every
    probabilities = 1.0 / (1.0 + np.exp(-result.belief_trajectories))
    for k in range(result.cfg.n_tracked):
        ax.plot(steps, probabilities[:, k], lw=0.8, alpha=0.5)

    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("step")
    ax.set_ylabel("perceived probability")
    ax.set_title("Opinion trajectories of tracked agents")
    return ax


def plot_info_trajectories(result: SimResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Line plot: unique claims seen over time for each tracked agent."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    steps = np.arange(result.repertoire_history.shape[0]) * result.cfg.record_every
    for k in range(result.cfg.n_tracked):
        ax.plot(steps, result.repertoire_history[:, k], lw=0.8, alpha=0.5)

    ax.set_xlabel("step")
    ax.set_ylabel("unique claims seen")
    ax.set_title("Information exposure of tracked agents")
    return ax


def _igraph_layout_positions(g: ig.Graph) -> np.ndarray:
    """Return (N, 2) Kamada-Kawai layout positions."""
    layout = g.layout("kk")
    return np.array(layout.coords)


def _draw_network(
    g: ig.Graph,
    node_values: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    cbar_label: str,
    ax: plt.Axes,
) -> None:
    """Shared drawing logic for both network plots."""
    pos = _igraph_layout_positions(g)
    edges = np.array(g.get_edgelist())

    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mapper.to_rgba(node_values)

    # Draw edges first (grey, thin)
    for u, v in edges:
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                color="grey", lw=0.3, alpha=0.3, zorder=0)

    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=8, zorder=1, linewidths=0)

    plt.colorbar(mapper, ax=ax, label=cbar_label, shrink=0.7)
    ax.set_title(title)
    ax.axis("off")


def plot_network_beliefs(result: SimResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Network coloured by final log-odds belief. Blue = negative, red = positive."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    beliefs = result.final_beliefs
    extreme = max(abs(beliefs.min()), abs(beliefs.max()))
    _draw_network(
        g            = result.graph,
        node_values  = beliefs,
        cmap         = "RdBu_r",
        vmin         = -extreme,
        vmax         =  extreme,
        title        = "Network: final belief (log-odds)",
        cbar_label   = "log-odds",
        ax           = ax,
    )
    return ax


def plot_network_info(result: SimResult, ax: plt.Axes | None = None) -> plt.Axes:
    """Network coloured by total unique claims seen at end of simulation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    counts = result.info_counts
    _draw_network(
        g            = result.graph,
        node_values  = counts,
        cmap         = "viridis",
        vmin         = counts.min(),
        vmax         = counts.max(),
        title        = "Network: unique claims seen",
        cbar_label   = "# unique claims",
        ax           = ax,
    )
    return ax


# plot_trajectory_grid.py
def plot_trajectory_grid(sweep, betas, n_reps=1, record_every=1):
    n_rows = len(betas)
    fig, axes = plt.subplots(n_rows, n_reps, figsize=(4 * n_reps, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    steps = np.arange(sweep[betas[0]]["trajectories"].shape[1]) * record_every

    for row, beta in enumerate(betas):
        traj = sweep[beta]["trajectories"]   # (R, n_records, N)
        for rep in range(n_reps):
            ax = axes[row, rep]
            opinion = 1.0 / (1.0 + np.exp(-traj[rep]))   # (n_records, N)
            ax.plot(steps, opinion, lw=0.4, alpha=0.3)
            ax.axhline(0.5, color="black", lw=0.8, ls="--")
            ax.set_ylim(0, 1)
            ax.set_title(f"β = {beta}, rep {rep}")

    for ax in axes[-1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("opinion")

    fig.tight_layout()
    return fig

# plot_trajectory_heatmap_grid.py
def plot_trajectory_heatmap_grid(sweep, betas, n_reps=1, n_bins=20, record_every=1):
    n_rows = len(betas)
    fig, axes = plt.subplots(n_rows, n_reps, figsize=(4 * n_reps, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for row, beta in enumerate(betas):
        traj = sweep[beta]["trajectories"]   # (R, n_records, N)
        n_records = traj.shape[1]
        for rep in range(n_reps):
            ax = axes[row, rep]
            opinion = 1.0 / (1.0 + np.exp(-traj[rep]))   # (n_records, N)

            heat = np.empty((n_bins, n_records))
            for t in range(n_records):
                heat[:, t] = np.histogram(opinion[t], bins=bins)[0]

            ax.imshow(heat, origin="lower", aspect="auto",
                      extent=[0, n_records * record_every, 0.0, 1.0],
                      cmap="viridis")
            ax.axhline(0.5, color="white", lw=0.8, ls="--")
            ax.set_title(f"β = {beta}, rep {rep}")

    for ax in axes[-1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("opinion")

    fig.tight_layout()
    return fig


    # plot_trajectory_heatmap_avg.py
def plot_trajectory_heatmap_avg(sweep, betas, n_reps=1, n_bins=20, record_every=1):
    n_rows = len(betas)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for row, beta in enumerate(betas):
        traj = sweep[beta]["trajectories"]   # (R, n_records, N)
        n_records = traj.shape[1]
        ax = axes[row, 0]

        heat = np.zeros((n_bins, n_records))
        for rep in range(n_reps):
            opinion = 1.0 / (1.0 + np.exp(-traj[rep]))   # (n_records, N)
            for t in range(n_records):
                heat[:, t] += np.histogram(opinion[t], bins=bins)[0]
        heat /= n_reps

        ax.imshow(heat, origin="lower", aspect="auto",
                  extent=[0, n_records * record_every, 0.0, 1.0],
                  cmap="viridis")
        ax.axhline(0.5, color="white", lw=0.8, ls="--")
        ax.set_title(f"β = {beta}  (avg over {n_reps} reps)")

    axes[-1, 0].set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("opinion")

    fig.tight_layout()
    return fig