import numpy as np
import matplotlib.pyplot as plt
import json
import os



def save_results(results, base_filepath):
    """
    Save results efficiently: NPZ for data, JSON for config.
    """
    import json
    
    # Extract config
    info = results['info']
    
    # Save numerical data as NPZ
    npz_path = base_filepath + '.npz'
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    
    results_copy = results.copy()
    results_copy.pop('info')  # Remove info from NPZ
    np.savez_compressed(npz_path, **results_copy)
    
    # Save config as JSON (human-readable)
    json_path = base_filepath + '_config.json'
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Saved data: {npz_path}")
    print(f"Saved config: {json_path}")



def load_results(base_filepath):
    """
    Load results from NPZ and JSON files.
    
    Args:
        base_filepath: path without extension (e.g., 'results/random_ranker_eps02')
        
    Returns:
        tuple: (data_dict, config_dict)
            - data_dict: dict with all numerical arrays (mean, pol, etc.)
            - config_dict: simulation configuration (info)
    """
    import json
    
    # Load numerical data from NPZ
    npz_path = base_filepath + '.npz'
    loaded = np.load(npz_path, allow_pickle=True)
    
    # Convert to regular dict (npz returns special dict-like object)
    data_dict = {key: loaded[key] for key in loaded.files}
    
    # Load config from JSON
    json_path = base_filepath + '_config.json'
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    
    print(f"Loaded data from: {npz_path}")
    print(f"Loaded config from: {json_path}")
    print(f"Contains {data_dict.get('n_replicas', 'N/A')} replicas")
    
    return data_dict, config_dict







def plot_simulation_results(results, info):
    """
    Generate comprehensive visualization of simulation results.
    
    Args:
        results: dict with all simulation outputs
        info: configuration dictionary
    """
    fig = plt.figure(figsize=(16, 10))
    
    n_steps = len(results['mean'])
    time = np.arange(n_steps)
    
    # Top row: Opinion dynamics
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, results['mean'], label='Mean opinion')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Opinion')
    ax1.set_title('Opinion Mean Evolution')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, results['pol'], label='Polarization', color='C1')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Variance')
    ax2.set_title('Polarization (Opinion Variance)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time, results['homophily'], label='Homophily', color='C2')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Homophily')
    ax3.set_title('Network Homophily')
    ax3.grid(True, alpha=0.3)
    
    # Middle row: Exposure and inequality
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, results['filter_bubble'], label='Filter Bubble', color='C3')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Mean |opinion diff|')
    ax4.set_title('Filter Bubble Strength')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, results['gini_success'], label='Success Gini', color='C4')
    ax5.plot(time, results['gini_reach'], label='Reach Gini', color='C5', linestyle='--')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Gini Coefficient')
    ax5.set_title('Inequality Evolution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    for i in range(0, len(results['opinions'][0]), max(1, len(results['opinions'][0])//20)):
        ax6.plot(time, results['opinions'][:, i], alpha=0.3, linewidth=0.5)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Opinion')
    ax6.set_title('Opinion Trajectories (sample)')
    ax6.grid(True, alpha=0.3)
    
    # Bottom row: Histograms (LOG SCALE)
    ax7 = plt.subplot(3, 3, 7)
    like_bins = [0, 1, 2, 5, 10, 20, 50, 100]
    bin_labels = ['0', '1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
    ax7.bar(range(len(results['histogram_1d'])), results['histogram_1d'])
    ax7.set_xticks(range(len(results['histogram_1d'])))
    ax7.set_xticklabels(bin_labels, rotation=45)
    ax7.set_xlabel('Number of Likes')
    ax7.set_ylabel('Number of Posts (log scale)')
    ax7.set_yscale('log')
    ax7.set_title('Distribution of Post Success')
    ax7.grid(True, alpha=0.3, axis='y')
    
    ax8 = plt.subplot(3, 3, 8)
    # Add small constant to avoid log(0)
    histogram_2d_plot = results['histogram_2d'].T + 1
    im = ax8.imshow(histogram_2d_plot, aspect='auto', origin='lower', 
                    cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    ax8.set_xlabel('Opinion Bin')
    ax8.set_ylabel('Likes Bin')
    ax8.set_title('Opinion vs Success (2D, log scale)')
    ax8.set_xticks(range(10))
    ax8.set_xticklabels([f'{i/10:.1f}' for i in range(10)], rotation=45)
    ax8.set_yticks(range(len(bin_labels)))
    ax8.set_yticklabels(bin_labels)
    plt.colorbar(im, ax=ax8, label='Post Count (log)')
    
    # Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
Simulation Summary
━━━━━━━━━━━━━━━━━
Graph: {info['Graph']['type']} (n={info['Graph']['n']})
OD Model: {info['OD']['model']}
  ε = {info['OD']['epsilon']}
  μ = {info['OD']['mu']}
Ranker: {info['Ranker']['rule']}
Steps: {info['Simulation_details']['n_steps']}
Posts/step: {info.get('k_posts', 1)}

Final State
━━━━━━━━━━━━━━━━━
Mean opinion: {results['mean'][-1]:.3f}
Polarization: {results['pol'][-1]:.3f}
Homophily: {results['homophily'][-1]:.3f}
Filter bubble: {results['filter_bubble'][-1]:.3f}
Success Gini: {results['gini_success'][-1]:.3f}
Reach Gini: {results['gini_reach'][-1]:.3f}
    """
    ax9.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    return fig

def plot_replica_comparison(data, config):
    """
    Generate comprehensive visualization with individual replica trajectories.
    
    Args:
        data: dict with all numerical arrays from load_results (first element of tuple)
        config: simulation configuration (second element of tuple)
    """
    fig = plt.figure(figsize=(16, 10))
    
    n_replicas = data['n_replicas']
    n_steps = data['mean'].shape[1]
    time = np.arange(n_steps)
    
    # Compute means across replicas
    mean_avg = np.mean(data['mean'], axis=0)
    pol_avg = np.mean(data['pol'], axis=0) * 4  # Scale polarization
    homophily_avg = np.mean(data['homophily'], axis=0)
    fb_avg = np.mean(data['filter_bubble'], axis=0)
    gini_s_avg = np.mean(data['gini_success'], axis=0)
    gini_r_avg = np.mean(data['gini_reach'], axis=0)
    
    # Compute stds for summary text
    mean_std = np.std(data['mean'], axis=0)
    pol_std = np.std(data['pol'], axis=0) * 4
    homophily_std = np.std(data['homophily'], axis=0)
    fb_std = np.std(data['filter_bubble'], axis=0)
    gini_s_std = np.std(data['gini_success'], axis=0)
    gini_r_std = np.std(data['gini_reach'], axis=0)
    
    # Top row: Opinion dynamics
    ax1 = plt.subplot(3, 3, 1)
    for rep in range(n_replicas):
        ax1.plot(time, data['mean'][rep], alpha=0.2, color='C0', linewidth=0.5)
    ax1.plot(time, mean_avg, color='C0', linewidth=2, label='Mean')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Opinion')
    ax1.set_title(f'Opinion Mean (n={n_replicas} replicas)')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    for rep in range(n_replicas):
        ax2.plot(time, data['pol'][rep] * 4, alpha=0.2, color='C1', linewidth=0.5)
    ax2.plot(time, pol_avg, color='C1', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Polarization')
    ax2.set_title('Polarization')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    for rep in range(n_replicas):
        ax3.plot(time, data['homophily'][rep], alpha=0.2, color='C2', linewidth=0.5)
    ax3.plot(time, homophily_avg, color='C2', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Homophily')
    ax3.set_title('Network Homophily')
    ax3.grid(True, alpha=0.3)
    
    # Middle row: Exposure and inequality
    ax4 = plt.subplot(3, 3, 4)
    for rep in range(n_replicas):
        ax4.plot(time, data['filter_bubble'][rep], alpha=0.2, color='C3', linewidth=0.5)
    ax4.plot(time, fb_avg, color='C3', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Mean |opinion diff|')
    ax4.set_title('Filter Bubble Strength')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    for rep in range(n_replicas):
        ax5.plot(time, data['gini_success'][rep], alpha=0.2, color='C4', linewidth=0.5)
    ax5.plot(time, gini_s_avg, color='C4', linewidth=2)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Gini Coefficient')
    ax5.set_title('Success Inequality (Gini)')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    for rep in range(n_replicas):
        ax6.plot(time, data['gini_reach'][rep], alpha=0.2, color='C5', linewidth=0.5)
    ax6.plot(time, gini_r_avg, color='C5', linewidth=2)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Gini Coefficient')
    ax6.set_title('Reach Inequality (Gini)')
    ax6.grid(True, alpha=0.3)
    
    # Bottom row: Histograms (averaged across replicas)
    ax7 = plt.subplot(3, 3, 7)
    like_bins = [0, 1, 2, 5, 10, 20, 50, 100]
    bin_labels = ['0', '1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
    histogram_1d_avg = np.mean(data['histogram_1d'], axis=0)
    ax7.bar(range(len(histogram_1d_avg)), histogram_1d_avg)
    ax7.set_xticks(range(len(histogram_1d_avg)))
    ax7.set_xticklabels(bin_labels, rotation=45)
    ax7.set_xlabel('Number of Likes')
    ax7.set_ylabel('Number of Posts (log scale)')
    ax7.set_yscale('log')
    ax7.set_title('Distribution of Post Success (averaged)')
    ax7.grid(True, alpha=0.3, axis='y')
    
    ax8 = plt.subplot(3, 3, 8)
    histogram_2d_avg = np.mean(data['histogram_2d'], axis=0)
    histogram_2d_plot = histogram_2d_avg.T + 1
    im = ax8.imshow(histogram_2d_plot, aspect='auto', origin='lower', 
                    cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    ax8.set_xlabel('Opinion Bin')
    ax8.set_ylabel('Likes Bin')
    ax8.set_title('Opinion vs Success (averaged, log scale)')
    ax8.set_xticks(range(10))
    ax8.set_xticklabels([f'{i/10:.1f}' for i in range(10)], rotation=45)
    ax8.set_yticks(range(len(bin_labels)))
    ax8.set_yticklabels(bin_labels)
    plt.colorbar(im, ax=ax8, label='Post Count (log)')
    
    # Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Final values: mean ± std
    final_mean = f"{mean_avg[-1]:.3f} ± {mean_std[-1]:.3f}"
    final_pol = f"{pol_avg[-1]:.3f} ± {pol_std[-1]:.3f}"
    final_homophily = f"{homophily_avg[-1]:.3f} ± {homophily_std[-1]:.3f}"
    final_fb = f"{fb_avg[-1]:.3f} ± {fb_std[-1]:.3f}"
    final_gini_s = f"{gini_s_avg[-1]:.3f} ± {gini_s_std[-1]:.3f}"
    final_gini_r = f"{gini_r_avg[-1]:.3f} ± {gini_r_std[-1]:.3f}"
    
    summary_text = f"""
Simulation Summary
━━━━━━━━━━━━━━━━━
Replicas: {n_replicas}
Graph: {config['Graph']['type']} (n={config['Graph']['n']})
OD Model: {config['OD']['model']}
  ε = {config['OD']['epsilon']}
  μ = {config['OD']['mu']}
Ranker: {config['Ranker']['rule']}
Steps: {config['Simulation_details']['n_steps']}
Posts/step: {config.get('k_posts', 1)}

Final State (mean ± std)
━━━━━━━━━━━━━━━━━
Mean opinion: {final_mean}
Polarization: {final_pol}
Homophily: {final_homophily}
Filter bubble: {final_fb}
Success Gini: {final_gini_s}
Reach Gini: {final_gini_r}
    """
    ax9.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    return fig

def plot_first_replicas(data, config, n_replicas_to_plot=9):
    """
    Plot opinion trajectories for the first n replicas in a 3x3 grid.
    
    Args:
        data: dict with all numerical arrays from load_results
        config: simulation configuration
        n_replicas_to_plot: number of replicas to show (default 9 for 3x3 grid)
    """
    n_saved = data['n_saved_trajectories']
    n_to_plot = min(n_replicas_to_plot, n_saved)
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(n_to_plot)))
    
    fig = plt.figure(figsize=(16, 10))
    
    for i in range(n_to_plot):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        
        opinions = data['opinions'][i]  # Shape: (n_steps, n_users)
        n_steps, n_users = opinions.shape
        time = np.arange(n_steps)
        
        # Plot all user trajectories
        for user_id in range(n_users):
            ax.plot(time, opinions[:, user_id], alpha=0.3, linewidth=0.5)
        
        # Add mean trajectory
        mean_opinion = data['mean'][i]
        ax.plot(time, mean_opinion, color='black', linewidth=2, label='Mean')
        
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Opinion')
        ax.set_title(f'Replica {i+1}')
        ax.grid(True, alpha=0.3)
        
        # Add final stats as text
        final_mean = mean_opinion[-1]
        final_pol = data['pol'][i][-1] * 4
        ax.text(0.98, 0.02, f'μ={final_mean:.2f}\nπ={final_pol:.2f}', 
                transform=ax.transAxes, 
                verticalalignment='bottom', 
                horizontalalignment='right',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add overall title
    ranker_name = config['Ranker']['rule']
    epsilon = config['OD']['epsilon']
    fig.suptitle(f'{ranker_name} Ranker (ε={epsilon}) - First {n_to_plot} Replicas', 
                 fontsize=14, y=0.995)
    
    plt.tight_layout()
    return fig

def plot_ranker_comparison(results_list):
    """
    Compare multiple rankers side-by-side.
    
    Args:
        results_list: list of [data, config] pairs from different rankers
                     e.g., [[data1, config1], [data2, config2], ...]
    """
    fig = plt.figure(figsize=(18, 10))
    
    n_rankers = len(results_list)
    colors = plt.cm.tab10(np.linspace(0, 1, n_rankers))
    
    # Extract ranker names with parameters for legend
    ranker_names = []
    for data, config in results_list:
        rule = config['Ranker']['rule']
        if rule == 'Engagement':
            alpha = config['Ranker'].get('alpha', 1.0)
            ranker_names.append(f'{rule} (α={alpha})')
        elif rule == 'Narrative':
            target = config['Ranker'].get('target_opinion', 0.5)
            ranker_names.append(f'{rule} (target={target})')
        else:
            ranker_names.append(rule)
    
    # Top row: Opinion dynamics
    ax1 = plt.subplot(2, 3, 1)
    for idx, (data, config) in enumerate(results_list):
        n_replicas = data['n_replicas']
        time = np.arange(data['mean'].shape[1])
        
        # Plot all replicas
        for rep in range(n_replicas):
            ax1.plot(time, data['mean'][rep], alpha=0.1, color=colors[idx], linewidth=0.5)
        
        # Plot average
        mean_avg = np.mean(data['mean'], axis=0)
        ax1.plot(time, mean_avg, color=colors[idx], linewidth=2.5, label=ranker_names[idx])
    
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Opinion')
    ax1.set_title('Opinion Mean Evolution')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    for idx, (data, config) in enumerate(results_list):
        n_replicas = data['n_replicas']
        time = np.arange(data['pol'].shape[1])
        
        # Plot all replicas
        for rep in range(n_replicas):
            ax2.plot(time, data['pol'][rep] * 4, alpha=0.1, color=colors[idx], linewidth=0.5)
        
        # Plot average
        pol_avg = np.mean(data['pol'], axis=0) * 4
        ax2.plot(time, pol_avg, color=colors[idx], linewidth=2.5, label=ranker_names[idx])
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Polarization')
    ax2.set_title('Polarization Evolution')
    #ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    for idx, (data, config) in enumerate(results_list):
        n_replicas = data['n_replicas']
        time = np.arange(data['homophily'].shape[1])
        
        # Plot all replicas
        for rep in range(n_replicas):
            ax3.plot(time, data['homophily'][rep], alpha=0.1, color=colors[idx], linewidth=0.5)
        
        # Plot average
        homophily_avg = np.mean(data['homophily'], axis=0)
        ax3.plot(time, homophily_avg, color=colors[idx], linewidth=2.5, label=ranker_names[idx])
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Homophily')
    ax3.set_title('Network Homophily')
    #ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Bottom row: Filter bubble and Gini coefficients
    ax4 = plt.subplot(2, 3, 4)
    for idx, (data, config) in enumerate(results_list):
        n_replicas = data['n_replicas']
        time = np.arange(data['filter_bubble'].shape[1])
        
        # Plot all replicas
        for rep in range(n_replicas):
            ax4.plot(time, data['filter_bubble'][rep], alpha=0.1, color=colors[idx], linewidth=0.5)
        
        # Plot average
        fb_avg = np.mean(data['filter_bubble'], axis=0)
        ax4.plot(time, fb_avg, color=colors[idx], linewidth=2.5, label=ranker_names[idx])
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Filter Bubble Strength')
    ax4.set_title('Filter Bubble Evolution')
    #ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    for idx, (data, config) in enumerate(results_list):
        n_replicas = data['n_replicas']
        time = np.arange(data['gini_success'].shape[1])
        
        # Plot all replicas
        for rep in range(n_replicas):
            ax5.plot(time, data['gini_success'][rep], alpha=0.1, color=colors[idx], linewidth=0.5)
        
        # Plot average
        gini_s_avg = np.mean(data['gini_success'], axis=0)
        ax5.plot(time, gini_s_avg, color=colors[idx], linewidth=2.5, label=ranker_names[idx])
    
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Gini Coefficient')
    ax5.set_title('Success Inequality (Gini)')
    #ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    for idx, (data, config) in enumerate(results_list):
        n_replicas = data['n_replicas']
        time = np.arange(data['gini_reach'].shape[1])
        
        # Plot all replicas
        for rep in range(n_replicas):
            ax6.plot(time, data['gini_reach'][rep], alpha=0.1, color=colors[idx], linewidth=0.5)
        
        # Plot average
        gini_r_avg = np.mean(data['gini_reach'], axis=0)
        ax6.plot(time, gini_r_avg, color=colors[idx], linewidth=2.5, label=ranker_names[idx])
    
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Gini Coefficient')
    ax6.set_title('Reach Inequality (Gini)')
    #ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    # Overall title
    epsilon = results_list[0][1]['OD']['epsilon']
    n_users = results_list[0][1]['Graph']['n']
    fig.suptitle(f'Ranker Comparison (ε={epsilon}, n={n_users})', fontsize=14, y=0.995)
    
    plt.tight_layout()
    return fig