"""
Visualization module for recreating Figure 2d and 2e.

Implements figure generation following the styling and content of
Rumyantsev et al. 2020 Figure 2.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml


def create_figure_2d(
    real_correlations: np.ndarray,
    shuffled_correlations: np.ndarray,
    config_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Recreate Figure 2d from paper.
    
    Shows histograms of noise correlation coefficients for real and 
    trial-shuffled data.
    
    Paper Figure 2d: "Distribution of noise correlations for pairs of 
    neurons...The grey histogram shows the distribution obtained from 
    trial-shuffled data."
    
    Args:
        real_correlations: Correlation coefficients from real data
        shuffled_correlations: Correlation coefficients from shuffled data
        config_path: Path to visualization config (optional)
        
    Returns:
        fig, ax: Matplotlib figure and axes
        
    Example:
        >>> fig, ax = create_figure_2d(real_corr, shuffled_corr)
        >>> fig.savefig('figure_2d.png', dpi=300)
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'analysis_config.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    viz_config = config['visualization']['figure_2d']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram parameters
    bins = viz_config['n_bins']
    x_range = tuple(viz_config['x_range'])
    
    # Plot histograms
    ax.hist(
        real_correlations,
        bins=bins,
        range=x_range,
        alpha=0.7,
        color=viz_config['colors']['real'],
        label='Real data',
        density=False,
        edgecolor='none'
    )
    
    ax.hist(
        shuffled_correlations,
        bins=bins,
        range=x_range,
        alpha=0.6,
        color=viz_config['colors']['shuffled'],
        label='Shuffled',
        density=False,
        edgecolor='none'
    )
    
    # Styling to match paper
    ax.set_xlabel('Correlation coefficient', fontsize=14)
    ax.set_ylabel('Number of cell pairs', fontsize=14)
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # Add summary statistics annotation
    stats_text = (
        f"Real: μ={np.mean(real_correlations):.3f}, "
        f"σ={np.std(real_correlations):.3f}\n"
        f"Shuffled: μ={np.mean(shuffled_correlations):.3f}, "
        f"σ={np.std(shuffled_correlations):.3f}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    return fig, ax


def create_figure_2e(
    similarly_tuned_corr: np.ndarray,
    differently_tuned_corr: np.ndarray,
    pvalue: float,
    config_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Recreate Figure 2e from paper.
    
    Compares noise correlation distributions for similarly vs 
    differently tuned cell pairs.
    
    Paper Figure 2e: "Distribution of noise correlations for 
    similarly tuned (black) and differently tuned (red) cell pairs."
    
    Args:
        similarly_tuned_corr: Correlations for similarly tuned pairs
        differently_tuned_corr: Correlations for differently tuned pairs
        pvalue: KS test p-value
        config_path: Path to visualization config (optional)
        
    Returns:
        fig, ax: Matplotlib figure and axes
        
    Example:
        >>> fig, ax = create_figure_2e(sim_corr, diff_corr, pvalue=1e-10)
        >>> fig.savefig('figure_2e.png', dpi=300)
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'analysis_config.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    viz_config = config['visualization']['figure_2e']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histograms with density normalization for better comparison
    bins = 50
    ax.hist(
        similarly_tuned_corr,
        bins=bins,
        alpha=0.7,
        color=viz_config['colors']['similar'],
        label='Similarly tuned',
        density=True,
        edgecolor='none'
    )
    
    ax.hist(
        differently_tuned_corr,
        bins=bins,
        alpha=0.6,
        color=viz_config['colors']['different'],
        label='Differently tuned',
        density=True,
        edgecolor='none'
    )
    
    # Styling to match paper
    ax.set_xlabel('Correlation coefficient', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # Add p-value annotation
    ax.text(
        0.98, 0.98,
        f'KS test: p = {pvalue:.2e}',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    # Add mean values
    mean_text = (
        f"Similarly tuned: μ={np.mean(similarly_tuned_corr):.3f}\n"
        f"Differently tuned: μ={np.mean(differently_tuned_corr):.3f}"
    )
    ax.text(
        0.02, 0.98, mean_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    return fig, ax

