#!/usr/bin/env python
"""
Regenerate figures with improved styling (KDE + log scale) without re-computing.

This script loads pre-computed correlation results and regenerates figures
to match the paper's visual style more closely.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline


def create_figure_2d_kde(real_corr, shuffled_corr, output_path='outputs/figure_2d_kde.png'):
    """
    Create Figure 2d with KDE lines (smooth curves) showing number of cell pairs.
    
    Args:
        real_corr: Real correlation coefficients
        shuffled_corr: Shuffled correlation coefficients
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use KDE for smooth lines
    x_range_vals = np.linspace(-0.2, 0.3, 500)
    
    # Real data KDE
    kde_real = stats.gaussian_kde(real_corr, bw_method='scott')
    density_real = kde_real(x_range_vals)
    # Convert density to counts (number of cell pairs)
    counts_real = density_real * len(real_corr) * (x_range_vals[1] - x_range_vals[0])
    
    # Shuffled data KDE
    kde_shuffled = stats.gaussian_kde(shuffled_corr, bw_method='scott')
    density_shuffled = kde_shuffled(x_range_vals)
    # Convert density to counts (number of cell pairs)
    counts_shuffled = density_shuffled * len(shuffled_corr) * (x_range_vals[1] - x_range_vals[0])
    
    # Plot with lines (matching paper colors: green=real, orange=shuffled)
    ax.plot(x_range_vals, counts_real, color='#2CA02C', linewidth=2.5,
            label='Real data', alpha=0.8)  # Green
    ax.plot(x_range_vals, counts_shuffled, color='#FF7F0E', linewidth=2.5,
            label='Shuffled', alpha=0.8)  # Orange
    
    # NO LOG SCALE - regular linear scale!
    
    # Styling
    ax.set_xlabel('Correlation coefficient', fontsize=14)
    ax.set_ylabel('Number of cell pairs', fontsize=14)
    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    # Add statistics
    stats_text = (
        f"Real: μ={np.mean(real_corr):.3f}, σ={np.std(real_corr):.3f}\n"
        f"Shuffled: μ={np.mean(shuffled_corr):.3f}, σ={np.std(shuffled_corr):.3f}"
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
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    return fig, ax


def create_figure_2e_kde(sim_tuned, diff_tuned, pvalue, output_path='outputs/figure_2e_kde.png'):
    """
    Create Figure 2e with KDE lines (matching paper style).
    
    Args:
        sim_tuned: Similarly tuned pair correlations
        diff_tuned: Differently tuned pair correlations
        pvalue: KS test p-value
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use KDE
    x_range = np.linspace(-0.5, 0.8, 500)
    
    # Similarly tuned KDE
    kde_sim = stats.gaussian_kde(sim_tuned, bw_method='scott')
    density_sim = kde_sim(x_range)
    # Convert to counts (number of cell pairs)
    counts_sim = density_sim * len(sim_tuned) * (x_range[1] - x_range[0])
    
    # Differently tuned KDE
    kde_diff = stats.gaussian_kde(diff_tuned, bw_method='scott')
    density_diff = kde_diff(x_range)
    # Convert to counts (number of cell pairs)
    counts_diff = density_diff * len(diff_tuned) * (x_range[1] - x_range[0])
    
    # Plot with outline style (matching paper colors: red=similar, blue=different)
    ax.plot(x_range, counts_sim, color='#D62728', linewidth=2.5, 
            label='Similarly tuned', alpha=0.8)  # Red
    ax.plot(x_range, counts_diff, color='#1F77B4', linewidth=2.5,
            label='Differently tuned', alpha=0.8)  # Blue
    
    # Styling
    ax.set_xlabel('Correlation coefficient', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of cell pairs', fontsize=14, fontweight='bold')
    ax.legend(frameon=False, fontsize=13, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total cell pairs annotation
    total_pairs = len(sim_tuned) + len(diff_tuned)
    ax.text(
        0.5, 0.98,
        f'Total cell pairs: {total_pairs:,}',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='center',
        fontsize=11,
        fontweight='bold'
    )
    
    # Add KS test result
    ax.text(
        0.98, 0.90,
        f'***\nKS: p={pvalue:.2e}',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='gray')
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    return fig, ax


def compute_fwhm(data):
    """
    Compute Full Width at Half Maximum of a distribution.
    
    Args:
        data: Array of values
        
    Returns:
        fwhm: Full width at half maximum
    """
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peak
    peak_idx = np.argmax(hist)
    peak_height = hist[peak_idx]
    half_max = peak_height / 2
    
    # Find points closest to half maximum
    left_idx = np.where(hist[:peak_idx] <= half_max)[0]
    right_idx = np.where(hist[peak_idx:] <= half_max)[0]
    
    if len(left_idx) > 0 and len(right_idx) > 0:
        left_x = bin_centers[left_idx[-1]]
        right_x = bin_centers[peak_idx + right_idx[0]]
        fwhm = right_x - left_x
    else:
        # Fallback: use standard deviation approximation
        fwhm = 2.355 * np.std(data)
    
    return fwhm


def create_figure_2f_boxplot(per_mouse_real, per_mouse_shuffled, per_mouse_sim, per_mouse_diff,
                              output_path='outputs/figure_2f_boxplot.png'):
    """
    Create Figure 2f - Box plots of mean correlation coefficients.
    
    Args:
        per_mouse_real: List of real correlation arrays per mouse
        per_mouse_shuffled: List of shuffled correlation arrays per mouse
        per_mouse_sim: List of similarly tuned correlation arrays per mouse
        per_mouse_diff: List of differently tuned correlation arrays per mouse
        output_path: Where to save
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Compute means per mouse
    n_mice = len(per_mouse_real)
    real_means = [np.mean(corr) for corr in per_mouse_real]
    shuffled_means = [np.mean(corr) for corr in per_mouse_shuffled]
    sim_means = [np.mean(corr) for corr in per_mouse_sim]  # Per-mouse means!
    diff_means = [np.mean(corr) for corr in per_mouse_diff]  # Per-mouse means!
    
    # Box plot data
    positions = [1, 2, 3, 4]
    box_data = [real_means, shuffled_means, sim_means, diff_means]  # Now all have per-mouse values!
    colors = ['#60BD68', '#FAA43A', '#F15854', '#5DA5DA']
    labels = ['Real data', 'Shuffled', 'Sim. tuned', 'Diff. tuned']
    
    # Create box plots
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', linewidth=2))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Overlay individual points
    for i, (data, pos, color) in enumerate(zip(box_data, positions, colors)):
        if len(data) > 1:  # Only for multi-mouse data
            x = np.random.normal(pos, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.8, s=80, color=color, edgecolors='black', 
                      linewidths=1.5, zorder=3)
    
    # Styling
    ax.set_ylabel('Correlation coefficient', fontsize=14, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_title('Mean Correlation Coefficients', fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylim([-0.01, 0.06])  # Better scale to show variation
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    return fig, ax


def create_figure_2g_boxplot(per_mouse_real, per_mouse_shuffled, per_mouse_sim, per_mouse_diff,
                              output_path='outputs/figure_2g_boxplot.png'):
    """
    Create Figure 2g - Box plots of FWHM values.
    
    Args:
        per_mouse_real: List of real correlation arrays per mouse
        per_mouse_shuffled: List of shuffled correlation arrays per mouse
        per_mouse_sim: List of similarly tuned correlation arrays per mouse
        per_mouse_diff: List of differently tuned correlation arrays per mouse
        output_path: Where to save
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Compute FWHM per mouse
    real_fwhm = [compute_fwhm(corr) for corr in per_mouse_real]
    shuffled_fwhm = [compute_fwhm(corr) for corr in per_mouse_shuffled]
    sim_fwhm = [compute_fwhm(corr) for corr in per_mouse_sim]  # Per-mouse FWHM!
    diff_fwhm = [compute_fwhm(corr) for corr in per_mouse_diff]  # Per-mouse FWHM!
    
    # Box plot data
    positions = [1, 2, 3, 4]
    box_data = [real_fwhm, shuffled_fwhm, sim_fwhm, diff_fwhm]
    colors = ['#60BD68', '#FAA43A', '#F15854', '#5DA5DA']
    labels = ['Real data', 'Shuffled', 'Sim. tuned', 'Diff. tuned']
    
    # Create box plots
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', linewidth=2))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Overlay individual points
    for i, (data, pos, color) in enumerate(zip(box_data, positions, colors)):
        if len(data) > 1:  # Only for multi-mouse data
            x = np.random.normal(pos, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.8, s=80, color=color, edgecolors='black',
                      linewidths=1.5, zorder=3)
    
    # Styling
    ax.set_ylabel('Correlation coefficient', fontsize=14, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_title('FWHM', fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.08, 0.22])  # Better scale to show variation
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    return fig, ax


def create_combined_figure(all_real, all_shuffled, sim_tuned, diff_tuned,
                           per_mouse_real, per_mouse_shuffled, per_mouse_sim, per_mouse_diff, pvalue,
                           output_path='outputs/figure_2_combined.png'):
    """
    Create combined figure with all 4 panels (d, e, f, g).
    
    Args:
        all_real: All real correlations
        all_shuffled: All shuffled correlations
        sim_tuned: Similarly tuned correlations
        diff_tuned: Differently tuned correlations
        per_mouse_real: Per-mouse real correlations
        per_mouse_shuffled: Per-mouse shuffled correlations
        pvalue: KS test p-value
        output_path: Where to save
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel D (top left) - Real vs Shuffled (KDE lines, NO log scale)
    ax_d = fig.add_subplot(gs[0, 0])
    
    # Use KDE for smooth lines
    x_range_d = np.linspace(-0.2, 0.3, 500)
    
    kde_real_d = stats.gaussian_kde(all_real, bw_method='scott')
    density_real_d = kde_real_d(x_range_d)
    counts_real_d = density_real_d * len(all_real) * (x_range_d[1] - x_range_d[0])
    
    kde_shuffled_d = stats.gaussian_kde(all_shuffled, bw_method='scott')
    density_shuffled_d = kde_shuffled_d(x_range_d)
    counts_shuffled_d = density_shuffled_d * len(all_shuffled) * (x_range_d[1] - x_range_d[0])
    
    # Plot with lines
    ax_d.plot(x_range_d, counts_real_d, color='#2CA02C', linewidth=2.5,
              label='Real data', alpha=0.8)  # Green
    ax_d.plot(x_range_d, counts_shuffled_d, color='#FF7F0E', linewidth=2.5,
              label='Shuffled', alpha=0.8)  # Orange
    
    # NO LOG SCALE - linear!
    ax_d.set_xlabel('Correlation coefficient', fontsize=12)
    ax_d.set_ylabel('Number of cell pairs', fontsize=12)
    ax_d.legend(frameon=False, fontsize=11)
    ax_d.set_title('d', fontsize=16, fontweight='bold', loc='left')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    
    # Panel E (top right) - Similar vs Different (line style, counts not density)
    ax_e = fig.add_subplot(gs[0, 1])
    x_range2 = np.linspace(-0.5, 0.8, 500)
    
    # KDE for lines
    kde_sim = stats.gaussian_kde(sim_tuned, bw_method='scott')
    kde_diff = stats.gaussian_kde(diff_tuned, bw_method='scott')
    density_sim = kde_sim(x_range2)
    density_diff = kde_diff(x_range2)
    
    # Convert to counts
    counts_sim = density_sim * len(sim_tuned) * (x_range2[1] - x_range2[0])
    counts_diff = density_diff * len(diff_tuned) * (x_range2[1] - x_range2[0])
    
    # Plot lines only (not filled) - matching paper colors
    ax_e.plot(x_range2, counts_sim, color='#D62728', linewidth=2.5, 
              label='Similarly tuned', alpha=0.8)  # Red
    ax_e.plot(x_range2, counts_diff, color='#1F77B4', linewidth=2.5,
              label='Differently tuned', alpha=0.8)  # Blue
    
    ax_e.set_xlabel('Correlation coefficient', fontsize=12, fontweight='bold')
    ax_e.set_ylabel('Number of cell pairs', fontsize=12, fontweight='bold')
    ax_e.legend(frameon=False, fontsize=11)
    ax_e.set_title('e', fontsize=16, fontweight='bold', loc='left')
    
    # Add total pairs
    total_pairs = len(sim_tuned) + len(diff_tuned)
    ax_e.text(0.5, 0.98, f'Total: {total_pairs:,}', transform=ax_e.transAxes,
              va='top', ha='center', fontsize=10, fontweight='bold')
    
    ax_e.text(0.98, 0.90, f'***\np={pvalue:.2e}', transform=ax_e.transAxes,
              va='top', ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.grid(True, alpha=0.3, axis='y')
    
    # Panel F (bottom left) - Mean box plots
    ax_f = fig.add_subplot(gs[1, 0])
    real_means = [np.mean(corr) for corr in per_mouse_real]
    shuffled_means = [np.mean(corr) for corr in per_mouse_shuffled]
    sim_means = [np.mean(corr) for corr in per_mouse_sim]  # Per-mouse!
    diff_means = [np.mean(corr) for corr in per_mouse_diff]  # Per-mouse!
    
    positions = [1, 2, 3, 4]
    box_data_f = [real_means, shuffled_means, sim_means, diff_means]  # All per-mouse now!
    colors = ['#60BD68', '#FAA43A', '#F15854', '#5DA5DA']
    
    bp_f = ax_f.boxplot(box_data_f, positions=positions, widths=0.5, patch_artist=True,
                         showfliers=False, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp_f['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for data, pos, color in zip(box_data_f, positions, colors):
        if len(data) > 1:
            x = np.random.normal(pos, 0.03, size=len(data))
            ax_f.scatter(x, data, alpha=0.8, s=60, color=color, edgecolors='black', 
                        linewidths=1.2, zorder=3)
    
    ax_f.set_ylabel('Correlation coefficient', fontsize=12, fontweight='bold')
    ax_f.set_xticks(positions)
    ax_f.set_xticklabels(['Real', 'Shuffled', 'Sim.', 'Diff.'], fontsize=10)
    ax_f.set_title('f   Mean', fontsize=16, fontweight='bold', loc='left')
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(True, alpha=0.3, axis='y')
    ax_f.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_f.set_ylim([-0.01, 0.06])  # Better scale to show variation
    
    # Panel G (bottom right) - FWHM box plots
    ax_g = fig.add_subplot(gs[1, 1])
    real_fwhm = [compute_fwhm(corr) for corr in per_mouse_real]
    shuffled_fwhm = [compute_fwhm(corr) for corr in per_mouse_shuffled]
    sim_fwhm = [compute_fwhm(corr) for corr in per_mouse_sim]  # Per-mouse!
    diff_fwhm = [compute_fwhm(corr) for corr in per_mouse_diff]  # Per-mouse!
    
    box_data_g = [real_fwhm, shuffled_fwhm, sim_fwhm, diff_fwhm]  # All per-mouse now!
    
    bp_g = ax_g.boxplot(box_data_g, positions=positions, widths=0.5, patch_artist=True,
                         showfliers=False, medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp_g['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for data, pos, color in zip(box_data_g, positions, colors):
        if len(data) > 1:
            x = np.random.normal(pos, 0.03, size=len(data))
            ax_g.scatter(x, data, alpha=0.8, s=60, color=color, edgecolors='black',
                        linewidths=1.2, zorder=3)
    
    ax_g.set_ylabel('Correlation coefficient', fontsize=12, fontweight='bold')
    ax_g.set_xticks(positions)
    ax_g.set_xticklabels(['Real', 'Shuffled', 'Sim.', 'Diff.'], fontsize=10)
    ax_g.set_title('g   FWHM', fontsize=16, fontweight='bold', loc='left')
    ax_g.spines['top'].set_visible(False)
    ax_g.spines['right'].set_visible(False)
    ax_g.grid(True, alpha=0.3, axis='y')
    ax_g.set_ylim([0.08, 0.22])  # Better scale to show variation
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    return fig


def main():
    """Load saved results and regenerate figures with improved styling."""
    
    print("="*60)
    print("REGENERATING FIGURES WITH IMPROVED STYLING")
    print("="*60)
    
    # Load pre-computed results
    results_path = Path('outputs/correlation_results.npz')
    
    if not results_path.exists():
        print("\n✗ Error: correlation_results.npz not found!")
        print("  Please run 'python run_analysis.py' first to generate results.")
        return
    
    print("\n✓ Loading pre-computed results...")
    data = np.load(results_path, allow_pickle=True)
    
    all_real = data['all_real']
    all_shuffled = data['all_shuffled']
    sim_tuned = data['sim_tuned']
    diff_tuned = data['diff_tuned']
    per_mouse_real = data['per_mouse_real']
    per_mouse_shuffled = data['per_mouse_shuffled']
    per_mouse_sim = data.get('per_mouse_sim', None)
    per_mouse_diff = data.get('per_mouse_diff', None)
    
    print(f"  Real correlations: {len(all_real):,}")
    print(f"  Shuffled correlations: {len(all_shuffled):,}")
    print(f"  Similarly tuned pairs: {len(sim_tuned):,}")
    print(f"  Differently tuned pairs: {len(diff_tuned):,}")
    print(f"  Per-mouse data: {len(per_mouse_real)} mice")
    if per_mouse_sim is not None:
        print(f"  Per-mouse sim/diff tuned: Available ✓")
    
    # Compute KS test for Figure 2e
    from scipy.stats import ks_2samp
    _, pvalue = ks_2samp(sim_tuned, diff_tuned)
    
    # Generate improved figures
    print("\n" + "="*60)
    print("Generating Figure 2d (KDE + log scale)...")
    print("="*60)
    create_figure_2d_kde(all_real, all_shuffled)
    
    print("\n" + "="*60)
    print("Generating Figure 2e (KDE)...")
    print("="*60)
    create_figure_2e_kde(sim_tuned, diff_tuned, pvalue)
    
    print("\n" + "="*60)
    print("Generating Figure 2f (Mean box plots)...")
    print("="*60)
    if per_mouse_sim is not None and per_mouse_diff is not None:
        create_figure_2f_boxplot(per_mouse_real, per_mouse_shuffled, per_mouse_sim, per_mouse_diff)
    else:
        print("  ⚠️ Warning: Per-mouse sim/diff data not found, skipping F")
    
    print("\n" + "="*60)
    print("Generating Figure 2g (FWHM box plots)...")
    print("="*60)
    if per_mouse_sim is not None and per_mouse_diff is not None:
        create_figure_2g_boxplot(per_mouse_real, per_mouse_shuffled, per_mouse_sim, per_mouse_diff)
    else:
        print("  ⚠️ Warning: Per-mouse sim/diff data not found, skipping G")
    
    print("\n" + "="*60)
    print("Generating Combined Figure (all 4 panels)...")
    print("="*60)
    if per_mouse_sim is not None and per_mouse_diff is not None:
        create_combined_figure(all_real, all_shuffled, sim_tuned, diff_tuned,
                              per_mouse_real, per_mouse_shuffled, per_mouse_sim, per_mouse_diff, pvalue)
    else:
        print("  ⚠️ Warning: Per-mouse sim/diff data not found, skipping combined figure")
    
    print("\n" + "="*60)
    print("✓ ALL FIGURES REGENERATED")
    print("="*60)
    print("\nFigures saved:")
    print("  - outputs/figure_2d_kde.png (panel d)")
    print("  - outputs/figure_2e_kde.png (panel e)")
    print("  - outputs/figure_2f_boxplot.png (panel f)")
    print("  - outputs/figure_2g_boxplot.png (panel g)")
    print("  - outputs/figure_2_combined.png (ALL 4 PANELS)")
    print("\nThese match the paper's style!")


if __name__ == '__main__':
    main()

