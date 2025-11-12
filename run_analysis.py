#!/usr/bin/env python
"""
Rumyantsev et al. 2020 - Figure 2d/2e Recreation Script

Complete analysis pipeline for recreating Figure 2d and 2e.
This script can be run directly or converted to a Jupyter notebook.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from rumyantsev.analysis.noise_correlations import (compute_all_pairwise,
                                                    compute_noise_correlation,
                                                    shuffle_trials)
from rumyantsev.analysis.statistics import (compare_distributions,
                                            compute_summary_statistics,
                                            compute_variance_ratio)
from rumyantsev.analysis.tuning_similarity import (group_pairs_by_tuning,
                                                   select_top_active)
from rumyantsev.data.loader import DataLoader
from rumyantsev.preprocessing.trial_filtering import (apply_spike_threshold,
                                                      integrate_time_window,
                                                      reshape_to_matrix,
                                                      validate_trial_counts)
from rumyantsev.visualization.figure_2 import (create_figure_2d,
                                               create_figure_2e)


def main():
    """Run complete analysis pipeline."""
    
    print("="*60)
    print("RUMYANTSEV ET AL. 2020 - FIGURE 2D/2E RECREATION")
    print("="*60)
    
    # Load configuration
    config_path = Path('config/analysis_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("\n✓ Configuration loaded")
    
    # Show integration method
    integration_method = config['preprocessing'].get('integration_method', 'discrete_conservative')
    method_config = config['preprocessing']['methods'][integration_method]
    print(f"\n📊 Integration Method: {integration_method}")
    print(f"   Time window: bins [{method_config['time_window_start_bin']}, {method_config['time_window_end_bin']}]")
    print(f"   Actual time: [{method_config['time_window_start_bin']*0.275:.3f}s, {method_config['time_window_end_bin']*0.275:.3f}s]")
    print(f"   Duration: {(method_config['time_window_end_bin'] - method_config['time_window_start_bin'])*0.275:.3f}s")
    
    # Show spike threshold configuration
    spike_config = config['preprocessing'].get('spike_threshold', {})
    spike_enabled = spike_config.get('enabled', False)
    spike_value = spike_config.get('value', 0.5)
    print(f"\n🎯 Spike Threshold: {'ENABLED' if spike_enabled else 'DISABLED'}")
    if spike_enabled:
        print(f"   Threshold value: {spike_value}")
        print(f"   Effect: Amplitudes < {spike_value} set to 0")
    
    # Load data
    data_path = Path('coding_fidelity_bounds.dataset.parquet')
    loader = DataLoader(data_path)
    
    print(f"\n✓ Data loaded:")
    print(f"  Mice: {loader.n_mice}")
    print(f"  Neurons: {loader.count_total_cells()}")
    
    # Validate
    assert loader.n_mice == config['data']['expected_n_mice']
    assert loader.count_total_cells() == config['data']['expected_total_cells']
    print("  ✓ Validation passed")
    
    # Process each mouse
    results = {}
    mouse_ids = loader.data['mouse_id'].unique().sort().to_list()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {len(mouse_ids)} MICE")
    print(f"{'='*60}\n")
    
    for mouse_id in mouse_ids:
        print(f"Processing {mouse_id}...")
        
        # Extract mouse data
        mouse_data = loader.get_mouse_data(mouse_id)
        
        # Apply spike threshold if enabled
        spike_config = config['preprocessing'].get('spike_threshold', {})
        if spike_config.get('enabled', False):
            spike_value = spike_config.get('value', 0.5)
            print(f"  Applying spike threshold: {spike_value}")
            mouse_data = apply_spike_threshold(mouse_data, spike_value)
        
        # Validate trial counts
        try:
            validate_trial_counts(
                mouse_data,
                config['preprocessing']['expected_trials_per_stimulus_min'],
                config['preprocessing']['expected_trials_per_stimulus_max']
            )
        except ValueError as e:
            print(f"  ⚠ {e}")
        
        # Get integration method from config
        integration_method = config['preprocessing'].get('integration_method', 'discrete_conservative')
        method_config = config['preprocessing']['methods'][integration_method]
        
        print(f"  Integration method: {integration_method}")
        print(f"    Bins [{method_config['time_window_start_bin']}, {method_config['time_window_end_bin']}] = " +
              f"[{method_config['time_window_start_bin']*0.275:.3f}s, {method_config['time_window_end_bin']*0.275:.3f}s]")
        
        # Integrate time window
        integrated = integrate_time_window(
            mouse_data,
            method_config['time_window_start_bin'],
            method_config['time_window_end_bin']
        )
        
        # Reshape
        response_matrix, stimulus_labels = reshape_to_matrix(integrated)
        n_cells, n_trials = response_matrix.shape
        n_pairs = n_cells * (n_cells - 1) // 2
        
        print(f"  Cells: {n_cells}, Trials: {n_trials}, Pairs: {n_pairs:,}")
        
        # Compute correlations
        print("  Computing real correlations...")
        real_corr = compute_all_pairwise(response_matrix, stimulus_labels, show_progress=True)
        print("  Computing shuffled correlations...")
        shuffled_responses = shuffle_trials(response_matrix, stimulus_labels, random_seed=42)
        shuffled_corr = compute_all_pairwise(shuffled_responses, stimulus_labels, show_progress=True)
        
        results[mouse_id] = {
            'real_correlations': real_corr,
            'shuffled_correlations': shuffled_corr,
            'response_matrix': response_matrix,
            'stimulus_labels': stimulus_labels
        }
        
        print(f"  Real: μ={np.mean(real_corr):.4f}, Shuffled: μ={np.mean(shuffled_corr):.4f}\n")
    
    # Generate Figure 2d
    print(f"{'='*60}")
    print("FIGURE 2D: NOISE CORRELATION DISTRIBUTION")
    print(f"{'='*60}\n")
    
    all_real = np.concatenate([r['real_correlations'] for r in results.values()])
    all_shuffled = np.concatenate([r['shuffled_correlations'] for r in results.values()])
    
    fig_2d, _ = create_figure_2d(all_real, all_shuffled)
    Path('outputs').mkdir(exist_ok=True)
    fig_2d.savefig('outputs/figure_2d_recreation.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2d saved")
    
    # Stats
    real_stats = compute_summary_statistics(all_real)
    variance_ratio = compute_variance_ratio(all_shuffled, all_real)
    
    print(f"\nValidation:")
    print(f"  Mean correlation: {real_stats['mean']:.4f} (expected: 0.06)")
    print(f"  Total pairs: {real_stats['n_pairs']:,} (expected: ~6.95M)")
    print(f"  Variance ratio: {variance_ratio:.2f} (expected: ~0.5)")
    
    # Generate Figure 2e
    print(f"\n{'='*60}")
    print("FIGURE 2E: TUNING SIMILARITY ANALYSIS")
    print(f"{'='*60}\n")
    
    all_sim = []
    all_diff = []
    
    # IMPORTANT: Track per-mouse sim/diff for box plots
    per_mouse_sim = []
    per_mouse_diff = []
    
    for mouse_id, mouse_results in results.items():
        responses = mouse_results['response_matrix']
        stimuli = mouse_results['stimulus_labels']
        
        # Split by stimulus
        mask_A = stimuli == 30
        mask_B = stimuli == -30
        responses_A = responses[:, mask_A]
        responses_B = responses[:, mask_B]
        
        # Top 10% active
        top_indices = select_top_active(responses_A, responses_B, percentile=10)
        mean_A = responses_A.mean(axis=1)
        mean_B = responses_B.mean(axis=1)
        
        # Group pairs
        similar_pairs, different_pairs = group_pairs_by_tuning(
            mean_A[top_indices],
            mean_B[top_indices]
        )
        
        # Compute correlations per mouse
        mouse_sim = []
        mouse_diff = []
        
        for i, j in similar_pairs:
            orig_i, orig_j = top_indices[i], top_indices[j]
            r = compute_noise_correlation(responses[orig_i], responses[orig_j], stimuli)
            all_sim.append(r)
            mouse_sim.append(r)
        
        for i, j in different_pairs:
            orig_i, orig_j = top_indices[i], top_indices[j]
            r = compute_noise_correlation(responses[orig_i], responses[orig_j], stimuli)
            all_diff.append(r)
            mouse_diff.append(r)
        
        # Store per-mouse arrays
        per_mouse_sim.append(np.array(mouse_sim))
        per_mouse_diff.append(np.array(mouse_diff))
    
    sim_arr = np.array(all_sim)
    diff_arr = np.array(all_diff)
    
    # KS test
    ks_stat, ks_pvalue = compare_distributions(sim_arr, diff_arr)
    
    fig_2e, _ = create_figure_2e(sim_arr, diff_arr, ks_pvalue)
    fig_2e.savefig('outputs/figure_2e_recreation.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2e saved")
    
    print(f"\nKS Test:")
    print(f"  p-value: {ks_pvalue:.2e} (expected: < 1.3e-6)")
    print(f"  Status: {'✓ PASS' if ks_pvalue < 1.3e-6 else '✗ FAIL'}")
    
    # Summary
    summary = {
        'total_mice': loader.n_mice,
        'total_neurons': loader.count_total_cells(),
        'total_pairs': int(real_stats['n_pairs']),
        'mean_noise_correlation': float(real_stats['mean']),
        'std_noise_correlation': float(real_stats['std']),
        'shuffled_variance_ratio': float(variance_ratio),
        'mean_sim_tuned': float(np.mean(sim_arr)),
        'mean_diff_tuned': float(np.mean(diff_arr)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue)
    }
    
    with open('outputs/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save intermediate results for re-plotting without re-computing
    print("\n  Saving intermediate results...")
    
    # Extract per-mouse results for box plots (F & G)
    per_mouse_real = [results[m]['real_correlations'] for m in mouse_ids]
    per_mouse_shuffled = [results[m]['shuffled_correlations'] for m in mouse_ids]
    
    np.savez_compressed(
        'outputs/correlation_results.npz',
        all_real=all_real,
        all_shuffled=all_shuffled,
        sim_tuned=sim_arr,
        diff_tuned=diff_arr,
        mouse_ids=mouse_ids,
        per_mouse_real=np.array(per_mouse_real, dtype=object),
        per_mouse_shuffled=np.array(per_mouse_shuffled, dtype=object),
        per_mouse_sim=np.array(per_mouse_sim, dtype=object),
        per_mouse_diff=np.array(per_mouse_diff, dtype=object)
    )
    
    print(f"\n{'='*60}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    print("Outputs saved to:")
    print("  - outputs/figure_2d_recreation.png")
    print("  - outputs/figure_2e_recreation.png")
    print("  - outputs/summary_statistics.json")
    print("  - outputs/correlation_results.npz (for re-plotting)")


if __name__ == '__main__':
    main()

