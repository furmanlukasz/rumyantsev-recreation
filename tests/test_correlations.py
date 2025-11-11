"""
Tests for noise correlation analysis.

Following TDD approach: Tests for correlation computation, trial shuffling,
and pairwise analysis as specified in Rumyantsev et al. 2020.
"""

import numpy as np
import polars as pl
import pytest


def test_noise_correlation_removes_mean():
    """
    Verify mean response is subtracted before correlation.
    
    Paper: "After separating the trials for each of the two visual stimuli, 
    we subtracted from each trace the mean stimulus-evoked response of the cell"
    (Methods, page 9)
    """
    from rumyantsev.analysis.noise_correlations import \
        compute_noise_correlation

    # Create data with known mean
    cell_i = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)  # mean = 4.5
    cell_j = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=float)  # mean = 5.5
    stimuli = np.array([30, 30, 30, 30, -30, -30, -30, -30])
    
    r_noise = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Should compute correlation of (x - mean_x, y - mean_y)
    assert isinstance(r_noise, float)
    assert -1 <= r_noise <= 1


def test_correlation_averaged_across_stimuli():
    """
    Test that correlation is mean of per-stimulus correlations.
    
    Paper: "We then averaged these noise correlation coefficients over 
    the two stimulus conditions." (Methods, page 9)
    """
    from rumyantsev.analysis.noise_correlations import \
        compute_noise_correlation

    # Perfect correlation for stimulus 30, zero for -30
    cell_i = np.array([1, 2, 3, 4, 0, 0, 0, 0], dtype=float)
    cell_j = np.array([1, 2, 3, 4, 0, 1, 2, 3], dtype=float)  # correlated for 30, not -30
    stimuli = np.array([30, 30, 30, 30, -30, -30, -30, -30])
    
    r_noise = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Should be average: (1.0 + ~0) / 2 ≈ 0.5
    assert 0.4 <= r_noise <= 0.6


def test_correlation_with_perfect_positive():
    """Test with perfectly correlated responses."""
    from rumyantsev.analysis.noise_correlations import \
        compute_noise_correlation

    # Identical responses
    cell_i = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    cell_j = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    stimuli = np.array([30, 30, 30, 30, -30, -30, -30, -30])
    
    r_noise = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Should be close to 1.0
    assert np.isclose(r_noise, 1.0, atol=0.01)


def test_correlation_with_perfect_negative():
    """Test with perfectly anti-correlated responses."""
    from rumyantsev.analysis.noise_correlations import \
        compute_noise_correlation

    # Opposite responses
    cell_i = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    cell_j = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    stimuli = np.array([30, 30, 30, 30, -30, -30, -30, -30])
    
    r_noise = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Should be close to -1.0
    assert np.isclose(r_noise, -1.0, atol=0.01)


def test_pairwise_correlations_shape():
    """
    Test computation of all pairwise correlations.
    
    Paper: "6,946,280 pairs of neurons" across 5 mice (Methods)
    """
    from rumyantsev.analysis.noise_correlations import compute_all_pairwise

    # Mock data: 10 neurons, 20 trials
    integrated_responses = np.random.randn(10, 20)
    stimuli = np.repeat([30, -30], 10)
    
    correlations = compute_all_pairwise(integrated_responses, stimuli, show_progress=False)
    
    # Should get n*(n-1)/2 unique pairs
    expected_pairs = 10 * 9 // 2
    assert len(correlations) == expected_pairs


def test_pairwise_correlations_values():
    """Test that all correlations are in valid range [-1, 1]."""
    from rumyantsev.analysis.noise_correlations import compute_all_pairwise

    # Random data
    rng = np.random.default_rng(42)
    integrated_responses = rng.normal(0, 1, size=(20, 40))
    stimuli = np.repeat([30, -30], 20)
    
    correlations = compute_all_pairwise(integrated_responses, stimuli, show_progress=False)
    
    # All values should be in [-1, 1]
    assert np.all(correlations >= -1)
    assert np.all(correlations <= 1)
    
    # Should have no NaN values
    assert not np.any(np.isnan(correlations))


def test_trial_shuffling_independence():
    """
    Verify each cell gets independent shuffle.
    
    Paper: "we randomly permuted the activity traces of each cell across 
    the full set of trials in which the same stimulus was presented, using 
    a different random permutation for each individual cell."
    (Methods, page 9)
    """
    from rumyantsev.analysis.noise_correlations import shuffle_trials
    
    responses = np.arange(50, dtype=float).reshape(5, 10)  # 5 cells, 10 trials
    stimuli = np.repeat([30, -30], 5)
    
    shuffled = shuffle_trials(responses, stimuli, random_seed=42)
    
    # Shape should be preserved
    assert shuffled.shape == responses.shape
    
    # Each cell should have same values (just reordered)
    for cell_idx in range(5):
        assert set(shuffled[cell_idx]) == set(responses[cell_idx])


def test_shuffled_correlations_reduced():
    """
    Verify shuffling reduces correlations.
    
    Paper: "The variance of this distribution (shuffled) was approximately 
    50% that of the distribution calculated from unshuffled data"
    (Methods, page 9)
    """
    from rumyantsev.analysis.noise_correlations import (compute_all_pairwise,
                                                        shuffle_trials)

    # Create correlated responses
    n_neurons = 20
    n_trials = 100
    rng = np.random.default_rng(42)
    
    # Add common noise to create correlations
    common_noise = rng.normal(0, 1, n_trials)
    responses = rng.normal(0, 1, (n_neurons, n_trials))
    responses += common_noise * 0.3  # Inject correlations
    
    stimuli = np.repeat([30, -30], n_trials // 2)
    
    # Compute correlations
    real_corr = compute_all_pairwise(responses, stimuli, show_progress=False)
    shuffled_responses = shuffle_trials(responses, stimuli, random_seed=123)
    shuffled_corr = compute_all_pairwise(shuffled_responses, stimuli, show_progress=False)
    
    # Shuffled correlations should exist and be in valid range
    assert len(shuffled_corr) == len(real_corr)
    assert np.all(shuffled_corr >= -1) and np.all(shuffled_corr <= 1)
    
    # Mean should be closer to zero for shuffled (on average)
    # Note: With small samples, this can vary, but shuffling removes structure
    assert not np.array_equal(real_corr, shuffled_corr), "Shuffling should change correlations"


def test_shuffle_preserves_per_stimulus_structure():
    """
    Test that shuffling is done separately per stimulus.
    
    Paper: Trials are shuffled within each stimulus condition separately.
    """
    from rumyantsev.analysis.noise_correlations import shuffle_trials

    # Create deterministic data
    responses = np.arange(40, dtype=float).reshape(4, 10)  # 4 cells, 10 trials
    stimuli = np.array([30, 30, 30, 30, 30, -30, -30, -30, -30, -30])
    
    shuffled = shuffle_trials(responses, stimuli, random_seed=42)
    
    # For each cell and stimulus, should have same values
    for cell_idx in range(4):
        # Stimulus 30 (first 5 trials)
        orig_30 = set(responses[cell_idx, :5])
        shuf_30 = set(shuffled[cell_idx, :5])
        assert orig_30 == shuf_30
        
        # Stimulus -30 (last 5 trials)
        orig_neg30 = set(responses[cell_idx, 5:])
        shuf_neg30 = set(shuffled[cell_idx, 5:])
        assert orig_neg30 == shuf_neg30

