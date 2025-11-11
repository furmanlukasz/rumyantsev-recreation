"""
Tests for tuning similarity analysis.

Following TDD approach: Tests for classifying neuron pairs based on
tuning similarity (signal correlation) as specified in Rumyantsev et al. 2020.
"""

import numpy as np
import pytest


def test_similarly_tuned_classification():
    """
    Test identification of similarly tuned cell pairs.
    
    Paper: "we compared the statistical distributions of noise correlation 
    coefficients for two different sets of cell pairs, those with positive 
    and those with negative covariance of their mean response to the two 
    stimuli." (Methods, page 9)
    """
    from rumyantsev.analysis.tuning_similarity import classify_tuning

    # Both cells prefer stimulus A (30°)
    mean_A_i, mean_B_i = 10.0, 5.0  # prefers A
    mean_A_j, mean_B_j = 8.0, 3.0   # also prefers A
    
    classification = classify_tuning(mean_A_i, mean_B_i, mean_A_j, mean_B_j)
    assert classification == 'similar'


def test_differently_tuned_classification():
    """
    Test identification of differently tuned cell pairs.
    
    Paper: Pairs with negative covariance of mean responses.
    """
    from rumyantsev.analysis.tuning_similarity import classify_tuning

    # Opposite preferences
    mean_A_i, mean_B_i = 10.0, 5.0  # prefers A
    mean_A_j, mean_B_j = 3.0, 8.0   # prefers B
    
    classification = classify_tuning(mean_A_i, mean_B_i, mean_A_j, mean_B_j)
    assert classification == 'different'


def test_tuning_classification_edge_cases():
    """Test edge cases like equal responses."""
    from rumyantsev.analysis.tuning_similarity import classify_tuning

    # Equal responses to both stimuli (no preference)
    mean_A_i, mean_B_i = 5.0, 5.0
    mean_A_j, mean_B_j = 5.0, 5.0
    
    # Should classify (covariance = 0, but implementation may vary)
    classification = classify_tuning(mean_A_i, mean_B_i, mean_A_j, mean_B_j)
    assert classification in ['similar', 'different']


def test_top_active_cells_selection():
    """
    Test selection of top N% active cells.
    
    Paper: "To visually highlight the differences between the two distributions 
    (Fig. 2e), we also analysed only the most responsive cells, defined as those 
    cells with the top 10% values of √(r_A² + r_B²)" (Methods, page 9)
    """
    from rumyantsev.analysis.tuning_similarity import select_top_active

    # Create mock cell responses
    rng = np.random.default_rng(42)
    n_cells = 100
    responses_A = rng.normal(5, 2, size=(n_cells, 50))  # 50 trials
    responses_B = rng.normal(5, 2, size=(n_cells, 50))
    
    top_cells = select_top_active(responses_A, responses_B, percentile=10)
    
    # Should get 10% of cells
    assert len(top_cells) == 10


def test_top_active_selection_correct_metric():
    """
    Verify that activity metric is sqrt(r_A^2 + r_B^2).
    
    Paper specifies this exact formula for activity.
    """
    from rumyantsev.analysis.tuning_similarity import select_top_active

    # Create cells with known mean responses
    n_cells = 10
    n_trials = 20
    
    # Cell 0: high activity (mean_A=10, mean_B=10)
    # Cell 1-9: low activity (mean_A=1, mean_B=1)
    responses_A = np.ones((n_cells, n_trials))
    responses_B = np.ones((n_cells, n_trials))
    
    responses_A[0, :] = 10  # Cell 0 has high mean
    responses_B[0, :] = 10
    
    # Select top 10% (should be cell 0)
    top_cells = select_top_active(responses_A, responses_B, percentile=10)
    
    assert len(top_cells) == 1
    assert 0 in top_cells


def test_compute_signal_correlation():
    """
    Test signal correlation computation (covariance of means).
    
    This is used to classify similar vs different tuning.
    """
    from rumyantsev.analysis.tuning_similarity import \
        compute_signal_correlation

    # Perfect positive signal correlation
    mean_A_i, mean_B_i = 10.0, 5.0
    mean_A_j, mean_B_j = 10.0, 5.0  # Same tuning curve
    
    signal_corr = compute_signal_correlation(mean_A_i, mean_B_i, mean_A_j, mean_B_j)
    assert signal_corr > 0  # Positive covariance


def test_group_pairs_by_tuning():
    """
    Test grouping all neuron pairs by tuning similarity.
    
    This is needed to create separate distributions for Fig 2e.
    """
    from rumyantsev.analysis.tuning_similarity import group_pairs_by_tuning

    # Create mean responses for 5 cells
    mean_responses_A = np.array([10, 8, 3, 9, 2], dtype=float)
    mean_responses_B = np.array([5, 6, 8, 4, 9], dtype=float)
    
    # Cells 0, 1, 3 prefer A (higher mean_A)
    # Cells 2, 4 prefer B (higher mean_B)
    
    similar_pairs, different_pairs = group_pairs_by_tuning(
        mean_responses_A, mean_responses_B
    )
    
    # Should have some pairs in each category
    assert len(similar_pairs) > 0
    assert len(different_pairs) > 0
    
    # Total pairs should be n*(n-1)/2 = 5*4/2 = 10
    assert len(similar_pairs) + len(different_pairs) == 10


def test_tuning_similarity_with_correlations():
    """
    Integration test: Classify pairs and split correlations.
    
    This tests the full workflow for Figure 2e.
    """
    from rumyantsev.analysis.noise_correlations import compute_all_pairwise
    from rumyantsev.analysis.tuning_similarity import (group_pairs_by_tuning,
                                                       select_top_active)

    # Create mock data
    rng = np.random.default_rng(42)
    n_cells = 20
    n_trials = 40
    
    # Create responses with some structure
    responses = rng.normal(5, 2, size=(n_cells, n_trials))
    stimuli = np.repeat([30, -30], n_trials // 2)
    
    # Compute mean responses per stimulus
    mask_A = stimuli == 30
    mask_B = stimuli == -30
    mean_responses_A = responses[:, mask_A].mean(axis=1)
    mean_responses_B = responses[:, mask_B].mean(axis=1)
    
    # Group pairs by tuning
    similar_pairs, different_pairs = group_pairs_by_tuning(
        mean_responses_A, mean_responses_B
    )
    
    # Compute all correlations
    all_correlations = compute_all_pairwise(responses, stimuli, show_progress=False)
    
    # Should be able to split correlations by tuning
    n_total_pairs = n_cells * (n_cells - 1) // 2
    assert len(all_correlations) == n_total_pairs
    assert len(similar_pairs) + len(different_pairs) == n_total_pairs

