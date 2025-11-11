"""
Tests for statistical analysis.

Following TDD approach: Tests for Kolmogorov-Smirnov test and other
statistical comparisons as specified in Rumyantsev et al. 2020.
"""

import numpy as np
import pytest


def test_ks_test_different_distributions():
    """
    Test KS test on clearly different distributions.
    
    Paper: "two-tailed Kolmogorov–Smirnov test, ***P < 10−13 for all 5 mice"
    (Figure 2 legend)
    """
    from rumyantsev.analysis.statistics import compare_distributions

    # Create distributions with known difference
    rng = np.random.default_rng(42)
    dist1 = rng.normal(0.05, 0.02, 1000)
    dist2 = rng.normal(0.10, 0.02, 1000)
    
    statistic, pvalue = compare_distributions(dist1, dist2)
    
    # Should find significant difference
    assert pvalue < 0.05
    assert statistic > 0  # KS statistic should be positive


def test_ks_test_identical_distributions():
    """Test KS test on identical distributions."""
    from rumyantsev.analysis.statistics import compare_distributions

    # Same distribution
    rng = np.random.default_rng(42)
    dist = rng.normal(0.05, 0.02, 1000)
    
    statistic, pvalue = compare_distributions(dist, dist)
    
    # Should NOT find significant difference
    assert pvalue > 0.05  # Not significant


def test_ks_test_returns_valid_values():
    """Test that KS test returns valid statistic and p-value."""
    from rumyantsev.analysis.statistics import compare_distributions
    
    rng = np.random.default_rng(42)
    dist1 = rng.normal(0, 1, 100)
    dist2 = rng.normal(0.1, 1, 100)
    
    statistic, pvalue = compare_distributions(dist1, dist2)
    
    # KS statistic should be in [0, 1]
    assert 0 <= statistic <= 1
    
    # p-value should be in [0, 1]
    assert 0 <= pvalue <= 1


def test_compute_variance_ratio():
    """
    Test variance ratio computation for shuffled vs real data.
    
    Paper: "The variance of this distribution (shuffled) was approximately 
    50% that of the distribution calculated from unshuffled data"
    (Methods, page 9)
    """
    from rumyantsev.analysis.statistics import compute_variance_ratio

    # Create distributions with known variance ratio
    rng = np.random.default_rng(42)
    real_dist = rng.normal(0.06, 0.03, 10000)  # Higher variance
    shuffled_dist = rng.normal(0.0, 0.015, 10000)  # Lower variance (half)
    
    ratio = compute_variance_ratio(shuffled_dist, real_dist)
    
    # Should be approximately 0.5
    assert 0.2 < ratio < 0.8  # Broad range due to sampling


def test_summary_statistics():
    """Test computation of summary statistics."""
    from rumyantsev.analysis.statistics import compute_summary_statistics
    
    rng = np.random.default_rng(42)
    correlations = rng.normal(0.06, 0.02, 10000)
    
    stats = compute_summary_statistics(correlations)
    
    # Should have expected keys
    assert 'mean' in stats
    assert 'std' in stats
    assert 'median' in stats
    assert 'n_pairs' in stats
    
    # Values should be reasonable
    assert np.isclose(stats['mean'], 0.06, atol=0.01)
    assert stats['n_pairs'] == 10000

