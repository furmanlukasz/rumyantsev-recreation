"""
Statistical analysis module.

Implements statistical tests and summary computations for noise correlation
analysis following Rumyantsev et al. 2020.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.stats import ks_2samp


def compare_distributions(
    dist1: np.ndarray,
    dist2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform two-sample Kolmogorov-Smirnov test.
    
    Paper: "two-tailed Kolmogorov–Smirnov test, ***P < 10−13 for all 5 mice"
    (Figure 2 legend)
    
    The KS test compares the cumulative distributions of two samples to
    determine if they come from different distributions.
    
    Args:
        dist1: First distribution of correlation coefficients
        dist2: Second distribution of correlation coefficients
        
    Returns:
        statistic: KS test statistic (D = sup|F1(x) - F2(x)|)
        pvalue: Two-tailed p-value
        
    Example:
        >>> real_corr = np.array([0.05, 0.06, 0.07, ...])
        >>> shuffled_corr = np.array([0.01, 0.02, 0.00, ...])
        >>> stat, pval = compare_distributions(real_corr, shuffled_corr)
        >>> print(f"KS test: D={stat:.4f}, p={pval:.2e}")
        KS test: D=0.2543, p=1.23e-15
    """
    statistic, pvalue = ks_2samp(dist1, dist2)
    return float(statistic), float(pvalue)


def compute_variance_ratio(
    shuffled_correlations: np.ndarray,
    real_correlations: np.ndarray
) -> float:
    """
    Compute ratio of shuffled to real variance.
    
    Paper: "The variance of this distribution (shuffled) was approximately 
    50% that of the distribution calculated from unshuffled data"
    (Methods, page 9)
    
    Args:
        shuffled_correlations: Correlations from trial-shuffled data
        real_correlations: Correlations from real data
        
    Returns:
        ratio: Variance(shuffled) / Variance(real)
        
    Example:
        >>> ratio = compute_variance_ratio(shuffled, real)
        >>> print(f"Variance ratio: {ratio:.2f} (expected ~0.5)")
        Variance ratio: 0.48 (expected ~0.5)
    """
    var_shuffled = np.var(shuffled_correlations)
    var_real = np.var(real_correlations)
    
    return float(var_shuffled / var_real)


def compute_summary_statistics(correlations: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a distribution of correlations.
    
    Args:
        correlations: Array of correlation coefficients
        
    Returns:
        Dictionary with summary statistics:
        - mean: Mean correlation
        - std: Standard deviation
        - median: Median correlation
        - n_pairs: Number of pairs
        
    Example:
        >>> stats = compute_summary_statistics(correlations)
        >>> print(f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        Mean: 0.060 ± 0.025
    """
    return {
        'mean': float(np.mean(correlations)),
        'std': float(np.std(correlations)),
        'median': float(np.median(correlations)),
        'n_pairs': int(len(correlations))
    }

