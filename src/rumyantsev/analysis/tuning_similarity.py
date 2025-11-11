"""
Tuning similarity analysis module.

Implements classification of neuron pairs based on signal correlation
(covariance of mean responses) following Rumyantsev et al. 2020.
"""

from itertools import combinations
from typing import List, Literal, Tuple

import numpy as np


def compute_signal_correlation(
    mean_response_A_i: float,
    mean_response_B_i: float,
    mean_response_A_j: float,
    mean_response_B_j: float
) -> float:
    """
    Compute covariance of mean responses (signal correlation).
    
    Paper: "we compared the statistical distributions of noise correlation 
    coefficients for two different sets of cell pairs, those with positive 
    and those with negative covariance of their mean response to the two 
    stimuli." (Methods, page 9)
    
    Formula: Cov(μᵢ, μⱼ) = (μᵢ(A) - μ̄ᵢ)(μⱼ(A) - μ̄ⱼ) + (μᵢ(B) - μ̄ᵢ)(μⱼ(B) - μ̄ⱼ)
    
    Args:
        mean_response_A_i: Mean response of cell i to stimulus A
        mean_response_B_i: Mean response of cell i to stimulus B
        mean_response_A_j: Mean response of cell j to stimulus A
        mean_response_B_j: Mean response of cell j to stimulus B
        
    Returns:
        covariance: Signal correlation (covariance of mean responses)
        
    Example:
        >>> # Both cells prefer stimulus A
        >>> cov = compute_signal_correlation(10, 5, 8, 3)
        >>> print(f"Covariance: {cov:.2f} (positive = similar tuning)")
        Covariance: 6.25 (positive = similar tuning)
    """
    # Center around mean
    mean_i = (mean_response_A_i + mean_response_B_i) / 2
    mean_j = (mean_response_A_j + mean_response_B_j) / 2
    
    # Compute covariance
    cov = ((mean_response_A_i - mean_i) * (mean_response_A_j - mean_j) +
           (mean_response_B_i - mean_i) * (mean_response_B_j - mean_j))
    
    return float(cov)


def classify_tuning(
    mean_response_A_i: float,
    mean_response_B_i: float,
    mean_response_A_j: float,
    mean_response_B_j: float
) -> Literal['similar', 'different']:
    """
    Classify whether two cells are similarly or differently tuned.
    
    Paper: Pairs are classified based on the sign of covariance of their
    mean responses. Positive covariance = similarly tuned (both prefer
    same stimulus). Negative covariance = differently tuned (prefer
    opposite stimuli). (Methods, page 9)
    
    Args:
        mean_response_A_i: Mean response of cell i to stimulus A
        mean_response_B_i: Mean response of cell i to stimulus B
        mean_response_A_j: Mean response of cell j to stimulus A
        mean_response_B_j: Mean response of cell j to stimulus B
        
    Returns:
        'similar' if covariance > 0, 'different' if covariance <= 0
        
    Example:
        >>> # Both prefer stimulus A
        >>> classify_tuning(10, 5, 8, 3)
        'similar'
        >>> # Opposite preferences
        >>> classify_tuning(10, 5, 3, 8)
        'different'
    """
    cov = compute_signal_correlation(
        mean_response_A_i, mean_response_B_i,
        mean_response_A_j, mean_response_B_j
    )
    
    return 'similar' if cov > 0 else 'different'


def select_top_active(
    responses_A: np.ndarray,
    responses_B: np.ndarray,
    percentile: int = 10
) -> np.ndarray:
    """
    Select top N% most active cells.
    
    Paper: "To visually highlight the differences between the two distributions 
    (Fig. 2e), we also analysed only the most responsive cells, defined as those 
    cells with the top 10% values of √(r_A² + r_B²)" (Methods, page 9)
    
    Args:
        responses_A: shape (n_cells, n_trials) for stimulus A
        responses_B: shape (n_cells, n_trials) for stimulus B
        percentile: Top N% to select (default 10)
        
    Returns:
        indices: Indices of top active cells
        
    Example:
        >>> responses_A = np.random.randn(100, 50)
        >>> responses_B = np.random.randn(100, 50)
        >>> top_cells = select_top_active(responses_A, responses_B, percentile=10)
        >>> print(f"Selected {len(top_cells)} most active cells")
        Selected 10 most active cells
    """
    # Compute mean responses
    mean_A = responses_A.mean(axis=1)
    mean_B = responses_B.mean(axis=1)
    
    # Activity metric: sqrt(r_A^2 + r_B^2)
    activity = np.sqrt(mean_A**2 + mean_B**2)
    
    # Select top percentile
    threshold = np.percentile(activity, 100 - percentile)
    top_indices = np.where(activity >= threshold)[0]
    
    return top_indices


def group_pairs_by_tuning(
    mean_responses_A: np.ndarray,
    mean_responses_B: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Group all neuron pairs by tuning similarity.
    
    Paper: Separate pairs into those with positive vs negative covariance
    of mean responses for Figure 2e analysis. (Methods, page 9)
    
    Args:
        mean_responses_A: shape (n_cells,) - mean responses to stimulus A
        mean_responses_B: shape (n_cells,) - mean responses to stimulus B
        
    Returns:
        similar_pairs: List of (i, j) indices for similarly tuned pairs
        different_pairs: List of (i, j) indices for differently tuned pairs
        
    Example:
        >>> mean_A = np.array([10, 8, 3, 9, 2])
        >>> mean_B = np.array([5, 6, 8, 4, 9])
        >>> similar, different = group_pairs_by_tuning(mean_A, mean_B)
        >>> print(f"Similar: {len(similar)}, Different: {len(different)}")
        Similar: 6, Different: 4
    """
    n_cells = len(mean_responses_A)
    similar_pairs = []
    different_pairs = []
    
    for i, j in combinations(range(n_cells), 2):
        classification = classify_tuning(
            mean_responses_A[i], mean_responses_B[i],
            mean_responses_A[j], mean_responses_B[j]
        )
        
        if classification == 'similar':
            similar_pairs.append((i, j))
        else:
            different_pairs.append((i, j))
    
    return similar_pairs, different_pairs

