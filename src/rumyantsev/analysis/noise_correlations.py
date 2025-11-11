"""
Noise correlation analysis module.

Implements noise correlation computation following Rumyantsev et al. 2020:
- Per-stimulus correlation with mean subtraction
- Averaging across stimuli
- Trial shuffling for null distribution
- Efficient pairwise computation
"""

from itertools import combinations
from typing import Optional

import numpy as np
from tqdm import tqdm


def compute_noise_correlation(
    cell_i_responses: np.ndarray,
    cell_j_responses: np.ndarray,
    stimulus_labels: np.ndarray
) -> float:
    """
    Compute noise correlation between two neurons.
    
    Paper: "To compute correlation coefficients for the noise in the visual 
    responses of a pair of neurons, we first integrated the estimated spike 
    count of each cell between [0.5 s, 2 s] from the start of visual 
    stimulation. After separating the trials for each of the two visual 
    stimuli, we subtracted from each trace the mean stimulus-evoked response 
    of the cell and then calculated the Pearson correlation coefficient, r, 
    for the resulting set of responses from the two cells. We then averaged 
    these noise correlation coefficients over the two stimulus conditions."
    (Methods, page 9)
    
    Args:
        cell_i_responses: Integrated spike counts, shape (n_trials,)
        cell_j_responses: Integrated spike counts, shape (n_trials,)
        stimulus_labels: Stimulus type per trial, shape (n_trials,)
        
    Returns:
        r_noise: Noise correlation coefficient (averaged across stimuli)
        
    Example:
        >>> responses_i = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> responses_j = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> stimuli = np.array([30, 30, 30, 30, -30, -30, -30, -30])
        >>> r = compute_noise_correlation(responses_i, responses_j, stimuli)
        >>> print(f"Correlation: {r:.3f}")
        Correlation: 1.000
    """
    unique_stimuli = np.unique(stimulus_labels)
    correlations = []
    
    for stim in unique_stimuli:
        # Extract trials for this stimulus
        mask = stimulus_labels == stim
        responses_i = cell_i_responses[mask]
        responses_j = cell_j_responses[mask]
        
        # Remove mean (isolate noise)
        noise_i = responses_i - responses_i.mean()
        noise_j = responses_j - responses_j.mean()
        
        # Pearson correlation
        # Handle edge case of zero variance
        if noise_i.std() == 0 or noise_j.std() == 0:
            r = 0.0
        else:
            r = np.corrcoef(noise_i, noise_j)[0, 1]
        
        correlations.append(r)
    
    # Average across stimuli
    return float(np.mean(correlations))


def compute_all_pairwise(
    integrated_responses: np.ndarray,
    stimulus_labels: np.ndarray,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute noise correlations for all neuron pairs.
    
    Paper: "We calculated noise correlations for all possible pairs of 
    neurons in each recording..." Results in 6,946,280 pairs across 5 mice.
    (Methods, page 9)
    
    Args:
        integrated_responses: shape (n_neurons, n_trials)
        stimulus_labels: shape (n_trials,)
        show_progress: Whether to show progress bar
        
    Returns:
        correlations: shape (n_pairs,) where n_pairs = n_neurons * (n_neurons - 1) / 2
        
    Example:
        >>> responses = np.random.randn(100, 200)  # 100 neurons, 200 trials
        >>> stimuli = np.repeat([30, -30], 100)
        >>> corrs = compute_all_pairwise(responses, stimuli)
        >>> print(f"Computed {len(corrs)} pairwise correlations")
        Computed 4950 pairwise correlations
    """
    n_neurons = integrated_responses.shape[0]
    n_pairs = n_neurons * (n_neurons - 1) // 2
    correlations = np.empty(n_pairs)
    
    pairs_iter = combinations(range(n_neurons), 2)
    
    if show_progress:
        pairs_iter = tqdm(
            pairs_iter, 
            total=n_pairs,
            desc="Computing correlations"
        )
    
    for idx, (i, j) in enumerate(pairs_iter):
        correlations[idx] = compute_noise_correlation(
            integrated_responses[i],
            integrated_responses[j],
            stimulus_labels
        )
    
    return correlations


def shuffle_trials(
    integrated_responses: np.ndarray,
    stimulus_labels: np.ndarray,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Shuffle trials independently for each cell to create null distribution.
    
    Paper: "To create trial-shuffled datasets, we randomly permuted the 
    activity traces of each cell across the full set of trials in which the 
    same stimulus was presented, using a different random permutation for 
    each individual cell." (Methods, page 9)
    
    This destroys correlations between cells while preserving:
    - Each cell's response distribution
    - Trial structure within each stimulus
    
    Args:
        integrated_responses: shape (n_neurons, n_trials)
        stimulus_labels: shape (n_trials,)
        random_seed: For reproducibility
        
    Returns:
        shuffled_responses: shape (n_neurons, n_trials)
        
    Example:
        >>> responses = np.random.randn(50, 100)
        >>> stimuli = np.repeat([30, -30], 50)
        >>> shuffled = shuffle_trials(responses, stimuli, random_seed=42)
        >>> # Each cell's trials are independently shuffled within stimulus
    """
    rng = np.random.default_rng(random_seed)
    shuffled = np.empty_like(integrated_responses)
    
    n_neurons = integrated_responses.shape[0]
    unique_stimuli = np.unique(stimulus_labels)
    
    for cell_idx in range(n_neurons):
        for stim in unique_stimuli:
            # Get trials for this stimulus
            mask = stimulus_labels == stim
            stim_responses = integrated_responses[cell_idx, mask].copy()
            
            # Shuffle independently per cell
            rng.shuffle(stim_responses)
            shuffled[cell_idx, mask] = stim_responses
    
    return shuffled

