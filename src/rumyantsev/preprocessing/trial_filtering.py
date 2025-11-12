"""
Preprocessing module for trial filtering and time window integration.

Implements data preprocessing steps from Rumyantsev et al. 2020:
- Spike detection thresholding
- Time window integration [0.5s, 2.0s]
- Trial count validation (data appears pre-filtered for locomotion)
- Reshaping for correlation analysis
"""

from typing import Optional, Tuple

import numpy as np
import polars as pl


def apply_spike_threshold(
    data: pl.DataFrame,
    threshold: float = 0.5
) -> pl.DataFrame:
    """
    Apply spike detection threshold to amplitude values.
    
    Sets amplitude values below threshold to zero. This removes low-amplitude
    noise events that shouldn't count as spikes.
    
    Paper: Extended Data Fig 4c mentions threshold of 0.5 for spike counts.
    May also apply to main correlation analysis (Fig 2d/2e).
    
    Args:
        data: Neural data with 'amplitude' column
        threshold: Minimum amplitude to count as spike (default: 0.5)
        
    Returns:
        Data with thresholded amplitudes (values < threshold set to 0)
        
    Example:
        >>> data = apply_spike_threshold(mouse_data, threshold=0.5)
        >>> # Amplitudes < 0.5 are now 0
    """
    return data.with_columns(
        pl.when(pl.col('amplitude') >= threshold)
        .then(pl.col('amplitude'))
        .otherwise(0.0)
        .alias('amplitude')
    )


def integrate_time_window(
    data: pl.DataFrame, 
    start_bin: int, 
    end_bin: int
) -> pl.DataFrame:
    """
    Integrate amplitudes over specified time window (bin indices).
    
    Paper: "we integrated the estimated spike count of each cell between 
    [0.5 s, 2 s] from the start of visual stimulation" (Methods, page 9)
    
    Implementation: Time bins 0-13 at 0.275s each:
      - [0.5s, 2.0s] → bins [2, 7] (6 bins, 1.65s duration)
    
    Args:
        data: Neural data with 'sample_idx' and 'amplitude' columns
        start_bin: Start bin index (inclusive)
        end_bin: End bin index (inclusive)
        
    Returns:
        DataFrame with integrated amplitudes per cell per trial
        
    Example:
        >>> integrated = integrate_time_window(data, 2, 7)
        >>> print(integrated.columns)
        ['mouse_id', 'cell_idx', 'trial_idx', 'behavior', 'integrated_amplitude']
    """
    return (
        data
        .filter(
            (pl.col('sample_idx') >= start_bin) & 
            (pl.col('sample_idx') <= end_bin)
        )
        .group_by(['mouse_id', 'cell_idx', 'trial_idx', 'behavior'])
        .agg(pl.col('amplitude').sum().alias('integrated_amplitude'))
    )


def validate_trial_counts(
    data: pl.DataFrame, 
    expected_min: int, 
    expected_max: int
) -> None:
    """
    Validate that trial counts match expected post-filtering range.
    
    NOTE: Data appears to be pre-filtered (no locomotion_speed column).
    This function verifies trial counts match the expected range from the paper.
    
    Paper: "217-332 trials per stimulus after filtering" (Methods, page 7)
    
    Args:
        data: Neural data for single mouse
        expected_min: Minimum expected trials per stimulus
        expected_max: Maximum expected trials per stimulus
        
    Raises:
        ValueError: If trial counts out of expected range
        
    Example:
        >>> validate_trial_counts(mouse_data, 217, 332)
        # No error if counts are valid
    """
    trials_per_stim = (
        data
        .group_by('behavior')
        .agg(pl.col('trial_idx').n_unique().alias('n_trials'))
    )
    
    for row in trials_per_stim.iter_rows(named=True):
        n_trials = row['n_trials']
        if not (expected_min <= n_trials <= expected_max):
            raise ValueError(
                f"Stimulus {row['behavior']}: {n_trials} trials "
                f"(expected {expected_min}-{expected_max})"
            )


def extract_stimulus_labels(data: pl.DataFrame) -> np.ndarray:
    """
    Extract stimulus labels per trial in trial order.
    
    Paper: "Two drifting grating stimuli (±30° from vertical)" (Methods, page 6)
    
    Args:
        data: Neural data with 'trial_idx' and 'behavior' columns
        
    Returns:
        Array of stimulus codes (30, -30) for each trial, shape (n_trials,)
        
    Example:
        >>> labels = extract_stimulus_labels(data)
        >>> print(labels[:5])
        [30, -30, 30, -30, 30]
    """
    return (
        data
        .select(['trial_idx', 'behavior'])
        .unique()
        .sort('trial_idx')['behavior']
        .to_numpy()
    )


def reshape_to_matrix(integrated_data: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape integrated data to (n_cells, n_trials) matrix format.
    
    This format is required for pairwise correlation analysis.
    
    Args:
        integrated_data: Output from integrate_time_window()
        
    Returns:
        response_matrix: shape (n_cells, n_trials) - integrated spike counts
        stimulus_labels: shape (n_trials,) - stimulus per trial (30 or -30)
        
    Example:
        >>> responses, labels = reshape_to_matrix(integrated)
        >>> print(responses.shape)
        (100, 50)  # 100 cells, 50 trials
        >>> print(labels.shape)
        (50,)  # One label per trial
    """
    # Extract stimulus labels first
    stimulus_labels = extract_stimulus_labels(integrated_data)
    
    # Pivot to matrix: rows = cells, columns = trials
    matrix_data = (
        integrated_data
        .select(['cell_idx', 'trial_idx', 'integrated_amplitude'])
        .pivot(
            values='integrated_amplitude',
            index='cell_idx',
            columns='trial_idx'
        )
        .sort('cell_idx')
    )
    
    # Convert to numpy (drop cell_idx column)
    response_matrix = matrix_data.select(pl.all().exclude('cell_idx')).to_numpy()
    
    return response_matrix, stimulus_labels

