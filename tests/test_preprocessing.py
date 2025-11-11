"""
Tests for preprocessing module.

Following TDD approach: Write tests first, then implementation.
Tests validation of trial filtering and time window integration.
"""

import numpy as np
import polars as pl
import pytest


def test_time_window_integration_correct_bins(mock_neural_data, config):
    """
    Test amplitude integration over [0.5s, 2.0s] window (bins 2-7).
    
    Paper: "we integrated the estimated spike count of each cell between 
    [0.5 s, 2 s] from the start of visual stimulation" (Methods, page 9)
    
    Time conversion: 0.5s / 0.275s = bin 2, 2.0s / 0.275s = bin 7
    """
    from rumyantsev.preprocessing.trial_filtering import integrate_time_window
    
    start_bin = config['preprocessing']['time_window_start_bin']  # 2
    end_bin = config['preprocessing']['time_window_end_bin']  # 7
    
    integrated = integrate_time_window(mock_neural_data, start_bin, end_bin)
    
    # Should have one value per cell per trial
    n_cells = mock_neural_data['cell_idx'].n_unique()
    n_trials = mock_neural_data['trial_idx'].n_unique()
    assert len(integrated) == n_cells * n_trials
    
    # Check columns exist
    assert 'mouse_id' in integrated.columns
    assert 'cell_idx' in integrated.columns
    assert 'trial_idx' in integrated.columns
    assert 'behavior' in integrated.columns
    assert 'integrated_amplitude' in integrated.columns


def test_time_window_only_uses_specified_bins(mock_neural_data):
    """
    Verify integration only includes bins 2-7 (not all 14 bins).
    
    Paper specifies [0.5s, 2.0s] window, which corresponds to bins 2-7.
    """
    from rumyantsev.preprocessing.trial_filtering import integrate_time_window
    
    start_bin, end_bin = 2, 7
    integrated = integrate_time_window(mock_neural_data, start_bin, end_bin)
    
    # Get first cell, first trial
    first_row = integrated.filter(
        (pl.col('cell_idx') == 0) & (pl.col('trial_idx') == 0)
    )
    
    # Manually compute expected sum for bins 2-7
    manual_sum = (
        mock_neural_data
        .filter(
            (pl.col('cell_idx') == 0) &
            (pl.col('trial_idx') == 0) &
            (pl.col('sample_idx') >= start_bin) &
            (pl.col('sample_idx') <= end_bin)
        )['amplitude']
        .sum()
    )
    
    assert np.isclose(first_row['integrated_amplitude'][0], manual_sum)


def test_time_window_values_non_negative(mock_neural_data):
    """
    Test that integrated amplitudes are non-negative.
    
    Paper uses deconvolved spike counts, which should be positive.
    """
    from rumyantsev.preprocessing.trial_filtering import integrate_time_window
    
    integrated = integrate_time_window(mock_neural_data, 2, 7)
    
    # All integrated amplitudes should be >= 0
    assert (integrated['integrated_amplitude'] >= 0).all()


def test_time_bin_conversion_matches_paper(config):
    """
    Verify time-to-bin conversion matches paper specifications.
    
    Paper: 0.275s per bin after 2× downsampling (Methods, page 8)
    Window: [0.5s, 2.0s]
    Expected bins: [2, 7]
    """
    time_bin_size = config['preprocessing']['time_bin_size']  # 0.275s
    
    start_bin = round(0.5 / time_bin_size)
    end_bin = round(2.0 / time_bin_size)
    
    assert start_bin == 2
    assert end_bin == 7


def test_validate_trial_counts_accepts_valid_range(mock_neural_data, config):
    """
    Test that validation passes for trial counts in expected range.
    
    Paper: "217-332 trials per stimulus after filtering" (Methods)
    """
    from rumyantsev.preprocessing.trial_filtering import validate_trial_counts
    
    expected_min = config['preprocessing']['expected_trials_per_stimulus_min']
    expected_max = config['preprocessing']['expected_trials_per_stimulus_max']
    
    # Mock data has 25 trials per stimulus (50 total, alternating)
    # This is outside the paper range, so should raise error
    with pytest.raises(ValueError, match="trials"):
        validate_trial_counts(mock_neural_data, expected_min, expected_max)


def test_validate_trial_counts_with_real_data(config, data_path):
    """
    Validate actual dataset trial counts match paper range.
    
    Paper: "217-332 trials per stimulus after filtering" (Methods, page 7)
    
    This confirms data is pre-filtered for locomotion.
    """
    from rumyantsev.data.loader import DataLoader
    from rumyantsev.preprocessing.trial_filtering import validate_trial_counts
    
    if not data_path.exists():
        pytest.skip("Actual dataset not available")
    
    loader = DataLoader(data_path)
    expected_min = config['preprocessing']['expected_trials_per_stimulus_min']
    expected_max = config['preprocessing']['expected_trials_per_stimulus_max']
    
    # Test each mouse
    for mouse_id in loader.data['mouse_id'].unique():
        mouse_data = loader.get_mouse_data(mouse_id)
        
        # Should not raise error (data is pre-filtered)
        validate_trial_counts(mouse_data, expected_min, expected_max)


def test_reshape_to_matrix(mock_neural_data):
    """
    Test reshaping integrated data to (n_cells, n_trials) matrix.
    
    This format is needed for correlation analysis.
    """
    from rumyantsev.preprocessing.trial_filtering import (
        integrate_time_window, reshape_to_matrix)
    
    integrated = integrate_time_window(mock_neural_data, 2, 7)
    
    # Reshape to matrix
    response_matrix, stimulus_labels = reshape_to_matrix(integrated)
    
    # Check dimensions
    n_cells = mock_neural_data['cell_idx'].n_unique()
    n_trials = mock_neural_data['trial_idx'].n_unique()
    
    assert response_matrix.shape == (n_cells, n_trials)
    assert stimulus_labels.shape == (n_trials,)
    
    # Check stimulus labels are correct
    assert set(stimulus_labels) == {30, -30}


def test_extract_stimulus_labels(mock_neural_data):
    """
    Test extraction of stimulus labels per trial.
    
    Paper uses two stimuli: ±30° gratings (Methods, page 6)
    """
    from rumyantsev.preprocessing.trial_filtering import \
        extract_stimulus_labels
    
    labels = extract_stimulus_labels(mock_neural_data)
    
    # Should have one label per trial
    n_trials = mock_neural_data['trial_idx'].n_unique()
    assert len(labels) == n_trials
    
    # Labels should be behavior values (30, -30)
    assert set(labels) == {30, -30}
    
    # Check labels are in trial order
    expected_labels = (
        mock_neural_data
        .select(['trial_idx', 'behavior'])
        .unique()
        .sort('trial_idx')['behavior']
        .to_numpy()
    )
    np.testing.assert_array_equal(labels, expected_labels)

