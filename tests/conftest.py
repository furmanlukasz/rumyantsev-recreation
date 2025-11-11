"""Pytest configuration and fixtures for rumyantsev-recreation tests."""

import pytest
import numpy as np
import polars as pl
from pathlib import Path
import yaml


@pytest.fixture
def mock_neural_data():
    """
    Create synthetic data matching ACTUAL dataset structure.
    
    Uses actual column names from the parquet file:
    - mouse_id, cell_idx, trial_idx, sample_idx, behavior, amplitude
    """
    n_cells = 100
    n_trials = 50
    n_time_bins = 14  # Actual: 0-13
    
    rng = np.random.default_rng(42)
    
    # Create long-format data (one row per observation)
    rows = []
    for cell_idx in range(n_cells):
        for trial_idx in range(n_trials):
            # Alternate between stimuli
            behavior = 30 if trial_idx % 2 == 0 else -30
            for sample_idx in range(n_time_bins):
                rows.append({
                    'mouse_id': 'Mouse_TEST',
                    'cell_idx': cell_idx,
                    'trial_idx': trial_idx,
                    'sample_idx': sample_idx,
                    'behavior': behavior,
                    'amplitude': float(rng.gamma(2, 0.5)),  # Positive floats
                })
    
    return pl.DataFrame(rows)


@pytest.fixture
def mock_multi_mouse_data():
    """
    Create synthetic data with multiple mice to test per-mouse processing.
    
    Critical: cell_idx is per-mouse indexed (not globally unique).
    """
    rng = np.random.default_rng(123)
    
    rows = []
    mice = ['Mouse_A', 'Mouse_B', 'Mouse_C']
    n_cells_per_mouse = [50, 30, 40]  # Different counts per mouse
    n_trials = 40
    n_time_bins = 14
    
    for mouse_id, n_cells in zip(mice, n_cells_per_mouse):
        # cell_idx starts from 0 for each mouse!
        for cell_idx in range(n_cells):
            for trial_idx in range(n_trials):
                behavior = 30 if trial_idx % 2 == 0 else -30
                for sample_idx in range(n_time_bins):
                    rows.append({
                        'mouse_id': mouse_id,
                        'cell_idx': cell_idx,  # Per-mouse indexed!
                        'trial_idx': trial_idx,
                        'sample_idx': sample_idx,
                        'behavior': behavior,
                        'amplitude': float(rng.gamma(2, 0.5)),
                    })
    
    return pl.DataFrame(rows)


@pytest.fixture
def config():
    """Load analysis configuration from config file."""
    config_path = Path(__file__).parent.parent / 'config' / 'analysis_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def data_path():
    """Path to the actual parquet dataset."""
    return Path(__file__).parent.parent / 'coding_fidelity_bounds.dataset.parquet'

