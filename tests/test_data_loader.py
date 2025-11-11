"""
Tests for data loading module.

Following TDD approach: Write tests first, then implementation.
All tests reference Rumyantsev et al. 2020 methodology.
"""

import pytest
import polars as pl
from pathlib import Path


def test_data_loader_validates_structure(mock_neural_data, tmp_path):
    """
    Ensure data has ACTUAL columns from parquet file.
    
    Critical columns (actual names in dataset):
    - mouse_id, cell_idx, trial_idx, sample_idx, behavior, amplitude
    """
    from rumyantsev.data.loader import DataLoader
    
    # Save mock data to temporary parquet file
    parquet_path = tmp_path / "test_data.parquet"
    mock_neural_data.write_parquet(parquet_path)
    
    # Load and validate
    loader = DataLoader(parquet_path)
    
    # Check ACTUAL column names
    assert 'mouse_id' in loader.data.columns
    assert 'cell_idx' in loader.data.columns
    assert 'trial_idx' in loader.data.columns
    assert 'sample_idx' in loader.data.columns
    assert 'behavior' in loader.data.columns
    assert 'amplitude' in loader.data.columns


def test_data_loader_rejects_missing_columns(tmp_path):
    """Ensure loader raises error when required columns are missing."""
    from rumyantsev.data.loader import DataLoader
    
    # Create data with missing columns
    incomplete_data = pl.DataFrame({
        'mouse_id': ['Mouse_A'],
        'cell_idx': [0],
        # Missing other required columns!
    })
    
    parquet_path = tmp_path / "incomplete.parquet"
    incomplete_data.write_parquet(parquet_path)
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Missing required columns"):
        DataLoader(parquet_path)


def test_get_mouse_data(mock_neural_data, tmp_path):
    """
    Test extraction of single mouse data.
    
    Paper: Data collected from 5 mice (Methods, page 6)
    """
    from rumyantsev.data.loader import DataLoader
    
    parquet_path = tmp_path / "test_data.parquet"
    mock_neural_data.write_parquet(parquet_path)
    
    loader = DataLoader(parquet_path)
    mouse_data = loader.get_mouse_data('Mouse_TEST')
    
    # Should only contain data for this mouse
    assert mouse_data['mouse_id'].unique().to_list() == ['Mouse_TEST']
    assert len(mouse_data) > 0


def test_count_total_cells_per_mouse(mock_multi_mouse_data, tmp_path):
    """
    Test that cell counting respects per-mouse indexing.
    
    CRITICAL: cell_idx is NOT globally unique - it's per-mouse indexed!
    
    Paper: "8,029 cells from 5 mice" (Methods, page 6)
    """
    from rumyantsev.data.loader import DataLoader
    
    parquet_path = tmp_path / "multi_mouse.parquet"
    mock_multi_mouse_data.write_parquet(parquet_path)
    
    loader = DataLoader(parquet_path)
    
    # Total cells should be sum across mice, not n_unique(cell_idx)
    total_cells = loader.count_total_cells()
    
    # We created 50 + 30 + 40 = 120 total cells
    assert total_cells == 120, f"Expected 120 total cells, got {total_cells}"
    
    # Verify: naive approach would give wrong answer!
    naive_count = loader.data['cell_idx'].n_unique()
    assert naive_count == 50  # Only max cells in any one mouse
    assert total_cells != naive_count, "Must count per-mouse, not globally!"


def test_data_loader_counts_match_paper(config, data_path):
    """
    Validate data dimensions against paper statistics.
    
    Paper expectations:
    - 5 mice
    - 8,029 total cells
    - 217-332 trials per stimulus after filtering
    
    (Methods, page 6-7)
    """
    from rumyantsev.data.loader import DataLoader
    
    if not data_path.exists():
        pytest.skip("Actual dataset not available")
    
    loader = DataLoader(data_path)
    
    # Check mouse count
    assert loader.n_mice == config['data']['expected_n_mice'], \
        f"Expected {config['data']['expected_n_mice']} mice, got {loader.n_mice}"
    
    # Check total cells (must count per-mouse!)
    total_cells = loader.count_total_cells()
    assert total_cells == config['data']['expected_total_cells'], \
        f"Expected {config['data']['expected_total_cells']} cells, got {total_cells}"


def test_add_stimulus_labels(mock_neural_data, tmp_path):
    """
    Test mapping behavior codes to stimulus labels.
    
    Paper uses two drifting grating stimuli at ±30° from vertical.
    Map: 30 → 'A', -30 → 'B'
    
    (Methods, page 6)
    """
    from rumyantsev.data.loader import DataLoader
    
    parquet_path = tmp_path / "test_data.parquet"
    mock_neural_data.write_parquet(parquet_path)
    
    loader = DataLoader(parquet_path)
    data_with_labels = loader.add_stimulus_labels()
    
    # Check stimulus_type column added
    assert 'stimulus_type' in data_with_labels.columns
    
    # Check mapping is correct
    for row in data_with_labels.select(['behavior', 'stimulus_type']).unique().iter_rows(named=True):
        if row['behavior'] == 30:
            assert row['stimulus_type'] == 'A'
        elif row['behavior'] == -30:
            assert row['stimulus_type'] == 'B'


def test_get_cell_data(mock_neural_data, tmp_path):
    """
    Test extraction of all trials for a specific cell.
    
    Should return all time bins across all trials for one cell.
    """
    from rumyantsev.data.loader import DataLoader
    
    parquet_path = tmp_path / "test_data.parquet"
    mock_neural_data.write_parquet(parquet_path)
    
    loader = DataLoader(parquet_path)
    cell_data = loader.get_cell_data('Mouse_TEST', cell_idx=0)
    
    # Should only contain data for this cell
    assert cell_data['mouse_id'].unique().to_list() == ['Mouse_TEST']
    assert cell_data['cell_idx'].unique().to_list() == [0]
    assert len(cell_data) > 0

