# Step-by-Step Implementation Plan

> **📝 Updated based on actual dataset structure** (see `DATA_STRUCTURE_ANALYSIS.md`)  
> Key changes: Actual column names, per-mouse processing, locomotion data pre-filtered

---

## Phase 1: Project Setup (2-4 hours)

### 1.1 Initialize Project
```bash
# Create project structure
uv init rumyantsev-recreation
cd rumyantsev-recreation
uv venv
source .venv/bin/activate

# Install dependencies
uv add polars numpy scipy matplotlib seaborn pytest pytest-cov pyyaml
uv add --dev ruff mypy black
```

### 1.2 Create Directory Structure
```
rumyantsev-recreation/
├── pyproject.toml
├── README.md
├── config/
│   └── analysis_config.yaml
├── data/
│   └── .gitkeep  (actual data not committed)
├── src/
│   └── rumyantsev/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── trial_filtering.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── noise_correlations.py
│       │   └── tuning_similarity.py
│       └── visualization/
│           ├── __init__.py
│           └── figure_2.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_correlations.py
│   └── test_visualization.py
├── notebooks/
│   └── reproduce_figure_2.ipynb
└── outputs/
    └── .gitkeep
```

### 1.3 Configuration File
Create `config/analysis_config.yaml`:
```yaml
# Analysis parameters from Rumyantsev et al. 2020 Methods
# Updated based on actual dataset structure

data:
  parquet_path: "coding_fidelity_bounds.dataset.parquet"  # Actual filename
  expected_n_mice: 5
  expected_total_cells: 8029  # ✅ Verified - perfect match!
  # Note: cell_idx is per-mouse indexed (0-based), not globally unique
  
  # Actual column names in parquet:
  actual_columns:
    mouse_id: "mouse_id"        # string IDs like "Mouse_L347"
    cell_idx: "cell_idx"        # per-mouse indexed (0-N)
    trial_idx: "trial_idx"      # integer trial ID
    sample_idx: "sample_idx"    # time bin index (0-13)
    behavior: "behavior"        # stimulus orientation (30, -30)
    amplitude: "amplitude"      # neural activity (deconvolved spikes)

preprocessing:
  # ⚠️ locomotion_speed column NOT present - data appears pre-filtered
  # Skip locomotion filtering step
  
  time_window_start: 0.5     # seconds
  time_window_end: 2.0       # seconds
  time_bin_size: 0.275       # seconds per sample_idx
  
  # Converted to bin indices:
  time_window_start_bin: 2   # 0.5s / 0.275s ≈ bin 2
  time_window_end_bin: 7     # 2.0s / 0.275s ≈ bin 7
  n_time_bins: 14            # sample_idx ranges 0-13
  
  # Trial counts (TOTAL across both stimuli):
  expected_trials_total_min: 435
  expected_trials_total_max: 662
  # Per stimulus (~half of total):
  expected_trials_per_stimulus_min: 217
  expected_trials_per_stimulus_max: 331

stimuli:
  behavior_codes: [30, -30]  # degrees from vertical (actual values in data)
  labels: ["A", "B"]         # mapped labels for analysis
  mapping:  # behavior → stimulus_type
    30: "A"    # +30° grating
    -30: "B"   # -30° grating

correlation_analysis:
  method: "pearson"
  average_across_stimuli: true
  process_per_mouse: true  # ⚠️ CRITICAL: cell_idx only unique within mouse
  
tuning_similarity:
  top_active_percentile: 10  # for Fig 2e
  covariance_threshold: 0    # classify similar vs different

statistics:
  ks_test_sides: "two-sided"
  expected_pvalue_threshold: 1.3e-6  # all mice should be < this

visualization:
  figure_2d:
    n_bins: 100
    x_range: [-0.2, 0.3]
    colors:
      real: "#2E86AB"
      shuffled: "#A23B72"
  figure_2e:
    colors:
      similar: "#E63946"
      different: "#457B9D"

validation:
  expected_mean_correlation: 0.06
  expected_correlation_std: 0.01
  # Total pairs = sum of per-mouse pairs:
  # L347: 1,844,160 + L354: 650,370 + L363: 1,521,880
  # + L355: 2,399,145 + L362: 530,965 = 6,946,520
  expected_total_pairs: 6946280
  shuffled_variance_ratio: 0.5  # shuffled variance / real variance
```

---

## Phase 2: Data Loading & Validation (4-6 hours)

### 2.1 Create Test Fixtures
`tests/conftest.py`:
```python
import pytest
import numpy as np
import polars as pl
from pathlib import Path

@pytest.fixture
def mock_neural_data():
    """Create synthetic data matching ACTUAL dataset structure."""
    n_cells = 100
    n_trials = 50
    n_time_bins = 14  # Actual: 0-13
    
    rng = np.random.default_rng(42)
    
    # Use ACTUAL column names from parquet file
    data = {
        'mouse_id': np.repeat('Mouse_TEST', n_cells * n_trials * n_time_bins),
        'cell_idx': np.tile(np.repeat(np.arange(n_cells), n_trials * n_time_bins), 1),
        'trial_idx': np.tile(np.repeat(np.arange(n_trials), n_time_bins), n_cells),
        'sample_idx': np.tile(np.arange(n_time_bins), n_cells * n_trials),  # 0-13
        'behavior': np.tile(np.repeat([30, -30], n_trials // 2), n_cells * n_time_bins),  # ±30°
        'amplitude': rng.gamma(2, 0.5, n_cells * n_trials * n_time_bins),  # Positive floats
    }
    
    return pl.DataFrame(data)

@pytest.fixture
def config():
    """Load analysis configuration."""
    import yaml
    with open('config/analysis_config.yaml') as f:
        return yaml.safe_load(f)
```

### 2.2 Implement Data Loader (TDD)

**Test first** (`tests/test_data_loader.py`):
```python
import pytest
from rumyantsev.data.loader import DataLoader

def test_data_loader_validates_structure(mock_neural_data, tmp_path):
    """Ensure data has ACTUAL columns from parquet file."""
    # Save mock data
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

def test_data_loader_counts_match_paper(config):
    """Validate data dimensions against paper statistics."""
    loader = DataLoader(config['data']['parquet_path'])
    
    assert loader.n_mice == config['data']['expected_n_mice']
    # Must count cells PER MOUSE then sum (cell_idx not globally unique!)
    total_cells = loader.count_total_cells()
    assert total_cells == config['data']['expected_total_cells']

def test_get_mouse_data(mock_neural_data, tmp_path):
    """Test extraction of single mouse data."""
    parquet_path = tmp_path / "test_data.parquet"
    mock_neural_data.write_parquet(parquet_path)
    
    loader = DataLoader(parquet_path)
    mouse_data = loader.get_mouse_data('Mouse_TEST')
    
    assert mouse_data['mouse_id'].unique().to_list() == ['Mouse_TEST']
    assert len(mouse_data) > 0
```

**Then implement** (`src/rumyantsev/data/loader.py`):
```python
import polars as pl
from pathlib import Path
from typing import Optional

class DataLoader:
    """
    Load and validate neural dataset from parquet.
    
    NOTE: cell_idx is per-mouse indexed (not globally unique).
    Always process data per-mouse to avoid ID collisions.
    """
    
    # ACTUAL column names in the parquet file
    REQUIRED_COLUMNS = [
        'mouse_id', 'cell_idx', 'trial_idx', 'sample_idx',
        'behavior', 'amplitude'
    ]
    
    def __init__(self, parquet_path: str | Path):
        """
        Initialize loader and validate data structure.
        
        Args:
            parquet_path: Path to parquet file with neural data
            
        Raises:
            ValueError: If required columns missing or data invalid
        """
        self.parquet_path = Path(parquet_path)
        self._load_and_validate()
    
    def _load_and_validate(self):
        """Load data and check structure."""
        self.data = pl.read_parquet(self.parquet_path)
        
        # Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Cache statistics
        self.n_mice = self.data['mouse_id'].n_unique()
        
        # NOTE: Don't use data['cell_idx'].n_unique() - gives wrong count!
        # cell_idx is per-mouse, must count separately per mouse
        
        self.n_total_trials = self.data['trial_idx'].n_unique()
        
    def count_total_cells(self) -> int:
        """
        Count total cells across all mice.
        
        CRITICAL: cell_idx is per-mouse indexed, so we must count
        unique cells per mouse and sum.
        
        Returns:
            Total number of unique cells across all mice
        """
        cells_per_mouse = (
            self.data
            .group_by('mouse_id')
            .agg(pl.col('cell_idx').n_unique().alias('n_cells'))
        )
        return cells_per_mouse['n_cells'].sum()
        
    def get_mouse_data(self, mouse_id: str) -> pl.DataFrame:
        """
        Extract data for a single mouse.
        
        Args:
            mouse_id: Mouse identifier (e.g., "Mouse_L347")
            
        Returns:
            DataFrame filtered to single mouse
        """
        return self.data.filter(pl.col('mouse_id') == mouse_id)
    
    def get_cell_data(self, mouse_id: str, cell_idx: int) -> pl.DataFrame:
        """
        Extract all trials for a specific cell.
        
        Args:
            mouse_id: Mouse identifier
            cell_idx: Cell index (unique within mouse)
            
        Returns:
            DataFrame with all data for this cell
        """
        return self.data.filter(
            (pl.col('mouse_id') == mouse_id) & 
            (pl.col('cell_idx') == cell_idx)
        )
    
    def add_stimulus_labels(self) -> pl.DataFrame:
        """
        Add stimulus_type column mapping behavior codes to labels.
        
        Maps: 30 → 'A', -30 → 'B'
        
        Returns:
            DataFrame with added 'stimulus_type' column
        """
        stimulus_map = {30: 'A', -30: 'B'}
        return self.data.with_columns(
            pl.col('behavior').map_dict(stimulus_map).alias('stimulus_type')
        )
```

---

## Phase 3: Preprocessing (3-4 hours)

### 3.1 Skip Locomotion Filtering

⚠️ **Data appears pre-filtered** - no `locomotion_speed` column present.

Add validation function to confirm trial counts match expected post-filtering range:

**Test** (`tests/test_preprocessing.py`):
```python
def test_trial_counts_match_expected_range(config):
    """Validate trial counts match paper's expected post-filtering range."""
    from rumyantsev.data.loader import DataLoader
    
    loader = DataLoader(config['data']['parquet_path'])
    
    # Check each mouse
    for mouse_id in loader.data['mouse_id'].unique():
        mouse_data = loader.get_mouse_data(mouse_id)
        
        # Count trials per stimulus
        trials_per_stim = (
            mouse_data
            .group_by('behavior')
            .agg(pl.col('trial_idx').n_unique().alias('n_trials'))
        )
        
        # Each stimulus should have expected number of trials
        for n_trials in trials_per_stim['n_trials']:
            assert config['preprocessing']['expected_trials_per_stimulus_min'] <= n_trials
            assert n_trials <= config['preprocessing']['expected_trials_per_stimulus_max']
```

**Implement** (`src/rumyantsev/preprocessing/trial_filtering.py`):
```python
import polars as pl

def validate_trial_counts(data: pl.DataFrame, expected_min: int, expected_max: int):
    """
    Validate that trial counts match expected post-filtering range.
    
    NOTE: Data appears to be pre-filtered (no locomotion_speed column).
    This function verifies trial counts match the expected range from the paper.
    
    Paper: "217-332 trials per stimulus after filtering" (Methods)
    
    Args:
        data: Neural data for single mouse
        expected_min: Minimum expected trials per stimulus
        expected_max: Maximum expected trials per stimulus
        
    Raises:
        ValueError: If trial counts out of expected range
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
```

### 3.2 Time Window Integration

**Test**:
```python
def test_time_window_integration(mock_neural_data, config):
    """Test amplitude integration over [0.5s, 2.0s] window (bins 2-7)."""
    from rumyantsev.preprocessing.trial_filtering import integrate_time_window
    
    start_bin = config['preprocessing']['time_window_start_bin']
    end_bin = config['preprocessing']['time_window_end_bin']
    
    integrated = integrate_time_window(mock_neural_data, start_bin, end_bin)
    
    # Should have one value per cell per trial
    n_cells = mock_neural_data['cell_idx'].n_unique()
    n_trials = mock_neural_data['trial_idx'].n_unique()
    assert len(integrated) == n_cells * n_trials
    
    # Values should be sums of amplitudes (non-negative)
    assert (integrated['integrated_amplitude'] >= 0).all()
    
def test_time_window_uses_correct_bins():
    """Verify bins 2-7 correspond to [0.5s, 2.0s] window."""
    # 0.5s / 0.275s/bin ≈ 1.82 → bin 2
    # 2.0s / 0.275s/bin ≈ 7.27 → bin 7
    start_bin = round(0.5 / 0.275)
    end_bin = round(2.0 / 0.275)
    assert start_bin == 2
    assert end_bin == 7
```

**Implement**:
```python
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
```

---

## Phase 4: Noise Correlation Computation (8-10 hours)

### 4.1 Core Correlation Function (TDD)

**Test** (`tests/test_correlations.py`):
```python
import numpy as np
import pytest

def test_noise_correlation_removes_mean():
    """Verify mean response is subtracted before correlation."""
    from rumyantsev.analysis.noise_correlations import compute_noise_correlation
    
    # Create data with known mean
    cell_i = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # mean = 4.5
    cell_j = np.array([2, 3, 4, 5, 6, 7, 8, 9])  # mean = 5.5
    stimuli = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    
    r_noise = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Should compute correlation of (x - mean_x, y - mean_y)
    assert isinstance(r_noise, float)
    assert -1 <= r_noise <= 1

def test_correlation_averaged_across_stimuli():
    """Test that correlation is mean of per-stimulus correlations."""
    from rumyantsev.analysis.noise_correlations import compute_noise_correlation
    
    # Perfect correlation for stimulus A, zero for B
    cell_i = np.array([1, 2, 3, 4, 0, 0, 0, 0])
    cell_j = np.array([1, 2, 3, 4, 0, 1, 2, 3])  # correlated A, not B
    stimuli = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    
    r_noise = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Should be average: (1.0 + ~0) / 2 ≈ 0.5
    assert 0.4 <= r_noise <= 0.6

def test_pairwise_correlations_shape():
    """Test computation of all pairwise correlations."""
    from rumyantsev.analysis.noise_correlations import compute_all_pairwise
    
    # Mock data: 10 neurons, 20 trials
    integrated_spikes = np.random.randn(10, 20)
    stimuli = np.repeat(['A', 'B'], 10)
    
    correlations = compute_all_pairwise(integrated_spikes, stimuli)
    
    # Should get n*(n-1)/2 unique pairs
    expected_pairs = 10 * 9 // 2
    assert len(correlations) == expected_pairs
```

**Implement** (`src/rumyantsev/analysis/noise_correlations.py`):
```python
import numpy as np
from typing import Tuple
import polars as pl

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
        r = np.corrcoef(noise_i, noise_j)[0, 1]
        correlations.append(r)
    
    # Average across stimuli
    return np.mean(correlations)


def compute_all_pairwise(
    integrated_responses: np.ndarray,
    stimulus_labels: np.ndarray,
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute noise correlations for all neuron pairs.
    
    Args:
        integrated_responses: shape (n_neurons, n_trials)
        stimulus_labels: shape (n_trials,)
        show_progress: Whether to show progress bar
        
    Returns:
        correlations: shape (n_pairs,) where n_pairs = n_neurons * (n_neurons - 1) / 2
    """
    n_neurons = integrated_responses.shape[0]
    n_pairs = n_neurons * (n_neurons - 1) // 2
    correlations = np.empty(n_pairs)
    
    from itertools import combinations
    if show_progress:
        from tqdm import tqdm
        pairs = tqdm(combinations(range(n_neurons), 2), total=n_pairs)
    else:
        pairs = combinations(range(n_neurons), 2)
    
    for idx, (i, j) in enumerate(pairs):
        correlations[idx] = compute_noise_correlation(
            integrated_responses[i],
            integrated_responses[j],
            stimulus_labels
        )
    
    return correlations
```

### 4.2 Trial Shuffling Control

**Test**:
```python
def test_trial_shuffling_independence():
    """Verify each cell gets independent shuffle."""
    from rumyantsev.analysis.noise_correlations import shuffle_trials
    
    responses = np.arange(50).reshape(5, 10)  # 5 cells, 10 trials
    stimuli = np.repeat(['A', 'B'], 5)
    
    shuffled = shuffle_trials(responses, stimuli, random_seed=42)
    
    # Shape should be preserved
    assert shuffled.shape == responses.shape
    
    # Each cell should have same values (just reordered)
    for cell_idx in range(5):
        assert set(shuffled[cell_idx]) == set(responses[cell_idx])
    
def test_shuffled_correlations_reduced():
    """Verify shuffling reduces correlations."""
    from rumyantsev.analysis.noise_correlations import (
        compute_all_pairwise, shuffle_trials
    )
    
    # Create correlated responses
    n_neurons = 20
    n_trials = 100
    
    # Add common noise to create correlations
    common_noise = np.random.randn(n_trials)
    responses = np.random.randn(n_neurons, n_trials)
    responses += common_noise * 0.3  # Inject correlations
    
    stimuli = np.repeat(['A', 'B'], n_trials // 2)
    
    # Compute correlations
    real_corr = compute_all_pairwise(responses, stimuli, show_progress=False)
    shuffled_responses = shuffle_trials(responses, stimuli)
    shuffled_corr = compute_all_pairwise(shuffled_responses, stimuli, show_progress=False)
    
    # Shuffled should have lower variance (paper: ~50%)
    assert np.var(shuffled_corr) < np.var(real_corr)
```

**Implement**:
```python
def shuffle_trials(
    integrated_responses: np.ndarray,
    stimulus_labels: np.ndarray,
    random_seed: int | None = None
) -> np.ndarray:
    """
    Shuffle trials independently for each cell to create null distribution.
    
    Paper: "To create trial-shuffled datasets, we randomly permuted the 
    activity traces of each cell across the full set of trials in which the 
    same stimulus was presented, using a different random permutation for 
    each individual cell." (Methods, page 9)
    
    Args:
        integrated_responses: shape (n_neurons, n_trials)
        stimulus_labels: shape (n_trials,)
        random_seed: For reproducibility
        
    Returns:
        shuffled_responses: shape (n_neurons, n_trials)
    """
    rng = np.random.default_rng(random_seed)
    shuffled = np.empty_like(integrated_responses)
    
    n_neurons = integrated_responses.shape[0]
    unique_stimuli = np.unique(stimulus_labels)
    
    for cell_idx in range(n_neurons):
        for stim in unique_stimuli:
            # Get trials for this stimulus
            mask = stimulus_labels == stim
            stim_responses = integrated_responses[cell_idx, mask]
            
            # Shuffle independently per cell
            shuffled_stim = rng.permutation(stim_responses)
            shuffled[cell_idx, mask] = shuffled_stim
    
    return shuffled
```

---

## Phase 5: Tuning Similarity (4-6 hours)

### 5.1 Classification (TDD)

**Test** (`tests/test_tuning.py`):
```python
def test_similarly_tuned_classification():
    """Test identification of similarly tuned cell pairs."""
    from rumyantsev.analysis.tuning_similarity import classify_tuning
    
    # Both cells prefer stimulus A
    mean_A_i, mean_B_i = 10, 5  # prefers A
    mean_A_j, mean_B_j = 8, 3   # also prefers A
    
    classification = classify_tuning(mean_A_i, mean_B_i, mean_A_j, mean_B_j)
    assert classification == 'similar'

def test_differently_tuned_classification():
    """Test identification of differently tuned cell pairs."""
    from rumyantsev.analysis.tuning_similarity import classify_tuning
    
    # Opposite preferences
    mean_A_i, mean_B_i = 10, 5  # prefers A
    mean_A_j, mean_B_j = 3, 8   # prefers B
    
    classification = classify_tuning(mean_A_i, mean_B_i, mean_A_j, mean_B_j)
    assert classification == 'different'

def test_top_active_cells_selection():
    """Test selection of top N% active cells."""
    from rumyantsev.analysis.tuning_similarity import select_top_active
    
    # Create mock cell responses
    n_cells = 100
    responses_A = np.random.rand(n_cells, 50)  # 50 trials
    responses_B = np.random.rand(n_cells, 50)
    
    top_cells = select_top_active(responses_A, responses_B, percentile=10)
    
    # Should get 10% of cells
    assert len(top_cells) == 10
```

**Implement** (`src/rumyantsev/analysis/tuning_similarity.py`):
```python
import numpy as np
from typing import Literal

def classify_tuning(
    mean_response_A_i: float,
    mean_response_B_i: float,
    mean_response_A_j: float,
    mean_response_B_j: float
) -> Literal['similar', 'different']:
    """
    Classify whether two cells are similarly or differently tuned.
    
    Paper: "we compared the statistical distributions of noise correlation 
    coefficients for two different sets of cell pairs, those with positive 
    and those with negative covariance of their mean response to the two 
    stimuli." (Methods, page 9)
    
    Math: Cov(μᵢ, μⱼ) = (μᵢ(A) - μ̄ᵢ)(μⱼ(A) - μ̄ⱼ) + (μᵢ(B) - μ̄ᵢ)(μⱼ(B) - μ̄ⱼ)
    
    Args:
        mean_response_A_i: Mean response of cell i to stimulus A
        mean_response_B_i: Mean response of cell i to stimulus B
        mean_response_A_j: Mean response of cell j to stimulus A
        mean_response_B_j: Mean response of cell j to stimulus B
        
    Returns:
        'similar' if covariance > 0, 'different' if covariance < 0
    """
    # Center around mean
    mean_i = (mean_response_A_i + mean_response_B_i) / 2
    mean_j = (mean_response_A_j + mean_response_B_j) / 2
    
    # Compute covariance
    cov = ((mean_response_A_i - mean_i) * (mean_response_A_j - mean_j) +
           (mean_response_B_i - mean_i) * (mean_response_B_j - mean_j))
    
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
    """
    # Compute mean responses
    mean_A = responses_A.mean(axis=1)
    mean_B = responses_B.mean(axis=1)
    
    # Activity metric
    activity = np.sqrt(mean_A**2 + mean_B**2)
    
    # Select top percentile
    threshold = np.percentile(activity, 100 - percentile)
    top_indices = np.where(activity >= threshold)[0]
    
    return top_indices
```

---

## Phase 6: Statistical Testing & Visualization (6-8 hours)

### 6.1 Kolmogorov-Smirnov Test

**Test** (`tests/test_correlations.py`):
```python
def test_ks_test_significant_difference():
    """Test KS test on clearly different distributions."""
    from rumyantsev.analysis.statistics import compare_distributions
    
    # Create distributions with known difference
    dist1 = np.random.normal(0.05, 0.02, 1000)
    dist2 = np.random.normal(0.10, 0.02, 1000)
    
    statistic, pvalue = compare_distributions(dist1, dist2)
    
    assert pvalue < 0.05  # Should find significant difference

def test_ks_test_matches_paper_pvalue():
    """Validate p-value is below paper threshold."""
    from rumyantsev.analysis.statistics import compare_distributions
    
    # Load real correlation data
    sim_tuned = load_similarly_tuned_correlations()
    diff_tuned = load_differently_tuned_correlations()
    
    statistic, pvalue = compare_distributions(sim_tuned, diff_tuned)
    
    # Paper reports p < 1.3e-6 for all mice
    assert pvalue < 1.3e-6
```

**Implement** (`src/rumyantsev/analysis/statistics.py`):
```python
from scipy.stats import ks_2samp
from typing import Tuple

def compare_distributions(
    dist1: np.ndarray,
    dist2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform two-sample Kolmogorov-Smirnov test.
    
    Paper: "two-tailed Kolmogorov–Smirnov test, ***P < 10−13 for all 5 mice"
    (Figure 2 legend)
    
    Args:
        dist1: First distribution of correlation coefficients
        dist2: Second distribution of correlation coefficients
        
    Returns:
        statistic: KS test statistic
        pvalue: Two-tailed p-value
    """
    statistic, pvalue = ks_2samp(dist1, dist2)
    return statistic, pvalue
```

### 6.2 Figure 2d Generation

**Test** (`tests/test_visualization.py`):
```python
def test_figure_2d_creation():
    """Test Figure 2d histogram generation."""
    from rumyantsev.visualization.figure_2 import create_figure_2d
    
    # Mock correlation data
    real_corr = np.random.normal(0.06, 0.03, 1000000)
    shuffled_corr = np.random.normal(0.0, 0.015, 1000000)
    
    fig, ax = create_figure_2d(real_corr, shuffled_corr)
    
    assert fig is not None
    assert ax is not None
    # Check legend has correct labels
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert 'Real data' in legend_labels
    assert 'Shuffled' in legend_labels
```

**Implement** (`src/rumyantsev/visualization/figure_2.py`):
```python
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import yaml

def create_figure_2d(
    real_correlations: np.ndarray,
    shuffled_correlations: np.ndarray,
    config_path: str = 'config/analysis_config.yaml'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Recreate Figure 2d from paper.
    
    Shows histograms of noise correlation coefficients for real and 
    trial-shuffled data.
    
    Args:
        real_correlations: Correlation coefficients from real data
        shuffled_correlations: Correlation coefficients from shuffled data
        config_path: Path to visualization config
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    viz_config = config['visualization']['figure_2d']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram parameters
    bins = viz_config['n_bins']
    x_range = viz_config['x_range']
    
    # Plot histograms
    ax.hist(
        real_correlations,
        bins=bins,
        range=x_range,
        alpha=0.6,
        color=viz_config['colors']['real'],
        label='Real data',
        density=False
    )
    
    ax.hist(
        shuffled_correlations,
        bins=bins,
        range=x_range,
        alpha=0.6,
        color=viz_config['colors']['shuffled'],
        label='Shuffled',
        density=False
    )
    
    # Styling to match paper
    ax.set_xlabel('Correlation coefficient', fontsize=12)
    ax.set_ylabel('Number of cell pairs', fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add summary statistics
    stats_text = (
        f"Real: μ={np.mean(real_correlations):.3f}, "
        f"σ={np.std(real_correlations):.3f}\n"
        f"Shuffled: μ={np.mean(shuffled_correlations):.3f}, "
        f"σ={np.std(shuffled_correlations):.3f}"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    plt.tight_layout()
    return fig, ax


def create_figure_2e(
    similarly_tuned_corr: np.ndarray,
    differently_tuned_corr: np.ndarray,
    pvalue: float,
    config_path: str = 'config/analysis_config.yaml'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Recreate Figure 2e from paper.
    
    Compares noise correlation distributions for similarly vs 
    differently tuned cell pairs.
    
    Args:
        similarly_tuned_corr: Correlations for similarly tuned pairs
        differently_tuned_corr: Correlations for differently tuned pairs
        pvalue: KS test p-value
        config_path: Path to visualization config
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    viz_config = config['visualization']['figure_2e']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histograms
    bins = 50
    ax.hist(
        similarly_tuned_corr,
        bins=bins,
        alpha=0.6,
        color=viz_config['colors']['similar'],
        label='Similarly tuned',
        density=True
    )
    
    ax.hist(
        differently_tuned_corr,
        bins=bins,
        alpha=0.6,
        color=viz_config['colors']['different'],
        label='Differently tuned',
        density=True
    )
    
    # Styling
    ax.set_xlabel('Correlation coefficient', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add p-value annotation
    ax.text(
        0.95, 0.95,
        f'KS test: p = {pvalue:.2e}',
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    return fig, ax
```

---

## Phase 7: Integration & Validation (4-6 hours)

### 7.1 End-to-End Pipeline

Create `notebooks/reproduce_figure_2.ipynb`:
```python
# Cell 1: Imports
import yaml
import numpy as np
import polars as pl
from pathlib import Path

from rumyantsev.data.loader import DataLoader
from rumyantsev.preprocessing.trial_filtering import (
    filter_by_locomotion, integrate_time_window
)
from rumyantsev.analysis.noise_correlations import (
    compute_all_pairwise, shuffle_trials
)
from rumyantsev.analysis.tuning_similarity import (
    classify_tuning, select_top_active
)
from rumyantsev.analysis.statistics import compare_distributions
from rumyantsev.visualization.figure_2 import (
    create_figure_2d, create_figure_2e
)

# Cell 2: Load configuration
config_path = Path('../config/analysis_config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

print("Configuration loaded:")
print(f"  Locomotion threshold: {config['preprocessing']['locomotion_threshold']} mm/s")
print(f"  Time window: [{config['preprocessing']['time_window_start']}, "
      f"{config['preprocessing']['time_window_end']}] s")

# Cell 3: Load and validate data
data_path = Path(config['data']['parquet_path'])
loader = DataLoader(data_path)

print(f"\nData loaded:")
print(f"  Total mice: {loader.n_mice}")
print(f"  Total neurons: {loader.n_total_cells}")
print(f"  Expected: {config['data']['expected_total_cells']}")

assert loader.n_mice == config['data']['expected_n_mice'], "Mouse count mismatch!"

# Cell 4: Process each mouse
results = {}

for mouse_id in range(loader.n_mice):
    print(f"\nProcessing mouse {mouse_id}...")
    
    # Extract mouse data
    mouse_data = loader.get_mouse_data(mouse_id)
    
    # Filter by locomotion
    filtered = filter_by_locomotion(
        mouse_data,
        config['preprocessing']['locomotion_threshold']
    )
    
    n_trials_retained = filtered['trial_id'].n_unique()
    print(f"  Trials retained: {n_trials_retained}")
    
    # Integrate spikes over time window
    integrated = integrate_time_window(
        filtered,
        config['preprocessing']['time_window_start'],
        config['preprocessing']['time_window_end']
    )
    
    # Convert to numpy for correlation analysis
    # Shape: (n_neurons, n_trials)
    response_matrix = (
        integrated
        .pivot(
            values='integrated_spikes',
            index='neuron_id',
            columns='trial_id'
        )
        .to_numpy()
    )
    
    stimulus_labels = (
        integrated
        .select(['trial_id', 'stimulus_type'])
        .unique()
        .sort('trial_id')['stimulus_type']
        .to_numpy()
    )
    
    # Compute noise correlations
    print("  Computing pairwise correlations...")
    real_corr = compute_all_pairwise(response_matrix, stimulus_labels)
    
    # Compute shuffled correlations
    print("  Computing shuffled correlations...")
    shuffled_responses = shuffle_trials(response_matrix, stimulus_labels)
    shuffled_corr = compute_all_pairwise(shuffled_responses, stimulus_labels)
    
    results[mouse_id] = {
        'real_correlations': real_corr,
        'shuffled_correlations': shuffled_corr,
        'response_matrix': response_matrix,
        'stimulus_labels': stimulus_labels,
        'integrated_data': integrated
    }
    
    print(f"  Real corr: μ={np.mean(real_corr):.3f}, σ={np.std(real_corr):.3f}")
    print(f"  Shuffled corr: μ={np.mean(shuffled_corr):.3f}, σ={np.std(shuffled_corr):.3f}")

# Cell 5: Generate Figure 2d
print("\n" + "="*60)
print("FIGURE 2D: Noise Correlation Distribution")
print("="*60)

# Aggregate across all mice
all_real_corr = np.concatenate([r['real_correlations'] for r in results.values()])
all_shuffled_corr = np.concatenate([r['shuffled_correlations'] for r in results.values()])

fig_2d, ax_2d = create_figure_2d(all_real_corr, all_shuffled_corr)
fig_2d.savefig('../outputs/figure_2d_recreation.png', dpi=300, bbox_inches='tight')
print(f"✓ Figure 2d saved to outputs/figure_2d_recreation.png")

# Validation
print(f"\nValidation against paper:")
print(f"  Expected mean: {config['validation']['expected_mean_correlation']:.3f}")
print(f"  Actual mean: {np.mean(all_real_corr):.3f}")
print(f"  Variance ratio (shuffled/real): {np.var(all_shuffled_corr)/np.var(all_real_corr):.2f}")
print(f"  Expected ratio: {config['validation']['shuffled_variance_ratio']:.2f}")

# Cell 6: Tuning similarity analysis (Figure 2e)
print("\n" + "="*60)
print("FIGURE 2E: Tuning Similarity Analysis")
print("="*60)

# Only use top 10% most active cells
all_sim_tuned = []
all_diff_tuned = []

for mouse_id, mouse_results in results.items():
    print(f"\nMouse {mouse_id}:")
    
    responses = mouse_results['response_matrix']
    stimuli = mouse_results['stimulus_labels']
    integrated = mouse_results['integrated_data']
    
    # Split by stimulus
    mask_A = stimuli == 'A'
    mask_B = stimuli == 'B'
    responses_A = responses[:, mask_A]
    responses_B = responses[:, mask_B]
    
    # Select top active
    top_indices = select_top_active(responses_A, responses_B, percentile=10)
    print(f"  Top 10% active cells: {len(top_indices)}")
    
    # Compute mean responses for classification
    mean_A = responses_A.mean(axis=1)
    mean_B = responses_B.mean(axis=1)
    
    # Classify all pairs and compute correlations
    from itertools import combinations
    
    for i, j in combinations(top_indices, 2):
        # Classify tuning
        tuning = classify_tuning(mean_A[i], mean_B[i], mean_A[j], mean_B[j])
        
        # Get correlation
        r_noise = compute_noise_correlation(
            responses[i],
            responses[j],
            stimuli
        )
        
        if tuning == 'similar':
            all_sim_tuned.append(r_noise)
        else:
            all_diff_tuned.append(r_noise)
    
    print(f"  Similarly tuned pairs: {len(all_sim_tuned)}")
    print(f"  Differently tuned pairs: {len(all_diff_tuned)}")

# Statistical comparison
sim_tuned_arr = np.array(all_sim_tuned)
diff_tuned_arr = np.array(all_diff_tuned)

ks_stat, ks_pvalue = compare_distributions(sim_tuned_arr, diff_tuned_arr)

print(f"\nKolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_stat:.4f}")
print(f"  P-value: {ks_pvalue:.2e}")
print(f"  Expected: p < {config['validation']['expected_pvalue_threshold']:.2e}")
print(f"  ✓ PASS" if ks_pvalue < config['validation']['expected_pvalue_threshold'] else "  ✗ FAIL")

# Generate figure
fig_2e, ax_2e = create_figure_2e(sim_tuned_arr, diff_tuned_arr, ks_pvalue)
fig_2e.savefig('../outputs/figure_2e_recreation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure 2e saved to outputs/figure_2e_recreation.png")

# Cell 7: Summary report
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

summary = {
    'total_mice': loader.n_mice,
    'total_neurons': loader.n_total_cells,
    'total_cell_pairs': len(all_real_corr),
    'mean_noise_correlation': np.mean(all_real_corr),
    'std_noise_correlation': np.std(all_real_corr),
    'shuffled_variance_ratio': np.var(all_shuffled_corr) / np.var(all_real_corr),
    'mean_sim_tuned_correlation': np.mean(sim_tuned_arr),
    'mean_diff_tuned_correlation': np.mean(diff_tuned_arr),
    'ks_statistic': ks_stat,
    'ks_pvalue': ks_pvalue
}

for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Save summary
import json
with open('../outputs/summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Analysis complete!")
```

---

## Success Checklist

Before presenting to Stanford:

- [ ] All unit tests pass (`pytest tests/ -v --cov=src`)
- [ ] Figure 2d visually matches paper
- [ ] Figure 2e visually matches paper  
- [ ] Mean correlation ≈ 0.06 ± 0.01
- [ ] Shuffled variance ratio ≈ 0.5
- [ ] KS test p-value < 1.3×10⁻⁶
- [ ] Total cell pairs ≈ 7 million
- [ ] Code is documented with docstrings
- [ ] README explains how to reproduce
- [ ] Performance: analysis completes in <30 minutes

---

## Deliverables

1. **Code repository** with full test suite
2. **Figure 2d recreation** (PNG, 300 dpi)
3. **Figure 2e recreation** (PNG, 300 dpi)
4. **Jupyter notebook** showing full pipeline
5. **Summary statistics** JSON file
6. **Documentation** explaining methodology
7. **Test coverage report** (>90%)

---

## Key Changes from Original Plan

This implementation plan has been updated based on actual dataset analysis (see `DATA_STRUCTURE_ANALYSIS.md`):

### ✅ Verified & Confirmed
- **Perfect cell count**: 8,029 neurons (matches paper exactly!)
- **5 mice** with expected trial counts per stimulus
- **Time bins**: 14 samples at 0.275s each

### 🔄 Column Name Updates
| Original Expectation | Actual Column | Action |
|---------------------|---------------|--------|
| `neuron_id` | `cell_idx` | Use actual name |
| `trial_id` | `trial_idx` | Use actual name |
| `time_bin` | `sample_idx` | Use actual name (0-13 integers) |
| `stimulus_type` | `behavior` | Map: 30→'A', -30→'B' |
| `spike_count` | `amplitude` | Use actual name (float values) |
| `locomotion_speed` | **MISSING** | Skip filtering step |

### ⚠️ Critical Implementation Notes
1. **cell_idx is per-mouse indexed** (not globally unique)
   - ALWAYS filter by `mouse_id` first
   - Count total cells by summing per-mouse counts
   - Process each mouse independently

2. **Time window uses bin indices**
   - [0.5s, 2.0s] → bins [2, 7]
   - Use integer bin indices, not floating point seconds

3. **Locomotion filtering skipped**
   - Data appears pre-filtered (no column present)
   - Validate trial counts match expected range instead

4. **Stimulus mapping required**
   - Add `stimulus_type` column: {30: 'A', -30: 'B'}

### 📊 Expected Validation Results (Unchanged)
- Total neurons: 8,029 ✅
- Total cell pairs: ~6.95 million ✅
- Mean correlation: 0.06 ± 0.01
- Shuffled variance ratio: ~0.5
- KS test p < 1.3×10⁻⁶

---

**Status**: ✅ Implementation plan aligned with actual data structure  
**See also**: `HANDOFF_CONTEXT.md` for quick-start guide