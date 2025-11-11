# Instructions for Cursor AI Agent (Claude Sonnet 4.5)

## Your Role
You are implementing a scientific data analysis pipeline to recreate published results 
from a neuroscience paper. Your goal is to write high-quality, well-tested Python code 
following the test-driven development (TDD) methodology.

## Project Context
- **Paper**: Rumyantsev et al., "Fundamental bounds on the fidelity of sensory cortical coding", Nature 2020
- **Goal**: Recreate Figure 2d and 2e showing noise correlation analysis
- **Data**: Pre-processed neural spike counts in Parquet format
- **Validation**: Prof. Mark Schnitzer at Stanford University

## Core Principles

### 1. Test-Driven Development (TDD)
**ALWAYS follow this sequence**:
```
1. Write test first (RED)
2. Run test - it should fail
3. Write minimal code to pass test (GREEN)
4. Refactor if needed (REFACTOR)
5. Commit with clear message
```

### 2. Reference the Paper
Every function should have docstring citing relevant paper section:
```python
def compute_noise_correlation(...):
    """
    Compute noise correlation between two neurons.
    
    Paper: "To compute correlation coefficients for the noise in the visual 
    responses of a pair of neurons, we first integrated..." 
    (Methods, page 9)
    
    Args:
        ...
    """
```

### 3. Validate Against Paper Statistics
Each component should have validation tests checking against paper's reported values:
- Total neurons: ~8,029
- Total cell pairs: ~6.9 million
- Mean correlation: 0.06 ± 0.01
- Trials per stimulus: 217-332 after filtering
- p-value: < 1.3×10⁻⁶

## Implementation Sequence

### Phase 1: Setup
1. Initialize project with `uv`
2. Create directory structure
3. Set up `pyproject.toml` with dependencies
4. Create `config/analysis_config.yaml`

### Phase 2: Data Loading
**Files to create**:
- `tests/conftest.py` (fixtures)
- `tests/test_data_loader.py`
- `src/rumyantsev/data/loader.py`

**Key tests**:
- Data structure validation
- Expected column presence
- Mouse/neuron counts match paper

### Phase 3: Preprocessing
**Files to create**:
- `tests/test_preprocessing.py`
- `src/rumyantsev/preprocessing/trial_filtering.py`

**Key tests**:
- Locomotion filtering (<0.2 mm/s)
- Time window integration [0.5s, 2.0s]
- Trial count validation (217-332 range)

### Phase 4: Correlation Analysis
**Files to create**:
- `tests/test_correlations.py`
- `src/rumyantsev/analysis/noise_correlations.py`

**Key tests**:
- Mean subtraction before correlation
- Averaging across stimuli
- Trial shuffling independence
- Pairwise computation efficiency

### Phase 5: Tuning Similarity
**Files to create**:
- `tests/test_tuning.py`
- `src/rumyantsev/analysis/tuning_similarity.py`

**Key tests**:
- Similar vs different classification
- Top 10% cell selection
- Covariance calculation

### Phase 6: Statistical Testing
**Files to create**:
- `src/rumyantsev/analysis/statistics.py`

**Key tests**:
- KS test implementation
- P-value < threshold validation

### Phase 7: Visualization
**Files to create**:
- `tests/test_visualization.py`
- `src/rumyantsev/visualization/figure_2.py`

**Key tests**:
- Figure generation without errors
- Legend labels present
- Style matches paper

## Code Quality Requirements

### Type Hints
Use type hints everywhere:
```python
def compute_noise_correlation(
    cell_i_responses: np.ndarray,
    cell_j_responses: np.ndarray,
    stimulus_labels: np.ndarray
) -> float:
```

### Docstrings
Every function needs:
- Brief description
- Paper citation (if applicable)
- Args with types and descriptions
- Returns with type and description
- Example usage (if complex)

### Error Handling
Check assumptions and raise informative errors:
```python
if cell_i_responses.shape != cell_j_responses.shape:
    raise ValueError(
        f"Response shapes must match: "
        f"{cell_i_responses.shape} != {cell_j_responses.shape}"
    )
```

### Performance
For operations on large arrays:
- Use vectorized numpy operations
- Avoid Python loops where possible
- Show progress bars for long computations (use `tqdm`)

## Testing Strategy

### Unit Tests
Test individual functions in isolation:
```python
def test_noise_correlation_with_known_values():
    """Test correlation calculation with manually computed expected result."""
    # Arrange
    cell_i = np.array([1.0, 2.0, 3.0, 4.0])
    cell_j = np.array([1.0, 2.0, 3.0, 4.0])  # Perfect correlation
    stimuli = np.array(['A', 'A', 'B', 'B'])
    
    # Act
    r = compute_noise_correlation(cell_i, cell_j, stimuli)
    
    # Assert
    assert np.isclose(r, 1.0, atol=0.01)
```

### Integration Tests
Test complete workflows:
```python
def test_full_correlation_pipeline():
    """Test entire pipeline from data loading to correlation computation."""
    # Load data
    loader = DataLoader('test_data.parquet')
    
    # Process
    filtered = filter_by_locomotion(loader.data, threshold=0.2)
    integrated = integrate_time_window(filtered, 0.5, 2.0)
    
    # Analyze
    correlations = compute_all_pairwise(integrated['spikes'], integrated['stimuli'])
    
    # Validate
    assert len(correlations) > 0
    assert -1 <= correlations.min() <= correlations.max() <= 1
```

### Validation Tests
Test against paper statistics:
```python
def test_mean_correlation_matches_paper(config):
    """Validate mean noise correlation matches paper's reported value."""
    # Run full analysis
    results = run_complete_analysis(config)
    
    # Check
    mean_corr = np.mean(results['all_correlations'])
    expected = config['validation']['expected_mean_correlation']
    
    assert np.isclose(mean_corr, expected, rtol=0.2)  # 20% tolerance
```

## When You Get Stuck

### 1. Check Paper Methods Section
The Methods section (pages 6-13) contains detailed descriptions. Key subsections:
- "Noise correlations in the visual stimulus-evoked responses of pairs of cells"
- "Evaluations of cortical coding fidelity"

### 2. Check MATHEMATICAL_FRAMEWORK.md
Contains all equations and computational details.

### 3. Check Expected Values
From paper Extended Data Fig 6:
- 5,008 responsive cells (out of 8,029 total)
- Mean firing rates: 0.1-1.5 Hz typical
- Response magnitudes: varies widely

### 4. Use Print Debugging
Add validation prints during development:
```python
print(f"Shape after filtering: {filtered.shape}")
print(f"Mean correlation: {np.mean(correlations):.3f}")
print(f"Expected: ~0.06")
```

## Commit Message Format
```
<type>(<scope>): <subject>

<body>

<validation results if applicable>
```

Examples:
```
test(correlations): Add test for noise correlation averaging

Validates that correlations are correctly averaged across both
stimulus conditions as specified in Methods.

feat(correlations): Implement pairwise correlation computation

Computes noise correlations for all neuron pairs following
Rumyantsev et al. 2020 methodology (Methods, page 9).

Performance: ~30s for 1000 neurons

fix(preprocessing): Correct time window indexing

Bug was using inclusive end bound, paper specifies [0.5, 2.0].
Now correctly filters time bins.

Validation: Trial counts now match expected range 217-332
```

## Progress Tracking

After completing each phase, update checklist:
```markdown
## Implementation Progress

- [ ] Phase 1: Project setup
- [ ] Phase 2: Data loading (12/12 tests passing)
- [ ] Phase 3: Preprocessing (8/8 tests passing)
- [ ] Phase 4: Correlation analysis (0/10 tests)
- [ ] Phase 5: Tuning similarity
- [ ] Phase 6: Statistical testing
- [ ] Phase 7: Visualization
```

## Final Validation Checklist

Before considering work complete:
```markdown
## Pre-Submission Checklist

Code Quality:
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Coverage >90% (`pytest --cov=src --cov-report=html`)
- [ ] No linting errors (`ruff check src/`)
- [ ] Type checking passes (`mypy src/`)

Scientific Validation:
- [ ] Mean correlation: 0.06 ± 20% tolerance
- [ ] Shuffled variance ratio: ~0.5
- [ ] KS test p-value: < 1.3×10⁻⁶
- [ ] Total pairs: ~7 million ± 10%
- [ ] Trial counts: 217-332 per stimulus

Deliverables:
- [ ] Figure 2d PNG (300 dpi)
- [ ] Figure 2e PNG (300 dpi)
- [ ] Jupyter notebook runs end-to-end
- [ ] README with reproduction instructions
- [ ] Summary statistics JSON

Documentation:
- [ ] All functions have docstrings
- [ ] Paper citations present
- [ ] Mathematical framework referenced
- [ ] Known limitations documented
```

## Remember

1. **Quality over speed** - Take time to write good tests
2. **Paper is ground truth** - Always refer back to methods
3. **Validate frequently** - Check intermediate results match expectations
4. **Document decisions** - Explain why you made implementation choices
5. **Ask for help** - Use comments to flag uncertainties

Good luck! This work will demonstrate both your technical skills and your 
ability to rigorously reproduce scientific results.