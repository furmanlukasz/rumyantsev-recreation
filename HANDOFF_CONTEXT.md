# Handoff Context - Ready for Implementation

**Date**: November 11, 2025  
**Status**: ✅ Setup complete, dataset analyzed, ready for TDD implementation

---

## What's Been Done

### ✅ Environment Setup
- UV package manager installed (`~/.local/bin/uv`)
- Virtual environment created (`.venv/`)
- Core packages installed: `polars`, `pyarrow`, `numpy`

### ✅ Dataset Analysis Complete
- **File**: `coding_fidelity_bounds.dataset.parquet` (61.9M rows)
- **Structure**: Fully understood and documented
- **Validation**: All key metrics match paper expectations

### ✅ Key Discoveries

1. **Perfect neuron count**: 8,029 neurons across 5 mice ✅
2. **Cell indexing**: `cell_idx` is per-mouse (0-indexed), process each mouse separately
3. **Time bins**: 14 samples (0-13), corresponding to ~3.85s at 0.275s/bin
4. **Stimuli**: `behavior` column has values 30 and -30 (degrees)
5. **Pre-filtered data**: No `locomotion_speed` column; data appears pre-filtered

---

## Dataset Column Mapping

| Actual Column | Expected Column | Mapping |
|--------------|----------------|---------|
| `mouse_id` | `mouse_id` | Direct ✅ |
| `cell_idx` | `neuron_id` | Per-mouse indexed ⚠️ |
| `trial_idx` | `trial_id` | Direct ✅ |
| `sample_idx` | `time_bin` | Direct ✅ |
| `behavior` | `stimulus_type` | Map: 30→'A', -30→'B' |
| `amplitude` | `spike_count` | Direct ✅ |

---

## Key Parameters

```yaml
Data Structure:
  - 5 mice with 1,031-2,191 cells each (total: 8,029)
  - 14 time bins per trial (sample_idx: 0-13)
  - 435-662 total trials per mouse (~217-331 per stimulus)
  - 2 stimuli (±30° gratings)

Time Window [0.5s, 2.0s]:
  - Time bin size: 0.275s
  - Start bin: 2 (0.55s)
  - End bin: 7 (1.925s)
  - Bins to use: 2, 3, 4, 5, 6, 7 (6 bins, 1.65s duration)

Expected Results (from paper):
  - Mean noise correlation: 0.06 ± 0.01
  - Total cell pairs: ~6.95 million
  - Shuffled variance ratio: ~0.5
  - KS test p-value: < 1.3×10⁻⁶
```

---

## Next Steps (Implementation Order)

### Phase 1: Project Structure (1-2 hours)
```bash
# Create directory structure
mkdir -p src/rumyantsev/{data,preprocessing,analysis,visualization}
mkdir -p tests config outputs notebooks

# Initialize pyproject.toml with dependencies
uv add numpy scipy matplotlib seaborn pytest pytest-cov pyyaml tqdm
```

### Phase 2: Configuration (30 min)
Create `config/analysis_config.yaml` with:
- Column mappings
- Time window parameters (bins 2-7)
- Validation thresholds
- Stimulus mapping (30→'A', -30→'B')

### Phase 3: Data Loader (TDD, 3-4 hours)
**File**: `src/rumyantsev/data/loader.py`

Key requirements:
- Load parquet with polars
- Process per-mouse (avoid cell_idx collisions)
- Map columns to expected names
- Validate structure

**Critical**: Always filter by `mouse_id` first!

### Phase 4: Preprocessing (TDD, 2-3 hours)
**File**: `src/rumyantsev/preprocessing/trial_filtering.py`

Functions needed:
1. `integrate_time_window()` - Sum amplitude over bins 2-7
2. `reshape_to_matrix()` - Convert to (n_cells, n_trials) array
3. `extract_stimulus_labels()` - Map behavior → ['A', 'B']

### Phase 5: Correlation Analysis (TDD, 6-8 hours)
**File**: `src/rumyantsev/analysis/noise_correlations.py`

Functions needed:
1. `compute_noise_correlation()` - Pearson r after mean subtraction
2. `compute_all_pairwise()` - All cell pairs within mouse
3. `shuffle_trials()` - Independent shuffle per cell per stimulus
4. `compute_shuffled_correlations()` - Apply to shuffled data

### Phase 6: Tuning Similarity (TDD, 3-4 hours)
**File**: `src/rumyantsev/analysis/tuning_similarity.py`

Functions needed:
1. `classify_tuning()` - Covariance of mean responses
2. `select_top_active()` - Top 10% by √(r_A² + r_B²)
3. `group_by_tuning()` - Separate similar/different pairs

### Phase 7: Statistics & Visualization (4-5 hours)
**Files**: 
- `src/rumyantsev/analysis/statistics.py`
- `src/rumyantsev/visualization/figure_2.py`

Deliverables:
1. KS test implementation
2. Figure 2d (histogram, real vs shuffled)
3. Figure 2e (similar vs different tuning)

### Phase 8: Integration & Validation (3-4 hours)
**File**: `notebooks/reproduce_figure_2.ipynb`

End-to-end pipeline validation against paper metrics.

---

## Critical Implementation Notes

### 🚨 Must Process Per-Mouse
```python
# CORRECT: Process each mouse separately
for mouse_id in df['mouse_id'].unique():
    mouse_data = df.filter(pl.col('mouse_id') == mouse_id)
    # cell_idx is now unique within this subset
    
# WRONG: Process all at once
all_cells = df['cell_idx'].unique()  # ❌ Only 2,191 instead of 8,029
```

### ⏱️ Time Window Conversion
```python
# Map seconds to bin indices
def time_to_bin(time_seconds: float) -> int:
    return round(time_seconds / 0.275)

# [0.5s, 2.0s] → bins [2, 7]
start_bin = time_to_bin(0.5)  # = 2
end_bin = time_to_bin(2.0)    # = 7

# Extract and integrate
window_data = data.filter(
    (pl.col('sample_idx') >= start_bin) & 
    (pl.col('sample_idx') <= end_bin)
)
integrated = window_data.group_by(['cell_idx', 'trial_idx']).agg(
    pl.col('amplitude').sum().alias('spike_count')
)
```

### 🎯 Stimulus Mapping
```python
stimulus_map = {30: 'A', -30: 'B'}
df = df.with_columns(
    pl.col('behavior').map_dict(stimulus_map).alias('stimulus_type')
)
```

### 📊 Expected Validation Results
```python
# After processing all 5 mice:
assert total_neurons == 8029, f"Expected 8029, got {total_neurons}"
assert 6.9e6 <= total_pairs <= 7.0e6, f"Expected ~6.95M pairs, got {total_pairs}"
assert 0.05 <= mean_corr <= 0.07, f"Expected 0.06±0.01, got {mean_corr}"
assert ks_pvalue < 1.3e-6, f"Expected p<1.3e-6, got {ks_pvalue}"
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `CURSOR_AI_AGENT_INSTRUCTIONS.md` | TDD methodology, coding standards |
| `MATHEMATICAL_FRAMEWORK.md` | Equations, formulas, validation metrics |
| `PROJECT_OVERVIEW.md` | High-level goals, context |
| `IMPLEMENTATION_PLAN.md` | Original plan (needs minor adjustments) |
| `DATA_STRUCTURE_ANALYSIS.md` | Complete dataset analysis |
| `inspect_data.py` | Data inspection script (can delete later) |
| `verify_cell_idx.py` | Cell indexing verification (can delete later) |

---

## Test-Driven Development Checklist

For each component:
1. ✅ Write test first (RED)
2. ✅ Run test - should fail
3. ✅ Write minimal code (GREEN)
4. ✅ Refactor if needed
5. ✅ Commit with clear message
6. ✅ Cite paper in docstrings

---

## Success Criteria

### Code Quality
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Coverage >90%
- [ ] Type hints on all functions
- [ ] Docstrings with paper citations

### Scientific Validation
- [ ] Total neurons: 8,029
- [ ] Total pairs: ~6.95 million
- [ ] Mean correlation: 0.06 ± 20% tolerance
- [ ] Shuffled variance ratio: ~0.5
- [ ] KS test p < 1.3×10⁻⁶

### Deliverables
- [ ] Figure 2d PNG (300 dpi)
- [ ] Figure 2e PNG (300 dpi)
- [ ] Jupyter notebook (end-to-end)
- [ ] Summary statistics JSON

---

## Quick Start Commands

```bash
# Navigate to project
cd /Users/luki/Projects/Stanford/rumyantsev-recreation

# Install dependencies
/Users/luki/.local/bin/uv pip install numpy scipy matplotlib seaborn pytest pytest-cov pyyaml tqdm polars pyarrow

# Create structure
mkdir -p src/rumyantsev/{data,preprocessing,analysis,visualization} tests config outputs notebooks

# Start with tests (TDD)
touch tests/conftest.py
touch tests/test_data_loader.py

# Run tests
pytest tests/ -v
```

---

## Questions for Implementation

1. **Locomotion filtering**: Skip this step (data appears pre-filtered). Verify trial counts match expected range.

2. **Cell indexing**: Always process per-mouse to avoid collisions in `cell_idx`.

3. **Time bins**: Use bins 2-7 for [0.5s, 2.0s] window (6 bins total).

4. **Responsive cells**: Dataset has 8,029 cells total. Paper mentions 5,008 responsive. May need filtering based on response magnitude (check Extended Data Fig 6).

---

## Contact / Validation

**Validation Target**: Prof. Mark Schnitzer, Stanford University  
**Goal**: Demonstrate rigorous reproduction of published results  
**Timeline**: Estimated 25-35 hours remaining (setup: 4h ✅, implementation: 25-35h)

---

**Status**: 🟢 Ready to begin TDD implementation  
**Next Agent**: Start with Phase 1 (Project Structure)

