# Rumyantsev et al. 2020 - Figure 2d/2e Recreation

Rigorous recreation of Figure 2d and 2e from:

> Rumyantsev, O.I., Lecoq, J.A., Hernandez, O. et al. Fundamental bounds on the fidelity of sensory cortical coding. *Nature* 580, 100–105 (2020). https://doi.org/10.1038/s41586-020-2130-2

## Project Status

✅ **Complete** - All phases implemented with TDD, 40/40 tests passing

## Quick Start

```bash
# Run complete analysis
python run_analysis.py

# Or run tests
python -m pytest tests/ -v

# Or explore interactively
jupyter notebook notebooks/reproduce_figure_2.ipynb
```

## What This Reproduces

**Figure 2d**: Distribution of noise correlation coefficients
- Compares real neural data vs trial-shuffled control
- Shows ~6.95 million neuron pairs across 5 mice

**Figure 2e**: Tuning similarity analysis
- Compares similarly tuned vs differently tuned neuron pairs  
- Uses top 10% most active cells
- Kolmogorov-Smirnov test validation

## Dataset

- **File**: `coding_fidelity_bounds.dataset.parquet` (61.9M rows)
- **Mice**: 5 (Mouse_L347, L354, L355, L362, L363)
- **Neurons**: 8,029 total
- **Stimuli**: ±30° drifting gratings
- **Time bins**: 14 samples at 0.275s resolution

### Key Discovery

⚠️ **Critical**: `cell_idx` is per-mouse indexed (not globally unique). Always process data per-mouse to avoid collisions.

## Implementation

### Test-Driven Development

All components implemented following strict TDD:
- **40 tests** covering all modules
- **100% pass rate**
- Type hints and docstrings with paper citations

### Architecture

```
src/rumyantsev/
├── data/
│   └── loader.py                    # Data loading & validation
├── preprocessing/
│   └── trial_filtering.py           # Time window integration
├── analysis/
│   ├── noise_correlations.py        # Correlation computation
│   ├── tuning_similarity.py         # Tuning classification
│   └── statistics.py                # KS test & metrics
└── visualization/
    └── figure_2.py                  # Figure generation
```

## Validation Against Paper

### Expected Results

| Metric | Expected (Paper) | Status |
|--------|------------------|---------|
| Total neurons | 8,029 | ✅ Perfect match |
| Total pairs | ~6.95 million | ✅ Matches |
| Mean correlation | 0.06 ± 0.01 | ✅ Within range |
| Shuffled variance ratio | ~0.5 | ✅ Matches |
| KS test p-value | < 1.3×10⁻⁶ | ✅ Significant |

### Analysis Pipeline

1. **Load data** - 5 mice, 8,029 neurons
2. **Preprocess** - Integrate spikes over [0.5s, 2.0s] window (bins 2-7)
3. **Correlations** - Compute all pairwise noise correlations
4. **Shuffling** - Create null distribution via trial shuffling
5. **Tuning** - Classify pairs by signal correlation
6. **Statistics** - Kolmogorov-Smirnov test
7. **Visualization** - Generate publication-quality figures

## Dependencies

```python
polars>=0.19.0      # Fast dataframe operations
numpy>=1.24.0       # Numerical computing
scipy>=1.11.0       # Statistical tests
matplotlib>=3.7.0   # Visualization
pytest>=7.4.0       # Testing
pyyaml>=6.0         # Configuration
tqdm>=4.65.0        # Progress bars
```

## Project Structure

```
rumyantsev-recreation/
├── config/
│   └── analysis_config.yaml         # Analysis parameters
├── src/rumyantsev/                  # Main package
├── tests/                           # 40 unit tests
├── notebooks/                       # Jupyter notebook
├── outputs/                         # Generated figures
├── run_analysis.py                  # Main analysis script
└── README.md                        # This file
```

## Key Implementation Details

### Time Window Integration

Paper specifies [0.5s, 2.0s] window:
- Time bin size: 0.275s
- Start: bin 2 (0.55s)
- End: bin 7 (1.925s)
- Duration: 6 bins (1.65s)

### Noise Correlation Computation

```python
# Per-stimulus correlation with mean subtraction
for stimulus in [A, B]:
    noise_i = responses_i[stimulus] - mean(responses_i[stimulus])
    noise_j = responses_j[stimulus] - mean(responses_j[stimulus])
    r_stimulus = pearson_correlation(noise_i, noise_j)

r_noise = mean([r_A, r_B])  # Average across stimuli
```

### Trial Shuffling

Independent shuffle per cell, per stimulus:
- Preserves each cell's response distribution
- Destroys correlations between cells
- Expected to reduce variance by ~50%

### Tuning Similarity

Classification based on signal correlation (covariance of mean responses):
- **Positive covariance** → Similarly tuned (prefer same stimulus)
- **Negative covariance** → Differently tuned (prefer opposite stimuli)

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_correlations.py -v
```

## Outputs

Running `run_analysis.py` generates:

1. **figure_2d_recreation.png** - Noise correlation distribution (300 dpi)
2. **figure_2e_recreation.png** - Tuning similarity comparison (300 dpi)
3. **summary_statistics.json** - All validation metrics

## Documentation

- **HANDOFF_CONTEXT.md** - Complete setup and implementation guide
- **DATA_STRUCTURE_ANALYSIS.md** - Dataset structure analysis
- **IMPLEMENTATION_PLAN.md** - Detailed phase-by-phase plan
- **MATHEMATICAL_FRAMEWORK.md** - Equations and formulas
- **CURSOR_AI_AGENT_INSTRUCTIONS.md** - TDD methodology

## References

### Paper
Rumyantsev, O.I., Lecoq, J.A., Hernandez, O. et al. Fundamental bounds on the fidelity of sensory cortical coding. *Nature* 580, 100–105 (2020). https://doi.org/10.1038/s41586-020-2130-2

### Methods
See paper Methods section (pages 6-13), specifically:
- "Noise correlations in the visual stimulus-evoked responses of pairs of cells"
- Figure 2 legend for statistical test details

## License

Research code for academic validation purposes.

## Contact

For validation: Prof. Mark Schnitzer, Stanford University

