# Jupyter Notebook - Figure 2 Recreation

## Running the Analysis

The complete analysis pipeline is available in `reproduce_figure_2.ipynb`.

### Quick Start

```bash
cd /Users/luki/Projects/Stanford/rumyantsev-recreation
jupyter notebook notebooks/reproduce_figure_2.ipynb
```

### Expected Output

The notebook will generate:
- `outputs/figure_2d_recreation.png` - Noise correlation distribution
- `outputs/figure_2e_recreation.png` - Tuning similarity analysis
- `outputs/summary_statistics.json` - Summary metrics

### Validation Metrics

Expected results (from paper):
- Total neurons: 8,029
- Total pairs: ~6.95 million
- Mean correlation: 0.06 ± 0.01  
- Shuffled variance ratio: ~0.5
- KS test p-value: < 1.3×10⁻⁶

## Alternative: Python Script

For automated execution, run:

```bash
python run_analysis.py
```

