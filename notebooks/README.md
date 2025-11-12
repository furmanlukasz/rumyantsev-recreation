# Jupyter Notebooks

## Interactive Analysis Notebook

This directory contains an interactive Jupyter notebook for exploring the Rumyantsev et al. (2020) Figure 2 recreation.

### File

- **`reproduce_figure_2.ipynb`** - Complete walkthrough of the analysis with visualizations

### Installation

To use the notebook, install the package with notebook support:

```bash
# Using pip
pip install -e ".[notebook]"

# Using UV
uv pip install -e ".[notebook]"
```

This installs:
- `jupyter` - Jupyter notebook server
- `ipykernel` - Python kernel for Jupyter
- `notebook` - Jupyter notebook application

### Running the Notebook

```bash
# Start Jupyter Notebook
jupyter notebook reproduce_figure_2.ipynb

# Or use Jupyter Lab
jupyter lab reproduce_figure_2.ipynb

# Or start from the notebooks directory
cd notebooks
jupyter notebook
```

### What's Inside

The notebook provides an interactive walkthrough of:

1. **Data Loading**
   - Reading the parquet dataset
   - Validating structure and counts
   - Per-mouse data organization

2. **Preprocessing**
   - Time window integration [0.5s, 2.0s]
   - Matrix reshaping (neurons × trials)
   - Stimulus label extraction

3. **Noise Correlation Analysis**
   - Per-stimulus mean subtraction
   - Pairwise correlation computation
   - Trial shuffling for null distribution

4. **Tuning Similarity Analysis**
   - Top 10% active cell selection
   - Similarly vs differently tuned classification
   - Correlation comparison

5. **Statistical Testing**
   - Kolmogorov-Smirnov test
   - Summary statistics
   - Validation against paper

6. **Visualization**
   - Figure 2d - Noise correlation distributions
   - Figure 2e - Tuning similarity comparison
   - Interactive plots

### Features

- ✅ **Step-by-step execution** - Run cells individually
- ✅ **Inline visualizations** - See results immediately
- ✅ **Parameter modification** - Test different settings
- ✅ **Detailed explanations** - Understand each step
- ✅ **Markdown documentation** - Method descriptions with paper citations

### Troubleshooting

#### Kernel Not Found

```bash
# Make sure ipykernel is installed
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=rumyantsev-recreation
```

#### Import Errors

```bash
# Make sure the package is installed
pip install -e .

# Or with notebook dependencies
pip install -e ".[notebook]"
```

#### Dataset Not Found

Make sure `coding_fidelity_bounds.dataset.parquet` is in the project root directory (one level up from notebooks/).

### Alternative: Run Without Installation

If you prefer not to install the package, you can run the analysis scripts directly:

```bash
# From project root
python run_analysis.py              # Full analysis
python regenerate_figures.py        # Just re-plot figures
```

The notebook is best for:
- Learning how the analysis works
- Experimenting with parameters
- Interactive exploration
- Understanding the methodology

The scripts are best for:
- Quick reproduction of results
- Automated pipelines
- Batch processing

