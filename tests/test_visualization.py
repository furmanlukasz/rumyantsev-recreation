"""
Tests for visualization module.

Basic tests to ensure figures can be generated without errors.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')  # Non-interactive backend for testing


def test_figure_2d_creation():
    """
    Test Figure 2d histogram generation.
    
    Paper Figure 2d: Noise correlation distribution comparison.
    """
    from rumyantsev.visualization.figure_2 import create_figure_2d

    # Mock correlation data
    rng = np.random.default_rng(42)
    real_corr = rng.normal(0.06, 0.03, 100000)
    shuffled_corr = rng.normal(0.0, 0.015, 100000)
    
    fig, ax = create_figure_2d(real_corr, shuffled_corr)
    
    assert fig is not None
    assert ax is not None
    
    # Check legend has correct labels
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert 'Real data' in legend_labels
    assert 'Shuffled' in legend_labels
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figure_2e_creation():
    """
    Test Figure 2e tuning similarity histogram generation.
    
    Paper Figure 2e: Similarly vs differently tuned pairs comparison.
    """
    from rumyantsev.visualization.figure_2 import create_figure_2e

    # Mock correlation data
    rng = np.random.default_rng(42)
    sim_tuned_corr = rng.normal(0.08, 0.03, 50000)
    diff_tuned_corr = rng.normal(0.04, 0.03, 50000)
    pvalue = 1.2e-10
    
    fig, ax = create_figure_2e(sim_tuned_corr, diff_tuned_corr, pvalue)
    
    assert fig is not None
    assert ax is not None
    
    # Check legend
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert 'Similarly tuned' in legend_labels
    assert 'Differently tuned' in legend_labels
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_figures_have_correct_labels():
    """Test that figures have proper axis labels."""
    from rumyantsev.visualization.figure_2 import (create_figure_2d,
                                                   create_figure_2e)
    
    rng = np.random.default_rng(42)
    
    # Test Figure 2d
    fig1, ax1 = create_figure_2d(
        rng.normal(0.06, 0.03, 1000),
        rng.normal(0.0, 0.015, 1000)
    )
    
    assert ax1.get_xlabel() == 'Correlation coefficient'
    assert ax1.get_ylabel() == 'Number of cell pairs'
    
    # Test Figure 2e
    fig2, ax2 = create_figure_2e(
        rng.normal(0.08, 0.03, 1000),
        rng.normal(0.04, 0.03, 1000),
        pvalue=1e-10
    )
    
    assert ax2.get_xlabel() == 'Correlation coefficient'
    assert ax2.get_ylabel() == 'Probability density'
    
    # Clean up
    import matplotlib.pyplot as plt
    plt.close(fig1)
    plt.close(fig2)

