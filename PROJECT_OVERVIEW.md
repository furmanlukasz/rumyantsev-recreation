# Rumyantsev et al. 2020 - Figure 2d/e Recreation Project

## Project Goal
Recreate Figure 2d and 2e from "Fundamental bounds on the fidelity of sensory cortical coding" 
(Rumyantsev et al., Nature 2020) to demonstrate domain expertise in neuroscience data analysis 
for Stanford collaboration.

## Context
This is a validation task for Prof. Mark Schnitzer's lab at Stanford. Success requires:
1. Accurate reproduction of published results
2. Demonstration of systematic, test-driven methodology
3. Clear understanding of neuroscience concepts

## Paper Key Findings
- **Main Question**: Do correlated noise fluctuations limit neural coding accuracy?
- **Answer**: Yes, for ensembles >1000 neurons, information saturates
- **Key Insight**: Largest noise mode is orthogonal to signal (doesn't limit coding)
- **Result**: Weaker noise modes (~10x smaller) actually limit discrimination

## Figures to Recreate

### Figure 2d: Noise Correlation Distribution
- Histogram of pairwise noise correlations between neurons
- Compare real data vs trial-shuffled control
- Expected: Real data has broader distribution (mean r ≈ 0.06)
- Validation: Real data variance ~2x shuffled data variance

### Figure 2e: Tuning Similarity Analysis  
- Compare noise correlations for:
  - Similarly tuned cell pairs (positive mean response covariance)
  - Differently tuned cell pairs (negative mean response covariance)
- Expected: Similarly tuned pairs have higher noise correlations
- Validation: KS test p < 1.3×10⁻⁶

## Data Structure
**Input**: Parquet file with preprocessed neural data
- ~8,029 neurons across 5 mice
- ~350 trials per stimulus (±30° gratings)
- Spike counts already deconvolved from Ca²⁺ imaging
- Locomotion data for trial filtering

## Success Criteria
✅ Reproduce Figure 2d with matching statistics  
✅ Reproduce Figure 2e with matching statistics  
✅ All unit tests passing (>90% coverage)  
✅ Statistical validation matches paper  
✅ Clear documentation of methodology  

## Timeline
Estimated 32-46 hours for complete implementation with TDD approach.

## References
- Paper: Rumyantsev et al., Nature 2020 (included in project)
- Methods: See pages 6-13 of paper
- Key statistics: Extended Data Fig 6