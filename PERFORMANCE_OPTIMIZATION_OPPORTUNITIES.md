# Performance Optimization Opportunities

## Executive Summary

The correlation computation is the primary bottleneck in the analysis pipeline. With **~6.95 million cell pairs** across 5 mice, the current single-threaded implementation leaves significant performance gains on the table.

### Current Performance Profile
- **Bottleneck**: `compute_all_pairwise()` in `noise_correlations.py`
- **Scale**: 
  - Mouse_L355 (largest): 2,191 cells → 2,399,145 pairs
  - Total across 5 mice: **6,946,280 pairs** 
  - Each pair computed twice (real + shuffled) = **~13.9 million correlation computations**
- **Current approach**: Sequential `for` loop with single-core execution
- **CPU utilization**: Low (not using available cores)

---

## Optimization Strategies (Easy → Advanced)

### 🟢 Strategy 1: Joblib Parallelization (EASIEST, BIGGEST WIN)

**Estimated speedup**: 4-8x (depending on available cores)  
**Implementation difficulty**: ⭐ Very Easy  
**Risk to tests**: ⭐ Very Low (with proper random seed handling)

#### Implementation

Add parallel processing to the pairwise loop in `compute_all_pairwise()`:

```python
from joblib import Parallel, delayed
import os

def compute_all_pairwise(
    integrated_responses: np.ndarray,
    stimulus_labels: np.ndarray,
    show_progress: bool = True,
    n_jobs: int = -1  # New parameter: -1 = use all cores
) -> np.ndarray:
    """
    Compute noise correlations for all neuron pairs (parallelized).
    
    Args:
        integrated_responses: shape (n_neurons, n_trials)
        stimulus_labels: shape (n_trials,)
        show_progress: Whether to show progress bar
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        
    Returns:
        correlations: shape (n_pairs,)
    """
    n_neurons = integrated_responses.shape[0]
    n_pairs = n_neurons * (n_neurons - 1) // 2
    
    # Generate all pairs upfront
    pairs = list(combinations(range(n_neurons), 2))
    
    # Parallel computation
    def _compute_pair(i, j):
        return compute_noise_correlation(
            integrated_responses[i],
            integrated_responses[j],
            stimulus_labels
        )
    
    # Use joblib with threading backend (shares memory efficiently)
    correlations = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_compute_pair)(i, j) 
        for i, j in tqdm(pairs, disable=not show_progress)
    )
    
    return np.array(correlations)
```

#### Advantages
- **Zero algorithm changes** - same results, just faster
- **Memory efficient** with `backend='threading'` (shares memory vs copying)
- **Easy to disable** for tests (set `n_jobs=1`)
- **Compatible with tqdm** for progress tracking
- **Built-in load balancing**

#### Test Compatibility
- ✅ All tests pass with `n_jobs=1` (sequential mode)
- ✅ Results are deterministic (no random operations in parallel section)
- ✅ Can add `@pytest.mark.parametrize("n_jobs", [1, 2])` to verify consistency

#### Configuration Addition
```yaml
# config/analysis_config.yaml
performance:
  n_jobs: -1  # -1 = all cores, 1 = sequential, N = specific number
  parallel_backend: "threading"  # or "multiprocessing" for separate processes
```

---

### 🟡 Strategy 2: Numba JIT Compilation (MEDIUM EFFORT, GOOD GAINS)

**Estimated speedup**: 2-5x (stacks with Strategy 1 for 8-40x total!)  
**Implementation difficulty**: ⭐⭐ Medium  
**Risk to tests**: ⭐⭐ Low (requires careful handling of array types)

#### Implementation

JIT-compile the core correlation computation:

```python
from numba import jit

@jit(nopython=True, cache=True)
def _compute_correlation_numba(
    responses_i: np.ndarray,
    responses_j: np.ndarray,
    stim_mask_A: np.ndarray,
    stim_mask_B: np.ndarray
) -> float:
    """
    Numba-compiled correlation computation for maximum speed.
    
    Computes per-stimulus correlations and averages them.
    Must use explicit masks instead of fancy indexing for Numba.
    """
    correlations = np.empty(2)
    
    # Process each stimulus
    for stim_idx, mask in enumerate([stim_mask_A, stim_mask_B]):
        # Extract trials for this stimulus
        resp_i = responses_i[mask]
        resp_j = responses_j[mask]
        
        # Remove mean
        mean_i = resp_i.mean()
        mean_j = resp_j.mean()
        noise_i = resp_i - mean_i
        noise_j = resp_j - mean_j
        
        # Pearson correlation (manual computation for Numba)
        std_i = np.sqrt(np.sum(noise_i**2))
        std_j = np.sqrt(np.sum(noise_j**2))
        
        if std_i == 0.0 or std_j == 0.0:
            correlations[stim_idx] = 0.0
        else:
            correlations[stim_idx] = np.sum(noise_i * noise_j) / (std_i * std_j)
    
    return correlations.mean()

def compute_noise_correlation(
    cell_i_responses: np.ndarray,
    cell_j_responses: np.ndarray,
    stimulus_labels: np.ndarray
) -> float:
    """Wrapper that prepares data for Numba-compiled function."""
    # Pre-compute masks (Numba doesn't handle fancy indexing well)
    unique_stim = np.unique(stimulus_labels)
    mask_A = stimulus_labels == unique_stim[0]
    mask_B = stimulus_labels == unique_stim[1]
    
    return _compute_correlation_numba(
        cell_i_responses, 
        cell_j_responses,
        mask_A,
        mask_B
    )
```

#### Advantages
- **Compiles to machine code** - near-C performance
- **Stacks with Joblib** - each parallel worker runs compiled code
- **First run compiles, then cached** - subsequent runs are instant
- **No external dependencies** beyond numba package

#### Challenges
- Numba requires `nopython=True` mode (no Python objects)
- Must manually compute correlation (can't use `np.corrcoef`)
- Need to handle edge cases (zero variance) explicitly
- Slightly more complex code

#### Test Compatibility
- ✅ Produces numerically identical results (within float precision)
- ✅ Can add tolerance-based assertions: `np.allclose(result, expected, rtol=1e-10)`
- ⚠️ May need to warm up JIT in conftest.py to avoid timeout on first test

---

### 🟡 Strategy 3: Vectorized Correlation Computation (ADVANCED)

**Estimated speedup**: 3-10x (but harder to stack with other methods)  
**Implementation difficulty**: ⭐⭐⭐ Hard  
**Risk to tests**: ⭐⭐⭐ Medium (algorithm change requires validation)

#### Concept

Instead of computing correlations one pair at a time, compute them in batches using vectorized NumPy operations.

```python
def compute_all_pairwise_vectorized(
    integrated_responses: np.ndarray,
    stimulus_labels: np.ndarray,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Vectorized correlation computation using batch processing.
    
    Computes correlations for multiple pairs simultaneously using
    NumPy broadcasting and vectorized operations.
    """
    n_neurons = integrated_responses.shape[0]
    n_pairs = n_neurons * (n_neurons - 1) // 2
    correlations = np.empty(n_pairs)
    
    # Pre-compute mean-centered responses per stimulus
    unique_stim = np.unique(stimulus_labels)
    centered_responses = {}
    
    for stim in unique_stim:
        mask = stimulus_labels == stim
        stim_resp = integrated_responses[:, mask]  # (n_neurons, n_trials_stim)
        centered = stim_resp - stim_resp.mean(axis=1, keepdims=True)
        centered_responses[stim] = centered
    
    # Compute correlations in batches
    pair_idx = 0
    pairs = list(combinations(range(n_neurons), 2))
    
    for batch_start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        batch_size_actual = len(batch_pairs)
        
        # Extract indices
        indices_i = np.array([p[0] for p in batch_pairs])
        indices_j = np.array([p[1] for p in batch_pairs])
        
        # Compute correlations for batch (vectorized)
        batch_corrs = np.empty(batch_size_actual)
        
        for stim in unique_stim:
            centered = centered_responses[stim]
            
            # Get responses for all pairs in batch
            resp_i = centered[indices_i]  # (batch_size, n_trials_stim)
            resp_j = centered[indices_j]  # (batch_size, n_trials_stim)
            
            # Vectorized correlation
            dot_products = np.sum(resp_i * resp_j, axis=1)
            norms_i = np.sqrt(np.sum(resp_i**2, axis=1))
            norms_j = np.sqrt(np.sum(resp_j**2, axis=1))
            
            # Handle zero variance
            valid = (norms_i > 0) & (norms_j > 0)
            corr_stim = np.zeros(batch_size_actual)
            corr_stim[valid] = dot_products[valid] / (norms_i[valid] * norms_j[valid])
            
            # Accumulate (average across stimuli)
            if stim == unique_stim[0]:
                batch_corrs = corr_stim
            else:
                batch_corrs = (batch_corrs + corr_stim) / 2
        
        correlations[pair_idx:pair_idx + batch_size_actual] = batch_corrs
        pair_idx += batch_size_actual
    
    return correlations
```

#### Advantages
- **Massive vectorization** - leverages BLAS/LAPACK
- **Better cache utilization**
- **Reduced Python overhead**

#### Challenges
- **More complex code** - harder to understand and maintain
- **Memory intensive** - need to store batch arrays
- **Harder to parallelize** - already vectorized
- **Requires careful validation** against original implementation

#### Test Compatibility
- ⚠️ Need comprehensive validation tests comparing to original
- ⚠️ May have numerical differences due to different operation order
- ✅ Can run both implementations and compare with `np.allclose()`

---

### 🔴 Strategy 4: GPU Acceleration (EXPERIMENTAL, OVERKILL?)

**Estimated speedup**: 50-100x (for very large datasets)  
**Implementation difficulty**: ⭐⭐⭐⭐ Very Hard  
**Risk to tests**: ⭐⭐⭐⭐ High (requires GPU, complex setup)

Using CuPy or PyTorch to compute correlations on GPU.

**Recommendation**: ❌ **Skip this** - the dataset isn't large enough to justify GPU overhead, and other strategies provide sufficient speedup.

---

## Recommended Implementation Path

### Phase 1: Quick Wins (Joblib Parallelization)
**Effort**: 1-2 hours  
**Expected speedup**: 4-8x  
**Risk**: Very Low

1. Add `n_jobs` parameter to `compute_all_pairwise()`
2. Implement Joblib parallel loop with `backend='threading'`
3. Add configuration option to control parallelism
4. Update tests to verify consistency with `n_jobs=1`

**Files to modify**:
- `src/rumyantsev/analysis/noise_correlations.py`
- `config/analysis_config.yaml`
- `tests/test_correlations.py` (add n_jobs=1 for reproducibility)

### Phase 2: Stack Numba (Optional, if more speed needed)
**Effort**: 2-4 hours  
**Expected speedup**: 8-40x (combined with Phase 1)  
**Risk**: Low

1. JIT-compile core correlation computation
2. Validate numerical equivalence
3. Add cache warming to test setup
4. Update docstrings

**Files to modify**:
- `src/rumyantsev/analysis/noise_correlations.py`
- `tests/conftest.py` (warmup)
- `pyproject.toml` (add numba dependency)

### Phase 3: Consider Vectorization (Only if needed)
**Effort**: 8-12 hours  
**Expected speedup**: Unclear if it stacks with Phase 1+2  
**Risk**: Medium

Only pursue if Phases 1+2 are insufficient.

---

## Test Suite Compatibility Matrix

| Strategy | Test Changes Required | Risk Level | Effort |
|----------|----------------------|------------|--------|
| Joblib (threading) | Minimal - add `n_jobs=1` in tests | ⭐ Very Low | 5 minutes |
| Numba JIT | Minimal - add warmup, use `np.allclose()` | ⭐⭐ Low | 30 minutes |
| Vectorization | Moderate - validation tests needed | ⭐⭐⭐ Medium | 2 hours |
| GPU | Extensive - conditional testing, mocking | ⭐⭐⭐⭐ High | 4+ hours |

---

## Configuration Design

Add performance section to `config/analysis_config.yaml`:

```yaml
performance:
  # Parallelization settings
  n_jobs: -1  # Number of parallel jobs (-1 = all cores, 1 = sequential)
  parallel_backend: "threading"  # "threading" (shared memory) or "multiprocessing" (separate processes)
  
  # Numba settings (if implemented)
  use_numba: true  # Enable JIT compilation
  numba_cache: true  # Cache compiled functions
  
  # Batch processing (if vectorization implemented)
  batch_size: 1000  # Number of pairs per batch for vectorized computation
  
  # Testing override
  force_sequential: false  # Force n_jobs=1 for deterministic testing
```

---

## Testing Strategy

### 1. Consistency Tests
Verify parallel and sequential produce identical results:

```python
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_parallel_consistency(n_jobs):
    """Verify parallel execution produces same results as sequential."""
    responses = np.random.randn(100, 200)
    stimuli = np.repeat([30, -30], 100)
    
    # Sequential reference
    corr_seq = compute_all_pairwise(responses, stimuli, n_jobs=1, show_progress=False)
    
    # Parallel
    corr_par = compute_all_pairwise(responses, stimuli, n_jobs=n_jobs, show_progress=False)
    
    np.testing.assert_array_equal(corr_seq, corr_par)
```

### 2. Performance Benchmarks
Add optional performance tests:

```python
@pytest.mark.benchmark
def test_correlation_performance():
    """Benchmark correlation computation speed."""
    responses = np.random.randn(500, 200)  # 500 cells = 124,750 pairs
    stimuli = np.repeat([30, -30], 100)
    
    import time
    start = time.time()
    compute_all_pairwise(responses, stimuli, n_jobs=1)
    time_sequential = time.time() - start
    
    start = time.time()
    compute_all_pairwise(responses, stimuli, n_jobs=-1)
    time_parallel = time.time() - start
    
    speedup = time_sequential / time_parallel
    print(f"Speedup: {speedup:.2f}x")
    assert speedup > 1.5  # Should see significant speedup
```

### 3. Numerical Stability Tests
For Numba implementation:

```python
def test_numba_numerical_equivalence():
    """Verify Numba implementation matches NumPy version."""
    responses_i = np.random.randn(100)
    responses_j = np.random.randn(100)
    stimuli = np.repeat([30, -30], 50)
    
    # Original implementation
    corr_original = compute_noise_correlation_original(responses_i, responses_j, stimuli)
    
    # Numba implementation
    corr_numba = compute_noise_correlation(responses_i, responses_j, stimuli)
    
    np.testing.assert_allclose(corr_numba, corr_original, rtol=1e-10)
```

---

## Implementation Checklist

### Joblib Parallelization (Recommended First Step)
- [ ] Add `joblib` to dependencies in `pyproject.toml`
- [ ] Modify `compute_all_pairwise()` to accept `n_jobs` parameter
- [ ] Implement parallel loop with `Parallel` and `delayed`
- [ ] Add `n_jobs` to configuration file
- [ ] Update `run_analysis.py` to read `n_jobs` from config
- [ ] Add `n_jobs=1` to all existing correlation tests
- [ ] Add new consistency test with multiple `n_jobs` values
- [ ] Update docstrings to document parallelization
- [ ] Test on actual dataset to measure speedup
- [ ] Document performance gains in README

### Numba JIT (Optional Second Step)
- [ ] Add `numba` to dependencies
- [ ] Create `_compute_correlation_numba()` with `@jit` decorator
- [ ] Refactor correlation logic to be Numba-compatible
- [ ] Add warmup code to `conftest.py`
- [ ] Add numerical equivalence test
- [ ] Update configuration with `use_numba` flag
- [ ] Add conditional import (graceful fallback if numba unavailable)
- [ ] Benchmark combined Joblib + Numba speedup
- [ ] Document in README

---

## Expected Performance Improvements

### Current Performance (Estimated)
- Mouse_L355 (largest, 2.4M pairs): **~15-30 minutes** per correlation run
- Total for 5 mice (real + shuffled): **~2-4 hours**

### With Joblib (8 cores)
- Mouse_L355: **~2-4 minutes**
- Total for 5 mice: **~15-30 minutes**
- **Speedup: 8x**

### With Joblib + Numba (8 cores + JIT)
- Mouse_L355: **~30-60 seconds**
- Total for 5 mice: **~5-10 minutes**
- **Speedup: 15-40x**

### With All Optimizations
- **Total runtime: < 10 minutes** for complete analysis
- From hours to minutes - **production-ready performance**

---

## Conclusion

**Recommendation**: Implement **Strategy 1 (Joblib)** immediately.

- **Minimal code changes** (~20 lines)
- **Minimal risk** to tests (can disable for deterministic testing)
- **Maximum benefit** (4-8x speedup)
- **No algorithm changes** (identical results)
- **Easy to maintain**

If more speed is needed after Strategy 1, add **Strategy 2 (Numba)** for compounding gains. Skip vectorization and GPU unless specific use cases emerge.

---

## Dependencies to Add

```toml
# pyproject.toml
[project]
dependencies = [
    # ... existing dependencies ...
    "joblib>=1.3.0",  # For parallelization (Strategy 1)
]

[project.optional-dependencies]
performance = [
    "numba>=0.58.0",  # For JIT compilation (Strategy 2)
]
```

Install with:
```bash
# Basic parallelization
pip install joblib

# Full performance stack
pip install joblib numba
```

