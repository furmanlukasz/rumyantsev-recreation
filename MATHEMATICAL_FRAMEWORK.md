# Mathematical Framework - Noise Correlation Analysis

## 1. Core Concepts

### 1.1 Neural Response Model
For neuron $i$, trial $k$, stimulus $s \in \{A, B\}$:

$$r_i^{(k)}(s) = \mu_i(s) + \eta_i^{(k)}(s)$$

Where:
- $\mu_i(s)$ = mean response of neuron $i$ to stimulus $s$
- $\eta_i^{(k)}(s)$ = trial-to-trial noise fluctuation

**Implementation Note**: 
```python
# Spike counts integrated over [0.5s, 2.0s] window
response = spike_counts[:, time_bins_in_window].sum(axis=1)
mean_response = response.mean(axis=0)  # over trials
noise = response - mean_response  # per trial
```

---

## 2. Noise Correlation Coefficient

### 2.1 Definition
For neuron pair $(i, j)$, noise correlation averaged across both stimuli:

$$r_{ij}^{\text{noise}} = \frac{1}{2}\left[r_{ij}^{\text{noise}}(A) + r_{ij}^{\text{noise}}(B)\right]$$

Where for each stimulus $s$:

$$r_{ij}^{\text{noise}}(s) = \frac{\text{Cov}(\eta_i^{(k)}(s), \eta_j^{(k)}(s))}{\sigma_i(s) \cdot \sigma_j(s)}$$

### 2.2 Computational Steps
```python
def compute_noise_correlation(cell_i_responses, cell_j_responses, stimulus_labels):
    """
    Args:
        cell_i_responses: shape (n_trials,) - integrated spike counts
        cell_j_responses: shape (n_trials,) - integrated spike counts  
        stimulus_labels: shape (n_trials,) - 'A' or 'B' per trial
    
    Returns:
        r_noise: float - noise correlation coefficient
    """
    correlations = []
    
    for stim in ['A', 'B']:
        # Extract trials for this stimulus
        mask = stimulus_labels == stim
        responses_i = cell_i_responses[mask]
        responses_j = cell_j_responses[mask]
        
        # Remove mean (isolate noise)
        noise_i = responses_i - responses_i.mean()
        noise_j = responses_j - responses_j.mean()
        
        # Pearson correlation of noise
        r = np.corrcoef(noise_i, noise_j)[0, 1]
        correlations.append(r)
    
    # Average across stimuli
    return np.mean(correlations)
```

### 2.3 Expected Values (from paper)
- **Real data**: Mean $r^{\text{noise}} = 0.06 \pm 0.01$ (mean ± SD across mice)
- **Shuffled data**: Mean $r^{\text{noise}} \approx 0$, variance ~50% of real data
- **Total pairs**: 6,946,280 across 5 mice

---

## 3. Tuning Similarity Classification

### 3.1 Signal Correlation
For each cell pair, compute covariance of **mean responses**:

$$\text{Cov}(\mu_i, \mu_j) = (\mu_i(A) - \bar{\mu}_i)(\mu_j(A) - \bar{\mu}_j) + (\mu_i(B) - \bar{\mu}_i)(\mu_j(B) - \bar{\mu}_j)$$

Where $\bar{\mu}_i = \frac{\mu_i(A) + \mu_i(B)}{2}$

### 3.2 Classification
```
If Cov(μᵢ, μⱼ) > 0 → "Similarly tuned"  
If Cov(μᵢ, μⱼ) < 0 → "Differently tuned"
```

**Interpretation**:
- **Positive**: Both cells prefer same stimulus (e.g., both prefer A over B)
- **Negative**: Cells prefer opposite stimuli (one prefers A, other prefers B)

### 3.3 Implementation
```python
def classify_tuning_similarity(mean_response_A_i, mean_response_B_i,
                                mean_response_A_j, mean_response_B_j):
    """
    Classify whether two cells are similarly or differently tuned.
    
    Returns:
        'similar' or 'different'
    """
    # Center responses around mean
    mean_i = (mean_response_A_i + mean_response_B_i) / 2
    mean_j = (mean_response_A_j + mean_response_B_j) / 2
    
    # Compute covariance
    cov = ((mean_response_A_i - mean_i) * (mean_response_A_j - mean_j) + 
           (mean_response_B_i - mean_i) * (mean_response_B_j - mean_j))
    
    return 'similar' if cov > 0 else 'different'
```

---

## 4. Trial Shuffling (Control Analysis)

### 4.1 Purpose
Create null distribution where noise correlations are preserved within each cell 
but correlations **between** cells are destroyed.

### 4.2 Procedure
For each cell $i$ and each stimulus $s$:
1. Take all trial responses: $\{r_i^{(1)}(s), r_i^{(2)}(s), ..., r_i^{(P)}(s)\}$
2. Randomly permute trial indices: $\{\pi(1), \pi(2), ..., \pi(P)\}$
3. Reassign: $r_i^{(k)}(s) \leftarrow r_i^{(\pi(k))}(s)$

**Key**: Each cell gets a **different** random permutation.

### 4.3 Implementation
```python
def shuffle_trials(responses, stimulus_labels, random_seed=None):
    """
    Shuffle trials independently for each cell.
    
    Args:
        responses: shape (n_cells, n_trials)
        stimulus_labels: shape (n_trials,)
    
    Returns:
        shuffled_responses: shape (n_cells, n_trials)
    """
    rng = np.random.default_rng(random_seed)
    shuffled = np.empty_like(responses)
    
    for cell_idx in range(responses.shape[0]):
        for stim in ['A', 'B']:
            mask = stimulus_labels == stim
            stim_trials = responses[cell_idx, mask]
            
            # Independent shuffle per cell
            shuffled_trials = rng.permutation(stim_trials)
            shuffled[cell_idx, mask] = shuffled_trials
    
    return shuffled
```

---

## 5. Statistical Testing

### 5.1 Kolmogorov-Smirnov Test
Compare distributions of noise correlations for similarly vs differently tuned pairs.

**Null Hypothesis**: Both distributions are identical  
**Alternative**: Distributions differ

$$D = \sup_x |F_{\text{sim}}(x) - F_{\text{diff}}(x)|$$

Where $F$ are cumulative distribution functions.

### 5.2 Expected Result (from paper)
- Test statistic: $D$ (varies by mouse)
- p-value: $< 1.3 \times 10^{-6}$ for all 5 mice
- Effect size: Mean difference $\approx 2\times$ between groups

### 5.3 Implementation
```python
from scipy.stats import ks_2samp

def compare_correlation_distributions(sim_tuned_corrs, diff_tuned_corrs):
    """
    Perform KS test comparing two correlation distributions.
    
    Returns:
        statistic: float - KS statistic D
        pvalue: float - two-tailed p-value
    """
    statistic, pvalue = ks_2samp(sim_tuned_corrs, diff_tuned_corrs)
    return statistic, pvalue
```

---

## 6. Key Validation Metrics

### 6.1 Distribution Statistics (Figure 2d)

For **all cell pairs** (real data):
```
✓ Mean r^noise ≈ 0.06 ± 0.01 (across 5 mice)
✓ Distribution is Gaussian-like but wider than shuffled
✓ Variance(real) ≈ 2 × Variance(shuffled)
✓ Total pairs ≈ 7 million
```

For **shuffled data**:
```
✓ Mean r^noise ≈ 0 (by construction)
✓ Distribution is narrow Gaussian
✓ Reflects finite-sample estimation noise only
```

### 6.2 Tuning Comparison Statistics (Figure 2e)

Using **top 10% most active cells**:
```
✓ Similarly tuned: Mean r^noise ≈ 0.12
✓ Differently tuned: Mean r^noise ≈ 0.06  
✓ Ratio ≈ 2:1
✓ KS test: p < 1.3×10⁻⁶ (highly significant)
```

---

## 7. Common Pitfalls & Edge Cases

### 7.1 Locomotion Filtering
**Issue**: Paper uses speed < 0.2 mm/s threshold  
**Impact**: Reduces trials from 350 → 217-332 per stimulus  
**Validation**: Check retained trial count matches this range

### 7.2 Time Window Selection  
**Issue**: Must use [0.5s, 2.0s] window (not [0, 2s])  
**Reason**: Excludes transient onset response, focuses on sustained activity  
**Validation**: Visual response traces reach plateau by 0.5s (Extended Data Fig 7d)

### 7.3 Downsampling
**Issue**: Paper downsamples 2× to 0.275s bins  
**Impact**: Reduces temporal resolution but improves SNR  
**Validation**: Check time bin size matches paper specification

### 7.4 Finite Sample Bias
**Issue**: With P trials, correlation estimates have bias ~ 1/P  
**Impact**: Even shuffled data shows non-zero correlations  
**Solution**: This is expected and handled by comparing real vs shuffled

---

## 8. Dimensional Analysis Checks

Ensure all shapes are consistent:
```python
# Expected dimensions
n_mice = 5
n_cells_per_mouse ≈ 1000-2000
n_total_cells ≈ 8029
n_trials_initial = 350 per stimulus
n_trials_filtered ≈ 217-332 per stimulus
n_time_bins_in_window ≈ (2.0 - 0.5) / 0.275 ≈ 5-6 bins

# Per mouse data structure
responses: (n_cells, n_trials, n_time_bins)
stimulus_labels: (n_trials,)
locomotion_speed: (n_trials,)

# After integration
integrated_responses: (n_cells, n_trials)  # summed over time
noise_correlations: (n_cells, n_cells)  # symmetric matrix

# Pairwise correlations (flattened)
all_pairs: (n_cells * (n_cells - 1) / 2,)  # upper triangle
```

---

## 9. References to Paper Sections

- **Methods**: Pages 6-13 (especially "Noise correlations..." section)
- **Extended Data Fig 6**: Cell response statistics
- **Extended Data Fig 7**: Time course validation  
- **Figure 2 legend**: Statistical test details
- **Supplementary Table 1**: Exact p-values per mouse