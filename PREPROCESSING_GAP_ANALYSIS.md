# Preprocessing Gap Analysis: What's Done vs. What's Missing

## Summary from Paper (via Gemini + PDF)

### Preprocessing Pipeline Described in Paper

1. **Deconvolution**: Fast non-negative deconvolution on ΔF/F₀ traces
2. **Temporal downsampling**: 2× downsampling (sum adjacent bins) → 0.276s bins
3. **Locomotion filtering**: Exclude trials with speed > 0.2 mm/s
4. **Time integration**: Integrate spike counts over [0.5s, 2.0s]
5. **Mean response subtraction**: "the mean stimulus-evoked response of the cell was subtracted from each trace **after separating the trials for each of the two visual stimuli**"
6. **Correlation computation**: Pearson correlation on residuals

---

## What's Already in Your Dataset ✅

Based on the parquet file structure and your analysis:

### ✅ **1. Deconvolution** - ALREADY DONE
- Your `amplitude` column contains deconvolved spike estimates
- Non-negative deconvolution ensures no negative values
- **Status**: Pre-applied to data

### ✅ **2. Temporal Downsampling** - ALREADY DONE  
- You have 14 time bins at 0.275s resolution (close to paper's 0.276s)
- This matches the 2× downsampling mentioned
- **Status**: Pre-applied to data

### ✅ **3. Locomotion Filtering** - ALREADY DONE
- No locomotion_speed column in dataset
- Trial counts (217-332 per stimulus) match paper's post-filtering range
- **Status**: Pre-applied to data

---

## What You're Currently Doing ✅

### ✅ **4. Time Integration** - CORRECTLY IMPLEMENTED

**Your code** (`trial_filtering.py` line 50):
```python
.agg(pl.col('amplitude').sum().alias('integrated_amplitude'))
```

**Status**: ✅ Correctly summing over time bins [start_bin, end_bin]

---

### ✅ **5. Mean Response Subtraction** - CORRECTLY IMPLEMENTED!

**Your code** (`noise_correlations.py` lines 56-63):
```python
for stim in unique_stimuli:
    # Extract trials for this stimulus
    mask = stimulus_labels == stim
    responses_i = cell_i_responses[mask]
    responses_j = cell_j_responses[mask]
    
    # Remove mean (isolate noise)
    noise_i = responses_i - responses_i.mean()
    noise_j = responses_j - responses_j.mean()
```

**Status**: ✅ **You ARE doing this correctly!**
- Separating trials by stimulus
- Subtracting mean per stimulus per cell
- Computing correlations on residuals

---

## 🤔 So Why the Discrepancy? (0.037 vs 0.06)

Since you're implementing the analysis pipeline **correctly**, the discrepancy must come from:

### Hypothesis 1: Amplitude Column Preprocessing ⚠️

The `amplitude` values in your parquet file might have **different preprocessing** than what the paper used in their analysis.

**What Gemini Found**:
- Paper used "fast non-negative deconvolution"
- BUT: No details on exact algorithm or parameters

**Possible issues**:
1. **Different deconvolution algorithm** used to generate your dataset
2. **Different deconvolution parameters** (tau, threshold, etc.)
3. **Additional preprocessing** applied to dataset that wasn't mentioned
4. **Dataset is from different analysis** (e.g., Extended Data uses threshold 0.5)

### Hypothesis 2: Time Window Difference ⚠️

**Paper**: [0.5s, 2.0s]  
**Your data**: 14 bins at 0.275s = [0s, 3.85s] total

**Current config** (`analysis_config.yaml`):
```yaml
integration_method: "discrete_extended"  # bins [1,7] = [0.275s, 2.20s]
```

**Bins to seconds**:
- Bin 1: 0.275s
- Bin 7: 2.20s (exclusive end)
- Duration: 1.925s

**Paper target**: [0.5s, 2.0s] = 1.5s duration

**Your window**: [0.275s, 2.20s] = 1.925s duration

This is actually **longer** than the paper's window and includes earlier times. Let me calculate what bins exactly match [0.5s, 2.0s]:

- 0.5s / 0.275s = 1.818 → **Start at bin 2** (0.55s)
- 2.0s / 0.275s = 7.273 → **End at bin 7** (2.20s)

You should test **bins [2,7]** (conservative method) vs bins [1,7] (extended method).

### Hypothesis 3: Spike Threshold ⚠️

**From Gemini**: Extended Data Fig 4c used threshold 0.5 for spike counts.

**Current implementation**: No thresholding applied.

**Test**: Apply threshold where `amplitude < 0.5` is set to 0 before time integration.

### Hypothesis 4: Data Source Mismatch ⚠️

**Critical question**: Is your dataset the **exact same data** used for Figure 2d/2e?

Possibilities:
- Dataset might be for different figures
- Dataset might be preliminary/processed version
- Dataset might include trials/cells that were filtered out in paper

---

## Action Plan: Systematic Testing

### Test 1: Time Window (Quick)

You already have configs for this:

```yaml
# Test bins [2,7] - conservative
integration_method: "discrete_conservative"

# vs bins [1,7] - extended  
integration_method: "discrete_extended"
```

**Run both and compare**:
```bash
# Edit config to use discrete_conservative
python run_analysis.py
# Note the mean correlation

# Edit config to use discrete_extended  
python run_analysis.py
# Note the mean correlation
```

### Test 2: Spike Threshold (Medium)

Implement amplitude thresholding:

**Add to `trial_filtering.py`** before integration:

```python
def apply_spike_threshold(
    data: pl.DataFrame,
    threshold: float = 0.5
) -> pl.DataFrame:
    """
    Apply spike detection threshold to amplitude values.
    
    Paper: Extended Data Fig 4c used threshold 0.5 for spike counts.
    
    Args:
        data: Neural data with 'amplitude' column
        threshold: Minimum amplitude to count as spike
        
    Returns:
        Data with thresholded amplitudes
    """
    return data.with_columns(
        pl.when(pl.col('amplitude') >= threshold)
        .then(pl.col('amplitude'))
        .otherwise(0.0)
        .alias('amplitude')
    )
```

**Update config**:
```yaml
preprocessing:
  # Spike detection threshold
  spike_threshold:
    enabled: true
    value: 0.5  # From Extended Data Fig 4c
```

**Test**: Run with threshold 0.5 and compare.

### Test 3: Ask Author (High Priority!)

**Email the author with specific questions**:

> Hi [Author],
>
> I'm working with the `coding_fidelity_bounds.dataset.parquet` file to recreate Figure 2d/2e from your Nature 2020 paper.
>
> I've implemented the analysis pipeline correctly (per-stimulus mean subtraction, time integration [0.5s-2.0s], correlation averaging), but I'm getting **mean noise correlation = 0.037** vs the paper's **0.06**.
>
> **Questions about the dataset**:
>
> 1. Does the `amplitude` column in the parquet file represent the **raw output** of fast non-negative deconvolution, or was additional preprocessing applied?
>
> 2. For the main noise correlation analysis (Fig 2d/2e), was any **spike threshold** applied to the amplitude values? (I see threshold 0.5 was used for Extended Data Fig 4c)
>
> 3. What are the **exact time bin indices** you used for the [0.5s, 2.0s] window? With 0.275s bins, I calculate:
>    - 0.5s / 0.275s = 1.82 → bin 2?
>    - 2.0s / 0.275s = 7.27 → bin 7?
>
> 4. Can you confirm the dataset is the **exact data** used for Figure 2d/2e?
>
> 5. Is there any **additional preprocessing** applied to amplitudes between deconvolution and correlation analysis that isn't mentioned in the methods?
>
> Thanks for your help!

---

## Current Status Summary

| Step | Paper Description | Your Implementation | Status |
|------|------------------|---------------------|--------|
| Deconvolution | Fast non-negative | Pre-applied in dataset | ✅ Done |
| Downsampling | 2× (0.276s bins) | 0.275s bins in data | ✅ Done |
| Locomotion filter | Speed < 0.2 mm/s | Pre-applied (no column) | ✅ Done |
| Time integration | [0.5s, 2.0s] | Bins [1,7] or [2,7] | ⚠️ Test both |
| Mean subtraction | Per stimulus, per cell | `noise_i = responses_i - mean` | ✅ Correct |
| Spike threshold | Mentioned for Extended Data Fig 4c | Not applied | ⚠️ Test 0.5 |
| Correlation | Pearson, averaged | `np.corrcoef()` averaged | ✅ Correct |

---

## Most Likely Causes (Ranked)

1. **Time window mismatch** (bins [1,7] vs [2,7])
   - Quick to test
   - Could significantly affect results
   
2. **Spike threshold 0.5** not applied
   - Would reduce noise floor
   - Increase correlation magnitude
   
3. **Amplitude preprocessing** in dataset differs from analysis
   - Need author confirmation
   - Can't fix without knowing what was done

