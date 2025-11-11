# Time Window Integration Methods

## The Issue

The paper states: "we integrated the estimated spike count of each cell between [0.5 s, 2 s]"

But our data has **discrete bins** at 0.275s intervals:
- Bin 0 = [0.000s, 0.275s)
- Bin 1 = [0.275s, 0.550s)
- Bin 2 = [0.550s, 0.825s)
- ...
- Bin 7 = [1.925s, 2.200s)

## Available Methods

### Method 1: `discrete_conservative` (Current Default)

**Bins**: [2, 7] (inclusive)
**Actual Time**: [0.550s, 1.925s]
**Duration**: 1.375s

**Why this matters:**
- Slightly **shorter** than paper's [0.5s, 2.0s]
- More conservative - doesn't include any data before 0.5s
- May produce **slightly lower correlations**

### Method 2: `discrete_extended` (Alternative)

**Bins**: [1, 7] (inclusive)
**Actual Time**: [0.275s, 1.925s]
**Duration**: 1.650s

**Why this matters:**
- **Longer** integration window
- Includes bin 1 (0.275-0.55s) which partially overlaps with paper's 0.5s start
- May produce **slightly higher correlations** (closer to paper)
- Risk: includes 0.275s of "pre-stimulus" data

## How to Switch Methods

Edit `config/analysis_config.yaml`:

```yaml
preprocessing:
  integration_method: "discrete_conservative"  # Current
  # OR
  integration_method: "discrete_extended"      # Alternative
```

Then run:
```bash
python run_analysis.py
```

The script will show:
```
📊 Integration Method: discrete_extended
   Time window: bins [1, 7]
   Actual time: [0.275s, 1.925s]
   Duration: 1.650s
```

## Comparison Table

| Method | Bins | Actual Time | Duration | Expected Effect |
|--------|------|-------------|----------|-----------------|
| Paper (continuous) | N/A | [0.500s, 2.000s] | 1.500s | Reference |
| `discrete_conservative` | [2,7] | [0.550s, 1.925s] | 1.375s | Lower r (current) |
| `discrete_extended` | [1,7] | [0.275s, 1.925s] | 1.650s | Higher r (may match paper better) |

## Recommendation

**For Stanford validation:**
1. Run **both** methods
2. Compare results
3. Report which method produces mean r closer to paper's 0.06

**Hypothesis:**
- `discrete_extended` should yield **higher mean r** (closer to 0.06)
- This would validate that the time window mismatch explains the discrepancy

## Implementation Details

The config system:
- Stores all method definitions in `config/analysis_config.yaml`
- `run_analysis.py` reads the selected method
- Easy to add more methods if needed (e.g., interpolation-based continuous integration)

## Future Enhancement

Could implement **continuous interpolation** to get exact [0.5s, 2.0s]:
1. Interpolate between bin values
2. Integrate continuous function over [0.5s, 2.0s]
3. Would require more complex signal processing

For now, the discrete methods are simpler and likely sufficient.

