# Multi-Model Magnetic Diffusion Validation

## Overview

Extend the magnetic diffusion validation to test both `ResistiveMHD` and `ExtendedMHD` models side-by-side, comparing their results against the analytic solution.

## Design

### Test Structure

```python
MODELS_TO_TEST = ["resistive_mhd", "extended_mhd"]

def main() -> bool:
    results = {}
    for model_type in MODELS_TO_TEST:
        config = MagneticDiffusionConfiguration(model_type=model_type, ...)
        state, geometry, _ = run_simulation(config)
        l2_err = compute_l2_error(state, analytic)
        results[model_type] = {"l2_error": l2_err, "state": state}

    overall_pass = all(r["l2_error"] < THRESHOLD for r in results.values())
```

### Reporting

Summary table format:
```
Model           L2 Error    Linf Error   Status
─────────────────────────────────────────────────
resistive_mhd   0.0058      0.0089       PASS
extended_mhd    0.0061      0.0092       PASS
```

### Visualization

Combined plots showing all three on same axes:
- ResistiveMHD (solid blue)
- ExtendedMHD (solid orange)
- Analytic (dashed black)

Plot types:
1. B_z profile at y=0 (x-slice)
2. B_z profile at x=0 (y-slice)
3. Error plot (simulation - analytic)

### Files to Modify

- `validation/cases/analytic/magnetic_diffusion.py` - Main changes

### Expected Behavior

Both models should produce nearly identical results for pure diffusion (no Hall term, no electron pressure). Any significant difference indicates a bug.
