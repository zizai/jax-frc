# Validation Regression Reporting Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix console progress output and HTML report generation to match the design document.

**Architecture:** The validation cases have two modes: quick (NaN/Inf check only) and full (AGATE comparison). The full mode should display Field L2 tables, Scalar Metrics tables, and generate plots for the HTML report.

**Tech Stack:** Python, matplotlib, numpy, h5py

---

## Current Issues

1. **Quick test mode** - Works but intentionally skips detailed output (by design)
2. **Full test mode** - Needs verification that console output and HTML report match design
3. **Shape mismatch** - Previously fixed, but needs verification

---

## Task 1: Create Short Full-Mode Test Script

**Files:**
- Create: `validation/test_reporting.py`

**Step 1: Create a test script that runs full mode with minimal simulation**

```python
"""Quick verification of full-mode reporting output."""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Temporarily override resolutions and t_end for faster testing
import validation.cases.regression.orszag_tang as ot

# Save original values
orig_resolutions = ot.RESOLUTIONS
orig_quick_resolutions = ot.QUICK_RESOLUTIONS

# Override for fast test
ot.RESOLUTIONS = (256,)

# Patch setup_configuration to use shorter t_end
orig_setup = ot.setup_configuration
def fast_setup(quick_test, resolution):
    cfg = orig_setup(quick_test, resolution)
    cfg['t_end'] = 0.001  # Very short simulation
    return cfg
ot.setup_configuration = fast_setup

# Run full mode (not quick)
try:
    success = ot.main(quick_test=False)
finally:
    # Restore
    ot.RESOLUTIONS = orig_resolutions
    ot.QUICK_RESOLUTIONS = orig_quick_resolutions
    ot.setup_configuration = orig_setup

sys.exit(0 if success else 1)
```

**Step 2: Run the test script**

```bash
py validation/test_reporting.py
```

**Expected output format:**
```
Running validation: orszag_tang
  Orszag–Tang vortex regression vs AGATE reference data

Configuration:
  resolutions: (256,)
  L2 threshold: 0.01 (1%)
  Relative threshold: 0.05 (5% for energy metrics)

Downloading AGATE reference data...
  Resolution 256: OK

Resolution 256: [X.XXs]

  Field L2 Errors:
    Field            L2 Error   Threshold  Status
    ------------------------------------------------------
    density          X.XXXX     0.01       PASS/FAIL
    ...

  Scalar Metrics:
    Metric           JAX Value  AGATE Value  Rel Error  Threshold  Status
    -----------------------------------------------------------------------
    total_energy     XXX.X      XXX.X        X.XX%      5.0%       PASS/FAIL
    ...

  Summary: X/9 PASS

Report saved to: validation/reports/...

OVERALL: PASS/FAIL
```

---

## Task 2: Verify Console Output Format

**Files:** None (verification only)

**Step 1: Compare actual output with design document**

Check that the console output includes:
- [ ] Header with validation name and description
- [ ] Configuration section with resolutions and thresholds
- [ ] AGATE download status per resolution
- [ ] Per-resolution timing
- [ ] Field L2 Errors table with columns: Field, L2 Error, Threshold, Status
- [ ] Scalar Metrics table with columns: Metric, JAX Value, AGATE Value, Rel Error, Threshold, Status
- [ ] Summary line showing passed/total checks
- [ ] Report save path
- [ ] Overall PASS/FAIL status

**Step 2: Document any discrepancies**

If output doesn't match design, note what needs to be fixed.

---

## Task 3: Verify HTML Report Content

**Files:** None (verification only)

**Step 1: Open the generated HTML report**

Navigate to the report directory shown in console output.

**Step 2: Verify report sections**

Check that the HTML report contains:
- [ ] Header with PASS/FAIL badge
- [ ] Physics Background section with docstring
- [ ] Configuration table
- [ ] Results table with all metrics
- [ ] Plots section with:
  - [ ] Scalar comparison bar chart (JAX vs AGATE)
  - [ ] Error vs threshold summary (horizontal bars)
  - [ ] Density field comparison (3 panels: JAX, AGATE, Difference)
  - [ ] Magnetic field Bz comparison (3 panels)

---

## Task 4: Fix Any Issues Found

**Files:** Depends on issues found

If console output issues:
- Modify `validation/cases/regression/orszag_tang.py`
- Modify `validation/utils/reporting.py`

If HTML report issues:
- Modify `validation/utils/reporting.py` (for metrics table)
- Modify `validation/utils/plots.py` (for plot generation)
- Modify `validation/cases/regression/orszag_tang.py` (for plot calls)

---

## Task 5: Apply Same Fixes to reconnection_gem.py

**Files:**
- Modify: `validation/cases/regression/reconnection_gem.py`

Apply any fixes from Task 4 to the GEM reconnection validation case.

---

## Task 6: Clean Up Test Script

**Files:**
- Delete: `validation/test_reporting.py`

Remove the temporary test script after verification is complete.

---

## Verification Checklist

- [ ] Console output matches design document format
- [ ] HTML report has all required sections
- [ ] All 4 plot types are generated and visible
- [ ] Quick test mode still works
- [ ] Both orszag_tang and reconnection_gem produce correct output
