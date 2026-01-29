# Validation Regression Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify and finalize the validation regression reporting implementation that matches the design document.

**Architecture:** The implementation adds console progress output (Field L2 tables, Scalar Metrics tables) and HTML report visualizations (bar charts, error summaries, field comparison plots) to the regression validation cases.

**Tech Stack:** Python, matplotlib, numpy, h5py, jax_frc.validation.metrics

---

## Current State

The implementation is **already complete** from a previous session. Key changes made:

1. **`validation/utils/reporting.py`** - Added `print_field_l2_table()` and `print_scalar_metrics_table()`
2. **`validation/utils/plots.py`** - Created with `create_scalar_comparison_plot()`, `create_error_threshold_plot()`, `create_field_comparison_plot()`
3. **`validation/cases/regression/orszag_tang.py`** - Restructured main(), added field L2 computation, fixed AGATE data loading
4. **`validation/cases/regression/reconnection_gem.py`** - Same changes as orszag_tang.py

**Critical Fix Applied:** AGATE data shape transformation:
- Strip ghost cells: `[2:-2, 2:-2, :]`
- Transpose from xy-plane to xz-plane: `np.transpose(rho, (0, 2, 1))`
- Swap velocity/B components: `v[..., [0, 2, 1]]`

---

## Task 1: Verify Quick Test Mode

**Files:** None (verification only)

**Step 1: Run orszag_tang quick test**

```bash
py -m validation.cases.regression.orszag_tang --quick
```

**Expected output:**
```
Running validation: orszag_tang
  Orszag–Tang vortex regression vs AGATE reference data
  (QUICK TEST MODE)

Configuration:
  resolutions: (256,)
  L2 threshold: 0.01 (1%)
  Relative threshold: 0.05 (5% for energy metrics)

Resolution 256: [X.XXs]
  Quick test: PASS (NaN/Inf check)

Report saved to: validation/reports/YYYY-MM-DDTHH-MM-SS_orszag_tang

OVERALL: PASS (all resolutions passed)
```

**Step 2: Run reconnection_gem quick test**

```bash
py -m validation.cases.regression.reconnection_gem --quick
```

**Expected:** Similar output with PASS status

---

## Task 2: Verify Full Validation Mode (Single Resolution)

**Files:** None (verification only)

**Step 1: Modify orszag_tang.py temporarily to run only resolution 256**

Edit `validation/cases/regression/orszag_tang.py` line 47:
```python
# Change from:
RESOLUTIONS = (256, 512, 1024)
# To:
RESOLUTIONS = (256,)
```

**Step 2: Run full validation**

```bash
py -m validation.cases.regression.orszag_tang
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
    momentum         X.XXXX     0.01       PASS/FAIL
    magnetic_field   X.XXXX     0.01       PASS/FAIL
    pressure         X.XXXX     0.01       PASS/FAIL

  Scalar Metrics:
    Metric           JAX Value  AGATE Value  Rel Error  Threshold  Status
    -----------------------------------------------------------------------
    total_energy     XXX.X      XXX.X        X.XX%      5.0%       PASS/FAIL
    magnetic_energy  XXX.X      XXX.X        X.XX%      5.0%       PASS/FAIL
    kinetic_energy   XXX.X      XXX.X        X.XX%      5.0%       PASS/FAIL
    enstrophy        XXX.X      XXX.X        X.XX%      1.0%       PASS/FAIL
    max_current      X.XXX      X.XXX        X.XX%      1.0%       PASS/FAIL

  Summary: X/9 PASS

Report saved to: validation/reports/YYYY-MM-DDTHH-MM-SS_orszag_tang

OVERALL: PASS/FAIL
```

**Step 3: Revert the RESOLUTIONS change**

```python
RESOLUTIONS = (256, 512, 1024)
```

---

## Task 3: Verify HTML Report Content

**Files:** None (verification only)

**Step 1: Open the generated HTML report**

Navigate to `validation/reports/` and open the most recent `*_orszag_tang/report.html`

**Step 2: Verify report sections**

Check that the report contains:
- [ ] Header with PASS/FAIL badge
- [ ] Physics Background section
- [ ] Configuration table with resolutions, thresholds
- [ ] Results table with all metrics (field L2 errors + scalar metrics)
- [ ] Plots section with:
  - [ ] Scalar comparison bar chart
  - [ ] Error vs threshold summary
  - [ ] Density field comparison (3 panels)
  - [ ] Magnetic field Bz comparison (3 panels)

---

## Task 4: Document Any Issues Found

**Files:**
- Create: `docs/plans/2026-01-29-validation-issues.md` (if issues found)

**Step 1: If any verification fails, document the issue**

Include:
- What failed
- Error message or unexpected output
- Proposed fix

**Step 2: If all verifications pass, mark implementation complete**

Update the design document status from "Draft" to "Implemented".

---

## Verification Checklist

After all tasks complete:

- [ ] Quick test mode passes for both validation cases
- [ ] Full validation mode produces correct console output format
- [ ] HTML report contains all required sections
- [ ] Plots are visible and correctly formatted
- [ ] No shape mismatch errors between JAX and AGATE data

## Notes

The implementation handles the coordinate system difference between AGATE (xy-plane) and JAX (xz-plane):
- AGATE stores data as `(nx+4, ny+4, 1)` with 2 ghost cells on each side
- JAX stores data as `(nx, 1, nz)` without ghost cells
- The `load_agate_final_fields` function transforms AGATE data to match JAX format
