# Validation Regression Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve validation regression tests to have detailed console progress output and rich HTML report visualizations matching analytic validation quality.

**Architecture:** Add helper functions for tabular console output, create new plots.py module for visualization functions, modify regression validation files to use new reporting infrastructure. Field L2 errors compare spatial data, scalar metrics compare integrated quantities.

**Tech Stack:** Python, matplotlib, numpy, h5py (for AGATE data), jax_frc.validation.metrics

---

## Task 1: Add Console Output Helper Functions

**Files:**
- Modify: `validation/utils/reporting.py`

**Step 1: Add print_field_l2_table function**

Add at the end of `validation/utils/reporting.py`:

```python
def print_field_l2_table(field_errors: dict, threshold: float) -> None:
    """Print formatted table of field L2 errors to console."""
    print("  Field L2 Errors:")
    print(f"    {'Field':<18} {'L2 Error':<12} {'Threshold':<12} {'Status'}")
    print("    " + "-" * 54)
    for field, error in field_errors.items():
        status = "PASS" if error <= threshold else "FAIL"
        print(f"    {field:<18} {error:<12.4g} {threshold:<12.4g} {status}")
```

**Step 2: Add print_scalar_metrics_table function**

Add after print_field_l2_table:

```python
def print_scalar_metrics_table(metrics: dict) -> None:
    """Print formatted table of scalar metric comparisons to console."""
    print("  Scalar Metrics:")
    print(f"    {'Metric':<18} {'JAX Value':<12} {'AGATE Value':<12} "
          f"{'Rel Error':<10} {'Threshold':<10} {'Status'}")
    print("    " + "-" * 74)
    for name, data in metrics.items():
        jax_val = data['jax_value']
        agate_val = data['agate_value']
        rel_err = data['relative_error']
        threshold = data['threshold']
        status = "PASS" if data['passed'] else "FAIL"
        print(f"    {name:<18} {jax_val:<12.4g} {agate_val:<12.4g} "
              f"{rel_err:<10.2%} {threshold:<10.2%} {status}")
```

**Step 3: Commit**

```bash
git add validation/utils/reporting.py
git commit -m "feat(validation): add console output helper functions for field L2 and scalar metrics tables"
```

---

## Task 2: Create Plots Module with Scalar Comparison Plot

**Files:**
- Create: `validation/utils/plots.py`

**Step 1: Create plots.py with imports and scalar comparison plot**

```python
"""Plot generation functions for validation reports."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_scalar_comparison_plot(metrics: dict, resolution: int) -> Figure:
    """Create grouped bar chart comparing JAX vs AGATE scalar values.

    Args:
        metrics: Dict of metric name -> {jax_value, agate_value, ...}
        resolution: Grid resolution for title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(metrics.keys())
    jax_vals = [m['jax_value'] for m in metrics.values()]
    agate_vals = [m['agate_value'] for m in metrics.values()]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, jax_vals, width, label='JAX', color='steelblue')
    ax.bar(x + width/2, agate_vals, width, label='AGATE', color='coral')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(f'Scalar Metrics Comparison (Resolution {resolution})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    return fig
```

**Step 2: Commit**

```bash
git add validation/utils/plots.py
git commit -m "feat(validation): add plots module with scalar comparison bar chart"
```

---

## Task 3: Add Error Threshold Plot Function

**Files:**
- Modify: `validation/utils/plots.py`

**Step 1: Add create_error_threshold_plot function**

Append to `validation/utils/plots.py`:

```python
def create_error_threshold_plot(
    field_errors: dict,
    scalar_metrics: dict,
    l2_tol: float,
    rel_tol: float
) -> Figure:
    """Create horizontal bar chart showing errors relative to thresholds.

    Args:
        field_errors: Dict of field name -> L2 error value
        scalar_metrics: Dict of metric name -> {relative_error, threshold, passed}
        l2_tol: L2 error threshold for field comparisons
        rel_tol: Relative error threshold (unused, thresholds come from scalar_metrics)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine all errors with their thresholds
    items = []
    for name, error in field_errors.items():
        items.append((name, error, l2_tol, error <= l2_tol))
    for name, data in scalar_metrics.items():
        items.append((name, data['relative_error'], data['threshold'], data['passed']))

    names = [i[0] for i in items]
    errors = [i[1] for i in items]
    thresholds = [i[2] for i in items]
    passed = [i[3] for i in items]

    # Normalize errors to percentage of threshold
    normalized = [e/t * 100 if t > 0 else 0 for e, t in zip(errors, thresholds)]
    colors = ['green' if p else 'red' for p in passed]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, normalized, color=colors, alpha=0.7)
    ax.axvline(x=100, color='black', linestyle='--', linewidth=2, label='Threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Error (% of threshold)')
    ax.set_title('Error vs Threshold Summary')
    ax.legend()
    fig.tight_layout()
    return fig
```

**Step 2: Commit**

```bash
git add validation/utils/plots.py
git commit -m "feat(validation): add error vs threshold summary plot"
```

---

## Task 4: Add Field Comparison Plot Function

**Files:**
- Modify: `validation/utils/plots.py`

**Step 1: Add create_field_comparison_plot function**

Append to `validation/utils/plots.py`:

```python
def create_field_comparison_plot(
    jax_field: np.ndarray,
    agate_field: np.ndarray,
    field_name: str,
    resolution: int,
    l2_error_value: float
) -> Figure:
    """Create side-by-side contour plots: JAX, AGATE, and Difference.

    Args:
        jax_field: 2D array of JAX field values
        agate_field: 2D array of AGATE field values
        field_name: Name of the field for title
        resolution: Grid resolution for title
        l2_error_value: Pre-computed L2 error for annotation

    Returns:
        matplotlib Figure object
    """
    # Ensure numpy arrays
    jax_field = np.asarray(jax_field)
    agate_field = np.asarray(agate_field)

    # Compute difference
    diff_field = jax_field - agate_field

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common colorbar range for JAX and AGATE
    vmin = min(float(jax_field.min()), float(agate_field.min()))
    vmax = max(float(jax_field.max()), float(agate_field.max()))

    # JAX field
    im1 = axes[0].imshow(jax_field.T, origin='lower', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(f'JAX {field_name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    plt.colorbar(im1, ax=axes[0])

    # AGATE field
    im2 = axes[1].imshow(agate_field.T, origin='lower', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f'AGATE {field_name}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im2, ax=axes[1])

    # Difference (symmetric colormap)
    diff_abs_max = max(abs(float(diff_field.min())), abs(float(diff_field.max())))
    if diff_abs_max == 0:
        diff_abs_max = 1e-10  # Avoid zero range
    im3 = axes[2].imshow(diff_field.T, origin='lower', cmap='RdBu_r',
                          vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[2].set_title('Difference (JAX - AGATE)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    plt.colorbar(im3, ax=axes[2])

    # Add L2 error annotation
    fig.suptitle(f'{field_name.replace("_", " ").title()} Comparison '
                 f'(Resolution {resolution}) - L2 Error: {l2_error_value:.4g}')

    fig.tight_layout()
    return fig
```

**Step 2: Commit**

```bash
git add validation/utils/plots.py
git commit -m "feat(validation): add field comparison contour plot with difference panel"
```

---

## Task 5: Add AGATE Field Loading Function

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Add load_agate_final_fields function**

Add after the existing `load_agate_series` function:

```python
def load_agate_final_fields(case: str, resolution: int) -> dict:
    """Load final state spatial fields from AGATE reference data.

    Args:
        case: Case identifier ("ot" for Orszag-Tang)
        resolution: Grid resolution

    Returns:
        Dict with keys: density, momentum, magnetic_field, pressure
        Each value is a numpy array of the spatial field.
    """
    loader = AgateDataLoader()
    loader.ensure_files(case, resolution)
    case_dir = Path(loader.cache_dir) / case / str(resolution)

    def _state_key(path: Path) -> int:
        match = re.search(r"state_(\d+)", path.name)
        return int(match.group(1)) if match else 0

    state_files = sorted(case_dir.rglob("*.state_*.h5"), key=_state_key)
    if not state_files:
        raise FileNotFoundError(f"No AGATE state files found in {case_dir}")

    # Load the final state file
    final_state_path = state_files[-1]

    with h5py.File(final_state_path, "r") as f:
        sub = f["subID0"]
        vec = sub["vector"][:]

    # Parse the state vector into fields
    rho, p, v, B = _parse_state_vector(vec)

    # Compute momentum from density and velocity
    mom = rho[..., None] * v

    return {
        "density": rho,
        "momentum": mom,
        "magnetic_field": B,
        "pressure": p,
    }
```

**Step 2: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): add AGATE final field loading function"
```

---

## Task 6: Add Field L2 Error Computation Function

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Add compute_field_l2_errors function**

Add after `load_agate_final_fields`:

```python
def compute_field_l2_errors(jax_state, agate_fields: dict) -> dict:
    """Compute L2 errors between JAX and AGATE spatial fields.

    Args:
        jax_state: JAX simulation final state with n, v, B, p attributes
        agate_fields: Dict from load_agate_final_fields

    Returns:
        Dict of field name -> L2 error value
    """
    from jax_frc.validation.metrics import l2_error

    errors = {}

    # Density
    errors['density'] = float(l2_error(
        np.asarray(jax_state.n),
        agate_fields['density']
    ))

    # Momentum (rho * v)
    jax_mom = np.asarray(jax_state.n)[..., None] * np.asarray(jax_state.v)
    errors['momentum'] = float(l2_error(jax_mom, agate_fields['momentum']))

    # Magnetic field
    errors['magnetic_field'] = float(l2_error(
        np.asarray(jax_state.B),
        agate_fields['magnetic_field']
    ))

    # Pressure
    errors['pressure'] = float(l2_error(
        np.asarray(jax_state.p),
        agate_fields['pressure']
    ))

    return errors
```

**Step 2: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): add field L2 error computation function"
```

---

## Task 7: Update Main Function - Console Output

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Add imports at top of file**

Add these imports after existing imports:

```python
from validation.utils.reporting import (
    ValidationReport,
    print_field_l2_table,
    print_scalar_metrics_table,
)
from validation.utils.plots import (
    create_scalar_comparison_plot,
    create_error_threshold_plot,
    create_field_comparison_plot,
)
```

**Step 2: Update main function header to print configuration**

Replace the beginning of main() up to the resolution loop with:

```python
def main(quick_test: bool = False) -> bool:
    print(f"Running validation: {NAME}")
    print(f"  {DESCRIPTION}")
    if quick_test:
        print("  (QUICK TEST MODE)")
    print()

    print("Configuration:")
    resolutions = QUICK_RESOLUTIONS if quick_test else RESOLUTIONS
    print(f"  resolutions: {resolutions}")
    print(f"  L2 threshold: {L2_ERROR_TOL} ({L2_ERROR_TOL*100:.0f}%)")
    print(f"  Relative threshold: {RELATIVE_ERROR_TOL} ({RELATIVE_ERROR_TOL*100:.0f}% for energy metrics)")
    print()

    overall_pass = True
    all_results = {}
    all_metrics = {}

    # Download AGATE data before running simulations
    print("Downloading AGATE reference data...")
    for resolution in resolutions:
        try:
            loader = AgateDataLoader()
            loader.ensure_files("ot", resolution)
            print(f"  Resolution {resolution}: OK")
        except Exception as exc:
            print(f"  Resolution {resolution}: FAILED ({exc})")
    print()
```

**Step 3: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): update main function header with configuration output"
```

---

## Task 8: Update Main Function - Resolution Loop with Tables

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Replace the resolution loop**

Replace the for loop over resolutions with:

```python
    for resolution in resolutions:
        print(f"Resolution {resolution}: ", end="", flush=True)
        cfg = setup_configuration(quick_test, resolution)
        t_start = time.time()
        final_state, geometry, history = run_simulation(cfg)
        t_sim = time.time() - t_start
        print(f"[{t_sim:.2f}s]")

        if quick_test:
            # Quick test: just check for NaN/Inf
            jax_metrics = {key: np.array([m[key] for m in history["metrics"]])
                          for key in history["metrics"][0]}
            for key in jax_metrics:
                val = float(jax_metrics[key][-1])
                is_valid = not (np.isnan(val) or np.isinf(val))
                if not is_valid:
                    overall_pass = False
                all_metrics[f"{key}_r{resolution}"] = {
                    "value": val,
                    "passed": is_valid,
                    "description": "Quick test mode (NaN/Inf check only)",
                }
            print(f"  Quick test: {'PASS' if is_valid else 'FAIL'} (NaN/Inf check)")
            continue

        # Load AGATE reference data
        try:
            agate_fields = load_agate_final_fields("ot", resolution)
            agate_times, agate_scalar_metrics = load_agate_series("ot", resolution)
        except Exception as exc:
            print(f"  ERROR: Failed to load AGATE data: {exc}")
            overall_pass = False
            continue

        # Compute field L2 errors
        field_errors = compute_field_l2_errors(final_state, agate_fields)
        print_field_l2_table(field_errors, L2_ERROR_TOL)
        print()

        # Compute scalar metrics comparison
        jax_final_metrics = compute_metrics(
            final_state.n, final_state.p, final_state.v, final_state.B,
            geometry.dx, geometry.dy, geometry.dz
        )

        scalar_results = {}
        for key in jax_final_metrics:
            jax_val = jax_final_metrics[key]
            agate_val = float(agate_scalar_metrics[key][-1])
            passed, stats = compare_final_values(jax_val, agate_val, key)
            scalar_results[key] = {
                'jax_value': jax_val,
                'agate_value': agate_val,
                'relative_error': stats.get('relative_error', 0),
                'threshold': stats.get('threshold', RELATIVE_ERROR_TOL),
                'passed': passed,
            }

        print_scalar_metrics_table(scalar_results)
        print()

        # Summary for this resolution
        field_passed = sum(1 for e in field_errors.values() if e <= L2_ERROR_TOL)
        scalar_passed = sum(1 for m in scalar_results.values() if m['passed'])
        total_checks = len(field_errors) + len(scalar_results)
        total_passed = field_passed + scalar_passed
        res_pass = total_passed == total_checks
        overall_pass = overall_pass and res_pass

        print(f"  Summary: {total_passed}/{total_checks} PASS")
        print()

        # Store results for report generation
        all_results[resolution] = {
            'field_errors': field_errors,
            'scalar_metrics': scalar_results,
            'jax_state': final_state,
            'agate_fields': agate_fields,
        }

        # Store metrics for report
        for field, error in field_errors.items():
            all_metrics[f"{field}_l2_r{resolution}"] = {
                'value': error,
                'threshold': L2_ERROR_TOL,
                'passed': error <= L2_ERROR_TOL,
                'description': f'{field} L2 error vs AGATE',
            }
        for key, data in scalar_results.items():
            all_metrics[f"{key}_r{resolution}"] = {
                'jax_value': data['jax_value'],
                'agate_value': data['agate_value'],
                'relative_error': data['relative_error'],
                'threshold': data['threshold'],
                'passed': data['passed'],
                'description': f'{key} relative error vs AGATE',
            }
```

**Step 2: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): update resolution loop with field L2 and scalar metric tables"
```

---

## Task 9: Update Main Function - Report Generation with Plots

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Replace report generation section**

Replace the report generation code at the end of main() with:

```python
    # Generate report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration={
            "resolutions": resolutions,
            "L2_threshold": L2_ERROR_TOL,
            "relative_threshold": RELATIVE_ERROR_TOL,
        },
        metrics=all_metrics,
        overall_pass=overall_pass,
    )

    # Generate plots for each resolution (skip in quick test mode)
    if not quick_test:
        for resolution, data in all_results.items():
            # Plot 1: Scalar metrics comparison (bar chart)
            fig_scalar = create_scalar_comparison_plot(
                data['scalar_metrics'], resolution
            )
            report.add_plot(fig_scalar, name=f"scalar_comparison_r{resolution}",
                           caption=f"JAX vs AGATE scalar metrics at resolution {resolution}")
            plt.close(fig_scalar)

            # Plot 2: Error vs threshold summary
            fig_error = create_error_threshold_plot(
                data['field_errors'], data['scalar_metrics'],
                L2_ERROR_TOL, RELATIVE_ERROR_TOL
            )
            report.add_plot(fig_error, name=f"error_summary_r{resolution}",
                           caption=f"All errors as percentage of threshold at resolution {resolution}")
            plt.close(fig_error)

            # Plot 3: Field comparison contours (density)
            jax_density = np.asarray(data['jax_state'].n)[:, 0, :]
            agate_density = data['agate_fields']['density'][:, 0, :]
            fig_density = create_field_comparison_plot(
                jax_density, agate_density,
                'density', resolution, data['field_errors']['density']
            )
            report.add_plot(fig_density, name=f"density_comparison_r{resolution}",
                           caption=f"Density field comparison at resolution {resolution}")
            plt.close(fig_density)

            # Plot 4: Field comparison contours (magnetic field Bz)
            jax_bz = np.asarray(data['jax_state'].B)[:, 0, :, 2]
            agate_bz = data['agate_fields']['magnetic_field'][:, 0, :, 2]
            fig_bz = create_field_comparison_plot(
                jax_bz, agate_bz,
                'magnetic_field_Bz', resolution, data['field_errors']['magnetic_field']
            )
            report.add_plot(fig_bz, name=f"magnetic_field_comparison_r{resolution}",
                           caption=f"Magnetic field Bz comparison at resolution {resolution}")
            plt.close(fig_bz)

    report_dir = report.save()
    print(f"Report saved to: {report_dir}")
    print()

    # Final result
    if overall_pass:
        print("OVERALL: PASS (all resolutions passed)")
    else:
        print("OVERALL: FAIL (some checks failed)")

    return bool(overall_pass)
```

**Step 2: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): add plot generation to report for orszag_tang"
```

---

## Task 10: Apply Same Changes to reconnection_gem.py

**Files:**
- Modify: `validation/cases/regression/reconnection_gem.py`

**Step 1: Add the same imports**

Add after existing imports:

```python
from validation.utils.reporting import (
    ValidationReport,
    print_field_l2_table,
    print_scalar_metrics_table,
)
from validation.utils.plots import (
    create_scalar_comparison_plot,
    create_error_threshold_plot,
    create_field_comparison_plot,
)
```

**Step 2: Add load_agate_final_fields function**

Copy the same function from orszag_tang.py, changing "ot" to "gem" in the docstring.

**Step 3: Add compute_field_l2_errors function**

Copy the same function from orszag_tang.py.

**Step 4: Update main function**

Apply the same changes to main() as in orszag_tang.py, but use "gem" instead of "ot" for the case identifier.

**Step 5: Commit**

```bash
git add validation/cases/regression/reconnection_gem.py
git commit -m "feat(validation): apply same reporting improvements to reconnection_gem"
```

---

## Task 11: Run Validation Tests

**Step 1: Run orszag_tang quick test**

```bash
py -m validation.cases.regression.orszag_tang --quick
```

Expected: PASS with console output showing configuration and quick test results.

**Step 2: Run reconnection_gem quick test**

```bash
py -m validation.cases.regression.reconnection_gem --quick
```

Expected: PASS with console output showing configuration and quick test results.

**Step 3: Commit any fixes if needed**

---

## Task 12: Final Verification and Commit

**Step 1: Run full validation (optional, takes longer)**

```bash
py -m validation.cases.regression.orszag_tang
```

Expected: Full console output with Field L2 Errors table, Scalar Metrics table, and HTML report with plots.

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat(validation): complete regression reporting improvements

- Add console output helper functions (print_field_l2_table, print_scalar_metrics_table)
- Add plots module with scalar comparison, error threshold, and field comparison plots
- Update orszag_tang and reconnection_gem with detailed progress output
- Generate rich HTML reports with visualizations"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Console output helpers | `validation/utils/reporting.py` |
| 2 | Scalar comparison plot | `validation/utils/plots.py` (new) |
| 3 | Error threshold plot | `validation/utils/plots.py` |
| 4 | Field comparison plot | `validation/utils/plots.py` |
| 5 | AGATE field loading | `validation/cases/regression/orszag_tang.py` |
| 6 | Field L2 error computation | `validation/cases/regression/orszag_tang.py` |
| 7 | Main function header | `validation/cases/regression/orszag_tang.py` |
| 8 | Resolution loop with tables | `validation/cases/regression/orszag_tang.py` |
| 9 | Report generation with plots | `validation/cases/regression/orszag_tang.py` |
| 10 | Apply to reconnection_gem | `validation/cases/regression/reconnection_gem.py` |
| 11 | Run validation tests | - |
| 12 | Final verification | - |
