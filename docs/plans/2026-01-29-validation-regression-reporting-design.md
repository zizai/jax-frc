# Validation Regression Reporting Improvements

**Date:** 2026-01-29
**Status:** Draft
**Author:** Claude (brainstorming session)

## Overview

Improve validation regression tests to have full parity with analytic validation style, including detailed progress reporting during execution and comprehensive final report visualizations.

## Goals

1. **Progress Reporting** - Detailed console output showing field L2 errors and scalar metrics per resolution
2. **Final Report Visualizations** - Rich plots comparing JAX vs AGATE data
3. **Parity with Analytic Validation** - Match the reporting quality of `validation/cases/analytic/magnetic_diffusion.py`

## Design

### 1. Console Progress Output

The console output will display detailed tables per resolution, showing both field L2 errors and scalar metrics:

```
Running validation: orszag_tang
  Orszag–Tang vortex regression vs AGATE reference data

Configuration:
  resolutions: (256, 512, 1024)
  L2 threshold: 0.01 (1%)
  Relative threshold: 0.05 (5% for energy metrics)

Downloading AGATE reference data...
  Resolution 256: OK
  Resolution 512: OK

Resolution 256: [5.2s]
  Field L2 Errors:
    Field            L2 Error   Threshold  Status
    ------------------------------------------------
    density          0.0012     0.01       PASS
    momentum         0.0018     0.01       PASS
    magnetic_field   0.0015     0.01       PASS
    pressure         0.0020     0.01       PASS

  Scalar Metrics:
    Metric           JAX Value  AGATE Value  Rel Error  Threshold  Status
    -----------------------------------------------------------------------
    total_energy     196.0      195.8        0.10%      5.0%       PASS
    magnetic_energy  119.2      119.0        0.17%      5.0%       PASS
    kinetic_energy   27.42      27.38        0.15%      5.0%       PASS
    enstrophy        248.0      247.5        0.20%      1.0%       PASS
    max_current      2.928      2.925        0.10%      1.0%       PASS

  Summary: 9/9 PASS

Resolution 512: [18.3s]
  ...

Report saved to: validation/reports/2026-01-29T10-30-00_orszag_tang

OVERALL: PASS (all resolutions passed)
```

### 2. HTML Report Visualizations

The final HTML report will include three types of visualizations:

#### 2.1 Scalar Metrics Comparison (Bar Charts)

Grouped bar chart comparing JAX vs AGATE final values for each scalar metric:
- X-axis: Metric names (total_energy, magnetic_energy, etc.)
- Y-axis: Values
- Two bars per metric: JAX (blue) and AGATE (coral)

#### 2.2 Error vs Threshold Summary

Horizontal bar chart showing errors as percentage of threshold:
- Y-axis: All metrics (field L2 errors + scalar metrics)
- X-axis: Error as percentage of threshold (0-100%+)
- Color coding: Green for PASS, Red for FAIL
- Vertical line at 100% marking the threshold

#### 2.3 Field Comparison Plots (2D Contours)

Side-by-side 2D contour plots for key fields (density, magnetic_field):
- Three panels: JAX field, AGATE field, Difference (JAX - AGATE)
- Common colorbar for JAX and AGATE panels
- Symmetric diverging colormap (RdBu) for difference panel
- L2 error annotation in figure title

### 3. Implementation Changes

#### 3.1 New Helper Functions in `validation/utils/reporting.py`

```python
def print_field_l2_table(field_errors: dict, threshold: float):
    """Print formatted table of field L2 errors."""
    print("  Field L2 Errors:")
    print(f"    {'Field':<18} {'L2 Error':<12} {'Threshold':<12} {'Status'}")
    print("    " + "-" * 54)
    for field, error in field_errors.items():
        status = "PASS" if error <= threshold else "FAIL"
        print(f"    {field:<18} {error:<12.4g} {threshold:<12.4g} {status}")

def print_scalar_metrics_table(metrics: dict):
    """Print formatted table of scalar metric comparisons."""
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

#### 3.2 New Plot Functions in `validation/utils/plots.py`

```python
def create_scalar_comparison_plot(metrics: dict, resolution: int) -> Figure:
    """Create grouped bar chart comparing JAX vs AGATE scalar values."""
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

def create_error_threshold_plot(field_errors: dict, scalar_metrics: dict,
                                 l2_tol: float, rel_tol: float) -> Figure:
    """Create horizontal bar chart showing errors relative to thresholds."""
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
    normalized = [e/t * 100 for e, t in zip(errors, thresholds)]
    colors = ['green' if p else 'red' for p in passed]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, normalized, color=colors, alpha=0.7)
    ax.axvline(x=100, color='black', linestyle='--', label='Threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Error (% of threshold)')
    ax.set_title('Error vs Threshold Summary')
    ax.legend()
    fig.tight_layout()
    return fig

def create_field_comparison_plot(jax_state, agate_fields: dict,
                                  field_name: str, resolution: int) -> Figure:
    """Create side-by-side contour plots: JAX, AGATE, and Difference."""
    from jax_frc.validation.metrics import l2_error

    # Extract the appropriate field (2D slice)
    if field_name == 'density':
        jax_field = jax_state.n[:, :, 0]
        agate_field = agate_fields['density'][:, :, 0]
    elif field_name == 'magnetic_field':
        jax_field = jax_state.B[:, :, 0, 2]  # Bz component
        agate_field = agate_fields['magnetic_field'][:, :, 0, 2]
    elif field_name == 'pressure':
        jax_field = jax_state.p[:, :, 0]
        agate_field = agate_fields['pressure'][:, :, 0]

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
    diff_max = max(abs(float(diff_field.min())), abs(float(diff_field.max())))
    im3 = axes[2].imshow(diff_field.T, origin='lower', cmap='RdBu_r',
                          vmin=-diff_max, vmax=diff_max)
    axes[2].set_title('Difference (JAX - AGATE)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    plt.colorbar(im3, ax=axes[2])

    # Add L2 error annotation
    l2_err = float(l2_error(jax_field, agate_field))
    fig.suptitle(f'{field_name.replace("_", " ").title()} Comparison '
                 f'(Resolution {resolution}) - L2 Error: {l2_err:.4g}')

    fig.tight_layout()
    return fig
```

#### 3.3 New Function: `compute_field_l2_errors`

```python
def compute_field_l2_errors(jax_state, agate_fields: dict, geometry) -> dict:
    """Compute L2 errors between JAX and AGATE spatial fields."""
    from jax_frc.validation.metrics import l2_error

    errors = {}

    # Density
    errors['density'] = float(l2_error(jax_state.n, agate_fields['density']))

    # Momentum components
    jax_mom = jax_state.n[..., None] * jax_state.v
    errors['momentum'] = float(l2_error(jax_mom, agate_fields['momentum']))

    # Magnetic field
    errors['magnetic_field'] = float(l2_error(jax_state.B, agate_fields['magnetic_field']))

    # Pressure
    errors['pressure'] = float(l2_error(jax_state.p, agate_fields['pressure']))

    return errors
```

#### 3.4 Modified Main Function Structure

```python
def main(quick_test: bool = False) -> bool:
    # ... existing setup code ...

    for resolution in resolutions:
        print(f"\nResolution {resolution}: ", end="")
        t_start = time.time()
        final_state, geometry, history = run_simulation(cfg)
        t_sim = time.time() - t_start
        print(f"[{t_sim:.2f}s]")

        # Load AGATE final state fields
        agate_fields = load_agate_fields("ot", resolution)

        # Compute field L2 errors
        field_errors = compute_field_l2_errors(final_state, agate_fields, geometry)
        print_field_l2_table(field_errors, L2_ERROR_TOL)

        # Compute scalar metrics
        jax_metrics = compute_metrics(final_state, geometry)
        agate_metrics = load_agate_metrics("ot", resolution)
        scalar_results = compare_scalar_metrics(jax_metrics, agate_metrics)
        print_scalar_metrics_table(scalar_results)

        # Summary
        total_checks = len(field_errors) + len(scalar_results)
        passed_checks = sum(1 for e in field_errors.values() if e <= L2_ERROR_TOL)
        passed_checks += sum(1 for m in scalar_results.values() if m['passed'])
        print(f"\n  Summary: {passed_checks}/{total_checks} PASS")

        # Store for report
        all_results[resolution] = {
            'field_errors': field_errors,
            'scalar_metrics': scalar_results,
            'jax_state': final_state,
            'agate_fields': agate_fields,
        }

    # Generate report with visualizations
    report = ValidationReport(...)

    for resolution, data in all_results.items():
        # Plot 1: Scalar metrics comparison
        fig_scalar = create_scalar_comparison_plot(data['scalar_metrics'], resolution)
        report.add_plot(fig_scalar, name=f"scalar_comparison_r{resolution}")

        # Plot 2: Error vs threshold summary
        fig_error = create_error_threshold_plot(
            data['field_errors'], data['scalar_metrics'],
            L2_ERROR_TOL, RELATIVE_ERROR_TOL
        )
        report.add_plot(fig_error, name=f"error_summary_r{resolution}")

        # Plot 3: Field comparison contours
        for field_name in ['density', 'magnetic_field']:
            fig_field = create_field_comparison_plot(
                data['jax_state'], data['agate_fields'],
                field_name, resolution
            )
            report.add_plot(fig_field, name=f"{field_name}_comparison_r{resolution}")

    report_dir = report.save()
```

### 4. Metrics Summary

| Metric Type | Metrics | Threshold |
|-------------|---------|-----------|
| **Field L2 Errors** | density, momentum, magnetic_field, pressure | 0.01 (1%) |
| **Scalar Metrics (Energy)** | total_energy, magnetic_energy, kinetic_energy | 0.05 (5%) |
| **Scalar Metrics (Other)** | enstrophy, max_current | 0.01 (1%) |

### 5. Files to Modify

| File | Changes |
|------|---------|
| `validation/utils/reporting.py` | Add `print_field_l2_table()`, `print_scalar_metrics_table()` |
| `validation/utils/plots.py` (new) | Add plot generation functions |
| `validation/cases/regression/orszag_tang.py` | Restructure main(), add field L2 computation |
| `validation/cases/regression/gem_reconnection.py` | Same changes as orszag_tang.py |

## Success Criteria

1. Console output matches the format shown in Section 1
2. HTML report includes all three visualization types
3. Both field L2 errors and scalar metrics are validated
4. Existing tests continue to pass
5. Report quality matches analytic validation cases

## Future Considerations

- Add convergence plots showing error vs resolution
- Support for time-series comparison if AGATE adds temporal data
- Automated regression testing in CI/CD pipeline
