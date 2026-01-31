# Validation Reporting Alignment (Brio-Wu + GEM)

## Context
We want minimal alignment of validation console + HTML reporting so Brio-Wu
and GEM show the same types of metrics and plots as Orszag-Tang. No shared
reporting refactors; only add missing elements and AGATE reference generation
for Brio-Wu. The Brio-Wu case is renamed from `brio_wu_shock` to `brio_wu`.

## Goals
- Add AGATE reference generation for Brio-Wu and use it in validation.
- Make Brio-Wu and GEM produce the same report sections/plot types as OT.
- Keep console output consistent with OT/GEM headings and tables.

## Non-goals
- No redesign of `validation/utils/reporting.py`.
- No large refactors or new shared helper layers.
- No full test suite requirement (quick validation is sufficient).

## Design

### AGATE runner support (Brio-Wu)
- Add `brio_wu` to `validation/utils/agate_runner.py` using
  `BrioWu()` from `agate.framework.scenario`, `cfl=0.4`,
  `slopeName="mcbeta"`, `mcbeta=1.3`, `ncells=resolution`,
  `t_end=0.2`, `dt_out=0.04` (6 snapshots).
- Save outputs under `validation/references/agate/brio_wu/<res>/`.
- Extend `validation/utils/agate_data.py` to recognize `brio`/`brio_wu`
  and map to `brio_wu`.

### Case rename
- Rename `validation/cases/regression/brio_wu_shock.py` to `validation/cases/regression/brio_wu.py`.
- Update case name constants, report name, and any case registry/imports.

### Brio-Wu metrics/plots
- Add AGATE load helpers mirroring OT/GEM:
  `load_agate_config`, `load_agate_snapshot`, and 1-D field extraction
  (strip ghost cells, take shock axis slice).
- Compute per-field L2 errors vs AGATE for density, pressure, velocity,
  and magnetic components available in the AGATE output.
- Add aggregate time-series metrics using OT’s `compute_metrics`.
- Console tables: `print_field_l2_table`, `print_aggregate_metrics_table`,
  `print_scalar_metrics_table`.
- HTML plots:
  - `create_field_error_evolution_plot`
  - `create_timeseries_comparison_plot` (kinetic/magnetic fraction)
  - `create_scalar_comparison_plot` (final metrics JAX vs AGATE)
  - `create_error_threshold_plot`
  - Keep existing density/B field profile plots

### GEM reporting alignment
- Add scalar comparison bar plot and error-threshold plot to HTML.
- Print scalar metrics table in console (matching OT).

## Verification
- `py -m scripts/run_validation.py --case brio_wu --quick`
- `py -m scripts/run_validation.py --case reconnection_gem --quick`
