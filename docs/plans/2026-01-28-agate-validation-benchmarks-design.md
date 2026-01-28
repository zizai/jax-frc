# AGATE Validation Benchmarks Design

## Goal
- Add two validation cases aligned with AGATE benchmarks:
  - Orszag-Tang vortex (MHD regression).
  - GEM reconnection (Hall reconnection).
- Auto-download and cache AGATE reference data from Zenodo record 15084058.
- Compare JAX-FRC time-series metrics against AGATE via a paired, block-bootstrap
  mean-relative-error check with 95% confidence intervals.

## Scope
- New validation cases:
  - `validation/cases/mhd_regression/orszag_tang.py`
  - `validation/cases/hall_reconnection/reconnection_gem.py`
- New AGATE loader utility for download + parsing.
- Metrics and regression utilities for time-series comparison.
- Default resolutions: 256 / 512 / 1024.
- Configs live in the validation routines (no new YAML).

## Data Source and Layout
- Source: Zenodo record `https://zenodo.org/records/15084058`
- Cache: `validation/references/agate/`
  - `gem/{resolution}/...`
  - `ot/{resolution}/...`
- Loader will query Zenodo JSON, download needed files, and unpack archives
  into the cache. Downloads are automatic if missing.

## Architecture and Data Flow
1. Case calls AGATE loader to ensure reference files are present.
2. JAX-FRC simulation runs at 256/512/1024.
3. Loader reads AGATE snapshots or reference time-series.
4. Metrics extracted from both JAX-FRC and AGATE outputs.
5. Regression check computes mean relative error with block bootstrap
   (95% CI) for each metric.
6. Validation report includes time-series plots and CI-based pass/fail.

## Metrics
For both cases:
- Total energy
- Magnetic energy
- Kinetic energy
- Enstrophy
- Max |J|

Optional for GEM if available in AGATE:
- Reconnection rate time series.

## Regression Analysis
- Paired, block-bootstrap mean relative error.
- CI evaluated at 95% confidence.
- Each case defines per-metric thresholds; pass if upper CI bound <= threshold.
- Time grids are aligned via interpolation if needed.

## Error Handling
- Missing download / parse errors are reported as failed metrics with
  actionable hints (how to pre-seed cache).
- Quick mode validates loader + parsing, uses reduced resolution and
  reduced bootstrap.

## Testing Plan
- Add unit tests for regression utility (block bootstrap CI shape and
  monotonicity under known perturbations).
- Smoke-run each case with `--quick`.

## Open Items
- Confirm AGATE file naming patterns once the Zenodo record is parsed.
- Choose default CI thresholds for each metric.
