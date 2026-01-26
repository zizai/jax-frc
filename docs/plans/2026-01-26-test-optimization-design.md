# Test Suite Optimization Design

## Goal

Reduce CI/CD test suite runtime from ~150s to ~60s (60% reduction) by tuning physics parameters in slow tests.

## Approach

Parameter tuning: reduce grid sizes, iteration counts, and increase resistivity values so physics validation completes faster while still catching regressions.

## Current Baseline

- **Total tests**: 198
- **Total runtime**: 150.91s (2m 31s)
- **Top 2 tests**: 98s (65% of runtime)

### Slowest Tests

| Test | Current Time | File |
|------|-------------|------|
| `test_decay_rate_scales_with_resistivity` | 57s | test_imex_validation.py |
| `test_1d_diffusion_decay_rate` | 41s | test_imex_validation.py |
| `test_uniform_field_is_stationary` | 12s | test_imex_validation.py |
| `test_imex_step_stable_with_large_dt` | 6s | test_linear_solvers.py |
| `test_diffusion_preserves_pattern_shape` | 6s | test_imex_validation.py |
| `test_projection_reduces_div_b` | 5s | test_operators_physics.py |

## Changes

### test_imex_validation.py

#### test_1d_diffusion_decay_rate

| Parameter | Before | After |
|-----------|--------|-------|
| Grid | 32x64 | 16x32 |
| Steps | 100 | 25 |
| dt | 2e-5 | 1e-4 |
| eta | 1e-4 | 5e-4 |

Rationale: Higher resistivity produces same measurable decay in fewer steps. Backward Euler is unconditionally stable so larger dt is fine.

#### test_decay_rate_scales_with_resistivity

| Parameter | Before | After |
|-----------|--------|-------|
| Grid | 16x32 | 8x16 |
| Steps | 50 (x2) | 15 (x2) |
| eta_low | 1e-5 | 1e-4 |
| eta_high | 1e-4 | 1e-3 |

Rationale: Test validates relative scaling, not absolute accuracy. Coarser grid and higher resistivity values still demonstrate the relationship.

#### test_uniform_field_is_stationary

| Parameter | Before | After |
|-----------|--------|-------|
| Grid | 16x32 | 8x16 |
| Steps | 20 | 10 |

Rationale: Verifying uniform fields don't diffuse requires minimal resolution.

#### test_diffusion_preserves_pattern_shape

| Parameter | Before | After |
|-----------|--------|-------|
| Grid | 16x32 | 8x16 |
| Steps | 10 | 5 |

Rationale: Pattern correlation check doesn't need fine resolution.

### test_linear_solvers.py

#### test_imex_step_stable_with_large_dt

Review and reduce grid/steps as appropriate.

### test_operators_physics.py

#### test_projection_reduces_div_b

Review and reduce grid size as appropriate.

## Expected Results

| Category | Before | After | Savings |
|----------|--------|-------|---------|
| Top 2 tests | 98s | ~20s | 78s |
| Secondary tests | ~30s | ~12s | 18s |
| **Total** | ~150s | ~55-60s | ~60% |

## Validation

1. Run full test suite before changes (baseline)
2. Apply changes
3. Run full test suite after changes
4. Verify all 198 tests still pass
5. Confirm timing improvement

## Non-Goals

- No new dependencies (pytest-xdist, etc.)
- No fixture restructuring
- No test reorganization
- No slow test markers (all tests run by default)
