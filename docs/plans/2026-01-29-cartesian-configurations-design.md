# Cartesian Configurations Design

## Summary

Deprecate `validation_benchmarks.py` and create new configuration classes with proper Cartesian coordinate naming for validation cases.

## Problem

The existing `CylindricalShockConfiguration`, `CylindricalVortexConfiguration`, and `CylindricalGEMConfiguration` classes use misleading "cylindrical" naming and legacy parameter names (`nr`, `r_min`, `r_max`) despite the codebase being fully Cartesian.

## Solution

Create three new configuration files with physics-based names and proper Cartesian parameter naming.

## New Files

1. `jax_frc/configurations/orszag_tang.py` - OrszagTangConfiguration
2. `jax_frc/configurations/gem_reconnection.py` - GEMReconnectionConfiguration
3. `jax_frc/configurations/brio_wu.py` - BrioWuConfiguration

## Key Changes

- `nr` → `nx`, `r_min` → `x_min`, `r_max` → `x_max`
- `Br_L`/`Br_R` → `Bx_L`/`Bx_R` (Cartesian field naming)
- Pseudo-2D defaults (ny=1 or nx=1 for thin dimensions)
- Physics parameters exposed in configuration (e.g., `eta`)

## Files to Modify

- `jax_frc/configurations/__init__.py` - Update registry
- `validation/cases/mhd_regression/orszag_tang.py` - Update import
- `validation/cases/hall_reconnection/reconnection_gem.py` - Update import

## Files to Delete

- `jax_frc/configurations/validation_benchmarks.py`
