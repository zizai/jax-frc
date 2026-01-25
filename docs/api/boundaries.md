# Boundary Conditions

Boundary condition classes for handling domain edges.

## Available Boundaries

| Boundary | Class | Description |
|----------|-------|-------------|
| Conducting Wall | `ConductingBoundary` | Perfect conductor (ψ = const) |
| Symmetry | `SymmetryBoundary` | Axisymmetric axis (r=0) |
| Time-Dependent | `TimeDependentMirrorBC` | Moving mirror for compression |

## Package Structure

```
jax_frc/boundaries/
├── base.py            # BoundaryCondition base class
├── conducting.py      # Conducting wall boundary
├── symmetry.py        # Symmetry axis boundary
└── time_dependent.py  # Time-dependent mirror BC for compression
```

## Usage

```python
from jax_frc.boundaries import ConductingBoundary, SymmetryBoundary

# Create boundary conditions
wall_bc = ConductingBoundary(psi_wall=0.0)
axis_bc = SymmetryBoundary()
```

## Time-Dependent Mirror Boundary

For compression simulations, mirrors can move according to a prescribed schedule:

```python
from jax_frc.boundaries import TimeDependentMirrorBC

mirror_bc = TimeDependentMirrorBC(
    initial_z=2.0,
    mirror_ratio=1.5,
    ramp_time=10.0
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `initial_z` | float | Initial mirror position |
| `mirror_ratio` | float | Final compression ratio |
| `ramp_time` | float | Time to reach full compression (in Alfven times) |
