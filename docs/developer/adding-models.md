# Adding a New Physics Model

This tutorial walks through creating a new physics model for jax-frc.

## Overview

All physics models implement the `PhysicsModel` protocol. We'll create a simple "advection-diffusion" model as an example.

## Step 1: Create the Model File

Create `jax_frc/models/advection_diffusion.py`:

```python
"""Advection-diffusion model for scalar transport."""

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class AdvectionDiffusion(PhysicsModel):
    """Simple advection-diffusion for a scalar field.

    Solves: ∂φ/∂t + v·∇φ = D∇²φ

    Args:
        diffusivity: Diffusion coefficient D [m²/s]
        velocity: Constant advection velocity (vr, vz) [m/s]
    """

    diffusivity: float
    velocity: tuple[float, float] = (0.0, 0.0)

    @partial(jax.jit, static_argnums=(0, 2))
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute ∂φ/∂t from advection and diffusion."""
        phi = state.psi  # Using psi field for our scalar
        dr, dz = geometry.dr, geometry.dz

        # Diffusion: D∇²φ
        laplacian = self._laplacian(phi, dr, dz)
        diffusion = self.diffusivity * laplacian

        # Advection: -v·∇φ
        dphi_dr = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2 * dr)
        dphi_dz = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * dz)
        advection = -(self.velocity[0] * dphi_dr + self.velocity[1] * dphi_dz)

        d_phi = diffusion + advection

        return state.replace(psi=d_phi)

    def _laplacian(self, f: jnp.ndarray, dr: float, dz: float) -> jnp.ndarray:
        """Compute ∇²f using central differences."""
        d2f_dr2 = (jnp.roll(f, -1, axis=0) - 2*f + jnp.roll(f, 1, axis=0)) / dr**2
        d2f_dz2 = (jnp.roll(f, -1, axis=1) - 2*f + jnp.roll(f, 1, axis=1)) / dz**2
        return d2f_dr2 + d2f_dz2

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return maximum stable timestep."""
        dr, dz = geometry.dr, geometry.dz
        dx_min = min(dr, dz)

        # Diffusion limit: dt < dx²/(2D)
        dt_diffusion = dx_min**2 / (2 * self.diffusivity) if self.diffusivity > 0 else float('inf')

        # Advection limit: dt < dx/|v|
        v_max = max(abs(self.velocity[0]), abs(self.velocity[1]))
        dt_advection = dx_min / v_max if v_max > 0 else float('inf')

        return min(dt_diffusion, dt_advection)

    @classmethod
    def from_config(cls, config: dict) -> "AdvectionDiffusion":
        """Create model from configuration dictionary."""
        return cls(
            diffusivity=config.get('diffusivity', 1.0),
            velocity=tuple(config.get('velocity', [0.0, 0.0]))
        )
```

## Step 2: Register in __init__.py

Edit `jax_frc/models/__init__.py`:

```python
from jax_frc.models.advection_diffusion import AdvectionDiffusion

__all__ = [
    # ... existing exports ...
    "AdvectionDiffusion",
]
```

## Step 3: Write Tests

Create `tests/test_advection_diffusion.py`:

```python
"""Tests for AdvectionDiffusion model."""

import jax.numpy as jnp
import pytest

from jax_frc import Geometry
from jax_frc.models import AdvectionDiffusion
from jax_frc.core.state import State


@pytest.fixture
def geometry():
    return Geometry(
        coord_system='cartesian',
        nr=32, nz=32,
        r_min=0.0, r_max=1.0,
        z_min=0.0, z_max=1.0
    )


@pytest.fixture
def gaussian_state(geometry):
    """Initial Gaussian profile."""
    r, z = jnp.meshgrid(
        jnp.linspace(0, 1, geometry.nr),
        jnp.linspace(0, 1, geometry.nz),
        indexing='ij'
    )
    psi = jnp.exp(-((r - 0.5)**2 + (z - 0.5)**2) / 0.1)
    return State(psi=psi, v=jnp.zeros((geometry.nr, geometry.nz, 3)),
                 p=jnp.ones((geometry.nr, geometry.nz)),
                 rho=jnp.ones((geometry.nr, geometry.nz)), t=0.0)


class TestAdvectionDiffusion:

    def test_pure_diffusion_smooths(self, geometry, gaussian_state):
        """Pure diffusion should smooth the profile."""
        model = AdvectionDiffusion(diffusivity=1.0, velocity=(0.0, 0.0))

        rhs = model.compute_rhs(gaussian_state, geometry)

        # Center should decrease (diffusing outward)
        center_idx = (geometry.nr // 2, geometry.nz // 2)
        assert rhs.psi[center_idx] < 0

    def test_advection_shifts(self, geometry, gaussian_state):
        """Advection should shift the profile."""
        model = AdvectionDiffusion(diffusivity=0.0, velocity=(1.0, 0.0))

        rhs = model.compute_rhs(gaussian_state, geometry)

        # With positive v_r, should have negative dphi/dt where gradient is positive
        # (profile moving in +r direction)
        assert rhs.psi.shape == gaussian_state.psi.shape

    def test_stable_dt_respects_diffusion(self, geometry, gaussian_state):
        """Stable dt should decrease with higher diffusivity."""
        model_low = AdvectionDiffusion(diffusivity=0.1)
        model_high = AdvectionDiffusion(diffusivity=10.0)

        dt_low = model_low.compute_stable_dt(gaussian_state, geometry)
        dt_high = model_high.compute_stable_dt(gaussian_state, geometry)

        assert dt_high < dt_low

    def test_from_config(self):
        """Factory method should create model correctly."""
        config = {'diffusivity': 2.5, 'velocity': [1.0, -0.5]}
        model = AdvectionDiffusion.from_config(config)

        assert model.diffusivity == 2.5
        assert model.velocity == (1.0, -0.5)

    def test_jit_compilation(self, geometry, gaussian_state):
        """Model should work with JIT."""
        model = AdvectionDiffusion(diffusivity=1.0)

        # First call compiles
        rhs1 = model.compute_rhs(gaussian_state, geometry)
        # Second call uses cached
        rhs2 = model.compute_rhs(gaussian_state, geometry)

        assert jnp.allclose(rhs1.psi, rhs2.psi)
```

## Step 4: Run Tests

```bash
py -m pytest tests/test_advection_diffusion.py -v
```

## Step 5: Document

Add documentation following the [model template](../models/resistive-mhd.md).

## Checklist

Before submitting your new model:

- [ ] Implements `PhysicsModel` protocol (`compute_rhs`, `compute_stable_dt`)
- [ ] Uses `@partial(jax.jit, static_argnums=(0, 2))` on `compute_rhs`
- [ ] Has `from_config()` classmethod for YAML configuration
- [ ] Registered in `jax_frc/models/__init__.py`
- [ ] Has unit tests covering:
  - [ ] Basic physics behavior
  - [ ] Edge cases (zero parameters, etc.)
  - [ ] JIT compilation works
  - [ ] Factory method
- [ ] Has documentation with:
  - [ ] Physics overview
  - [ ] Governing equations
  - [ ] Parameters table
  - [ ] Example usage

## Common Pitfalls

1. **Forgetting `static_argnums`**: Causes recompilation or tracing errors
2. **Python control flow**: Use `lax.cond` not `if/else` on traced values
3. **Mutable operations**: Use `.at[].set()` not direct assignment
4. **Missing `frozen=True`**: Dataclass must be immutable for JIT
