# Neutral-IMEX Integration Design (Composition-Based)

**Date**: 2026-01-26
**Status**: Draft
**Goal**: Unify neutral coupling and IMEX solver via composition-based architecture

## Overview

This design integrates two existing approved plans:
- `2026-01-26-imex-solver-design.md` - implicit diffusion for stiff resistive terms
- `2026-01-26-neutral-coupling-design.md` - Lamy Ridge-style neutral fluid + atomic physics

The key addition is a **composition-based architecture** using protocols and wrapper classes, allowing existing models to remain unchanged while gaining new capabilities.

## Architecture

```
jax_frc/models/
├── protocols.py         (NEW: SplitRHS, SourceTerms protocols)
├── coupled.py           (NEW: CoupledState, CoupledModel)
├── atomic_coupling.py   (NEW: wraps atomic_rates into SourceTerms)
├── resistive_mhd.py     (ADD: explicit_rhs, implicit_rhs methods)
├── extended_mhd.py      (ADD: explicit_rhs, implicit_rhs methods)
├── neutral_fluid.py     (existing, add SplitRHS if needed)
└── atomic_rates.py      (existing, complete)

jax_frc/solvers/
├── imex.py              (UPDATE: use SplitRHS protocol)
└── linear/
    ├── cg.py            (existing, complete)
    └── preconditioners.py (existing, complete)
```

## Component 1: Protocols

**File**: `jax_frc/models/protocols.py`

```python
from typing import Protocol, Tuple
from jax import Array
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

class SplitRHS(Protocol):
    """Protocol for models supporting IMEX time integration."""

    def explicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Terms safe for explicit integration (advection, ideal MHD).

        These terms have CFL constraints based on wave speeds, not diffusion.
        """
        ...

    def implicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Stiff terms needing implicit treatment (diffusion, resistivity).

        These terms would require prohibitively small timesteps if explicit.
        """
        ...

    def apply_implicit_operator(
        self, state: State, geometry: Geometry, dt: float, theta: float
    ) -> State:
        """Apply (I - theta*dt*L) for implicit solve.

        Used by CG solver in matrix-free form.
        """
        ...


class SourceTerms(Protocol):
    """Protocol for atomic/collision source terms coupling two fluids."""

    def compute_sources(
        self, plasma_state: State, neutral_state, geometry: Geometry
    ) -> Tuple:
        """Compute source terms for both plasma and neutral fluids.

        Returns:
            (plasma_sources, neutral_sources) where each contains
            mass, momentum, energy transfer rates.
        """
        ...
```

## Component 2: CoupledState

**File**: `jax_frc/models/coupled.py`

```python
from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState

@dataclass
class CoupledState:
    """Combined plasma + neutral state for coupled simulations."""
    plasma: State
    neutral: NeutralState

# Register as JAX pytree
from jax import tree_util
tree_util.register_pytree_node(
    CoupledState,
    lambda s: ((s.plasma, s.neutral), None),
    lambda _, children: CoupledState(children[0], children[1])
)


@dataclass
class SourceRates:
    """Source term rates for one fluid species."""
    mass: Array      # kg/m^3/s
    momentum: Array  # N/m^3 (vector)
    energy: Array    # W/m^3
```

## Component 3: CoupledModel

**File**: `jax_frc/models/coupled.py` (continued)

```python
from jax_frc.models.base import PhysicsModel
from jax_frc.models.protocols import SplitRHS, SourceTerms

@dataclass
class CoupledModelConfig:
    """Configuration for coupled plasma-neutral model."""
    source_subcycles: int = 10      # Subcycles for stiff atomic sources
    use_analytic_source: bool = False  # Use analytic integration for linear terms


class CoupledModel(PhysicsModel):
    """Composes plasma model + neutral model + atomic coupling.

    Implements SplitRHS protocol for use with ImexSolver.
    Handles operator splitting for stiff atomic sources.
    """

    def __init__(
        self,
        plasma_model: PhysicsModel,  # ResistiveMHD or ExtendedMHD
        neutral_model,               # NeutralFluid
        atomic_coupling: SourceTerms,
        config: CoupledModelConfig
    ):
        self.plasma = plasma_model
        self.neutral = neutral_model
        self.coupling = atomic_coupling
        self.config = config

    def explicit_rhs(
        self, state: CoupledState, geometry: Geometry, t: float
    ) -> CoupledState:
        """Explicit terms: advection for both fluids."""
        # Plasma advection (ideal MHD terms)
        d_plasma = self.plasma.explicit_rhs(state.plasma, geometry, t)

        # Neutral advection (HLLE flux divergence)
        d_neutral = self.neutral.compute_flux_rhs(state.neutral, geometry)

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def implicit_rhs(
        self, state: CoupledState, geometry: Geometry, t: float
    ) -> CoupledState:
        """Implicit terms: resistive diffusion, thermal conduction."""
        d_plasma = self.plasma.implicit_rhs(state.plasma, geometry, t)

        # Neutrals typically don't need implicit treatment
        d_neutral = NeutralState.zeros_like(state.neutral)

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def source_rhs(
        self, state: CoupledState, geometry: Geometry, t: float
    ) -> CoupledState:
        """Atomic source terms coupling plasma <-> neutrals."""
        plasma_src, neutral_src = self.coupling.compute_sources(
            state.plasma, state.neutral, geometry
        )

        d_plasma = State(
            rho=plasma_src.mass,
            momentum=plasma_src.momentum,
            energy=plasma_src.energy,
            psi=jnp.zeros_like(state.plasma.psi),
            B=jnp.zeros_like(state.plasma.B) if hasattr(state.plasma, 'B') else None
        )

        d_neutral = NeutralState(
            rho_n=neutral_src.mass,
            mom_n=neutral_src.momentum,
            E_n=neutral_src.energy
        )

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def apply_implicit_operator(
        self, state: CoupledState, geometry: Geometry, dt: float, theta: float
    ) -> CoupledState:
        """Apply (I - theta*dt*L) for implicit diffusion solve."""
        new_plasma = self.plasma.apply_implicit_operator(
            state.plasma, geometry, dt, theta
        )
        return CoupledState(plasma=new_plasma, neutral=state.neutral)
```

## Component 4: AtomicCoupling

**File**: `jax_frc/models/atomic_coupling.py`

```python
from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit

from jax_frc.constants import MI, QE
from jax_frc.models.atomic_rates import (
    ionization_rate, recombination_rate,
    charge_exchange_rates, bremsstrahlung, line_radiation,
    ionization_energy_loss
)
from jax_frc.models.coupled import SourceRates

@dataclass
class AtomicCouplingConfig:
    """Configuration for atomic physics coupling."""
    include_radiation: bool = True
    impurity_fraction: float = 0.0  # n_imp / n_e
    Z_eff: float = 1.0


class AtomicCoupling:
    """Wraps atomic_rates module into SourceTerms protocol."""

    def __init__(self, config: AtomicCouplingConfig):
        self.config = config

    @jit
    def compute_sources(
        self, plasma, neutral, geometry
    ):
        """Compute bidirectional source terms from atomic processes.

        Returns:
            (plasma_sources, neutral_sources) as SourceRates
        """
        # Extract quantities
        n_e = plasma.rho / MI
        n_i = n_e  # Quasi-neutrality
        n_n = neutral.rho_n / MI
        T_e = self._compute_temperature(plasma)

        v_i = plasma.momentum / jnp.maximum(plasma.rho[..., None], 1e-20)
        v_n = neutral.v_n

        # Rate coefficients (from atomic_rates.py)
        S_ion = ionization_rate(T_e, n_e, neutral.rho_n)
        S_rec = recombination_rate(T_e, n_e, n_i)
        R_cx, Q_cx = charge_exchange_rates(T_e, n_i, n_n, v_i, v_n)

        # Energy sinks
        P_ion = ionization_energy_loss(S_ion)
        P_rad = jnp.where(
            self.config.include_radiation,
            bremsstrahlung(n_e, T_e, self.config.Z_eff) +
            line_radiation(n_e, T_e, self.config.impurity_fraction * n_e),
            0.0
        )

        # Plasma gains from ionization, loses to recombination
        plasma_sources = SourceRates(
            mass=S_ion - S_rec,
            momentum=R_cx,
            energy=-P_ion - P_rad + Q_cx
        )

        # Neutrals: opposite signs (conservation)
        neutral_sources = SourceRates(
            mass=-S_ion + S_rec,
            momentum=-R_cx,
            energy=-Q_cx
        )

        return plasma_sources, neutral_sources

    def _compute_temperature(self, plasma):
        """Compute electron temperature from plasma state."""
        # Single-temperature model: T = p / (2n)
        n_e = plasma.rho / MI
        p = plasma.pressure  # Assumes State has pressure property
        return p / jnp.maximum(2 * n_e, 1e-10)
```

## Component 5: IMEX Solver Updates

**File**: `jax_frc/solvers/imex.py` (modifications)

```python
class ImexSolver(Solver):
    """IMEX time stepper using SplitRHS protocol."""

    def step(
        self, model, state, geometry, dt, t: float = 0.0
    ):
        """Strang splitting: explicit-implicit-explicit.

        For CoupledModel, also handles source term splitting.
        """
        # Check if model has source terms (CoupledModel)
        has_sources = hasattr(model, 'source_rhs')

        if has_sources:
            # Full Strang: sources(dt/2) -> transport(dt) -> sources(dt/2)
            state = self._source_step(model, state, geometry, dt/2, t)

        # Transport step with IMEX
        # Half explicit
        k_exp = model.explicit_rhs(state, geometry, t)
        state = self._add_scaled(state, k_exp, dt/2)

        # Full implicit (CG solve)
        state = self._implicit_solve(model, state, geometry, dt)

        # Half explicit
        k_exp = model.explicit_rhs(state, geometry, t + dt)
        state = self._add_scaled(state, k_exp, dt/2)

        if has_sources:
            state = self._source_step(model, state, geometry, dt/2, t + dt)

        return state

    def _source_step(self, model, state, geometry, dt, t):
        """Subcycled integration of stiff atomic sources."""
        n_sub = model.config.source_subcycles
        dt_sub = dt / n_sub

        def body(i, s):
            d_state = model.source_rhs(s, geometry, t + i * dt_sub)
            return self._add_scaled(s, d_state, dt_sub)

        return lax.fori_loop(0, n_sub, body, state)
```

## Component 6: ResistiveMHD SplitRHS Methods

**File**: `jax_frc/models/resistive_mhd.py` (additions)

```python
class ResistiveMHD(PhysicsModel):
    # ... existing code ...

    def explicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Advection term only: -v . grad(psi)"""
        v = state.momentum / jnp.maximum(state.rho[..., None], 1e-20)
        grad_psi = gradient(state.psi, geometry.dr, geometry.dz)
        advection = -jnp.sum(v * grad_psi, axis=-1)

        return State(
            psi=advection,
            rho=jnp.zeros_like(state.rho),
            momentum=jnp.zeros_like(state.momentum),
            energy=jnp.zeros_like(state.energy)
        )

    def implicit_rhs(self, state: State, geometry: Geometry, t: float) -> State:
        """Diffusion term only: (eta/mu0) * Laplacian_star(psi)"""
        eta = self.resistivity.compute(state, geometry)
        diffusion = (eta / MU0) * laplace_star(state.psi, geometry)

        return State(
            psi=diffusion,
            rho=jnp.zeros_like(state.rho),
            momentum=jnp.zeros_like(state.momentum),
            energy=jnp.zeros_like(state.energy)
        )

    def apply_implicit_operator(
        self, state: State, geometry: Geometry, dt: float, theta: float
    ) -> State:
        """Apply (I - theta*dt*L) where L is diffusion operator."""
        eta = self.resistivity.compute(state, geometry)
        L_psi = (eta / MU0) * laplace_star(state.psi, geometry)

        new_psi = state.psi - theta * dt * L_psi
        return State(psi=new_psi, rho=state.rho,
                     momentum=state.momentum, energy=state.energy)
```

## Test Problems

### Test 1: Diffusion-only (IMEX validation)

**File**: `tests/test_imex_diffusion.py`

```python
def test_gaussian_diffusion():
    """Verify implicit diffusion against analytic Gaussian solution.

    Setup: Gaussian temperature profile T(r,0) = T0 * exp(-r^2/sigma^2)
    Analytic: T(r,t) = T0/(1 + 4*kappa*t/sigma^2) * exp(-r^2/(sigma^2 + 4*kappa*t))

    Tests:
    - L2 error < 1% for 2nd-order scheme
    - Convergence rate ~2 when halving grid
    """
```

### Test 2: Ionization front (Neutral coupling)

**File**: `tests/test_ionization_front.py`

```python
def test_ionization_front_velocity():
    """Verify ionization wave propagation into neutral gas.

    Setup: 1D slab, hot plasma (left), cold neutrals (right)
    Expected: Front velocity ~ sqrt(k_ion * n_e * E_ion / (rho_n * c_v))

    Tests:
    - Front position matches expected velocity
    - Total mass conserved (plasma + neutral)
    """
```

### Test 3: Charge exchange equilibration (Momentum coupling)

**File**: `tests/test_cx_equilibration.py`

```python
def test_cx_velocity_relaxation():
    """Verify CX friction equilibrates plasma-neutral velocities.

    Setup: Stationary plasma, drifting neutrals
    Analytic: Delta_v(t) = Delta_v0 * exp(-t/tau_cx)

    Tests:
    - Exponential relaxation rate matches tau_cx
    - Total momentum conserved
    """
```

## Implementation Phases

### Phase 1: Foundation
1. Create `jax_frc/models/protocols.py` with `SplitRHS`, `SourceTerms`
2. Create `jax_frc/models/coupled.py` with `CoupledState`, `SourceRates`
3. Add `explicit_rhs()`, `implicit_rhs()`, `apply_implicit_operator()` to `ResistiveMHD`

### Phase 2: IMEX Wiring
1. Update `ImexSolver.step()` to use `SplitRHS` protocol
2. Update `ImexSolver._implicit_solve()` to use `apply_implicit_operator()`
3. Write `tests/test_imex_diffusion.py` with Gaussian analytic test

### Phase 3: Atomic Coupling
1. Create `jax_frc/models/atomic_coupling.py` implementing `SourceTerms`
2. Wire up all atomic rates from existing `atomic_rates.py`
3. Write `tests/test_cx_equilibration.py`

### Phase 4: Coupled Model
1. Implement `CoupledModel` class with Strang splitting
2. Add source subcycling in `ImexSolver`
3. Write `tests/test_ionization_front.py`

### Phase 5: Integration
1. End-to-end test: diffusion + ionization combined
2. Verify conservation laws hold for full coupled system
3. Add `SplitRHS` methods to `ExtendedMHD` if needed

## Design Decisions

1. **Composition over inheritance**: Existing models unchanged, new capabilities added via wrappers
2. **Protocols for flexibility**: Any model implementing `SplitRHS` works with `ImexSolver`
3. **Source term separation**: Atomic physics decoupled from transport for modularity
4. **Subcycling for stiffness**: Atomic sources can be very stiff; subcycling is simpler than implicit integration
5. **Simple test problems first**: Validate individual pieces before complex FRC scenarios

## References

- `2026-01-26-imex-solver-design.md` - Detailed CG solver and splitting design
- `2026-01-26-neutral-coupling-design.md` - Detailed atomic rates and neutral fluid design
- `plasma_physics.md` - Physics equations and model comparisons
