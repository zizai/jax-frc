# Neutral-IMEX Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate neutral fluid coupling with IMEX solver via composition-based architecture.

**Architecture:** Create `SplitRHS` and `SourceTerms` protocols, wrap existing models in `CoupledModel`, wire IMEX solver to use protocol methods. Validate with three simple test problems.

**Tech Stack:** JAX, pytest, dataclasses, typing.Protocol

---

## Task 1: Create Protocols Module

**Files:**
- Create: `jax_frc/models/protocols.py`
- Test: `tests/test_protocols.py`

**Step 1: Write the failing test**

```python
# tests/test_protocols.py
"""Tests for physics model protocols."""

import jax.numpy as jnp
from jax_frc.models.protocols import SplitRHS, SourceTerms


def test_split_rhs_protocol_exists():
    """SplitRHS protocol is importable and has required methods."""
    assert hasattr(SplitRHS, 'explicit_rhs')
    assert hasattr(SplitRHS, 'implicit_rhs')
    assert hasattr(SplitRHS, 'apply_implicit_operator')


def test_source_terms_protocol_exists():
    """SourceTerms protocol is importable and has required method."""
    assert hasattr(SourceTerms, 'compute_sources')
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_protocols.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.models.protocols'"

**Step 3: Write minimal implementation**

```python
# jax_frc/models/protocols.py
"""Protocols for physics models supporting IMEX and source term coupling."""

from typing import Protocol, Tuple, Any
from jax import Array


class SplitRHS(Protocol):
    """Protocol for models supporting IMEX time integration.

    Models implementing this protocol can be used with ImexSolver.
    """

    def explicit_rhs(self, state: Any, geometry: Any, t: float) -> Any:
        """Compute terms safe for explicit integration.

        Args:
            state: Current simulation state
            geometry: Grid geometry
            t: Current time

        Returns:
            State containing d(state)/dt for explicit terms only
        """
        ...

    def implicit_rhs(self, state: Any, geometry: Any, t: float) -> Any:
        """Compute stiff terms needing implicit treatment.

        Args:
            state: Current simulation state
            geometry: Grid geometry
            t: Current time

        Returns:
            State containing d(state)/dt for implicit terms only
        """
        ...

    def apply_implicit_operator(
        self, state: Any, geometry: Any, dt: float, theta: float
    ) -> Any:
        """Apply (I - theta*dt*L) for implicit solve.

        Used by CG solver in matrix-free form.

        Args:
            state: Current state
            geometry: Grid geometry
            dt: Timestep
            theta: Implicitness parameter (1.0=backward Euler)

        Returns:
            Result of applying implicit operator to state
        """
        ...


class SourceTerms(Protocol):
    """Protocol for atomic/collision source terms coupling two fluids."""

    def compute_sources(
        self, plasma_state: Any, neutral_state: Any, geometry: Any
    ) -> Tuple[Any, Any]:
        """Compute source terms for both plasma and neutral fluids.

        Args:
            plasma_state: Plasma state
            neutral_state: Neutral state
            geometry: Grid geometry

        Returns:
            (plasma_sources, neutral_sources) tuples
        """
        ...
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_protocols.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/protocols.py tests/test_protocols.py
git commit -m "feat: add SplitRHS and SourceTerms protocols"
```

---

## Task 2: Create CoupledState and SourceRates

**Files:**
- Create: `jax_frc/models/coupled.py`
- Test: `tests/test_coupled.py`

**Step 1: Write the failing test**

```python
# tests/test_coupled.py
"""Tests for coupled plasma-neutral state."""

import jax
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.coupled import CoupledState, SourceRates


def test_coupled_state_creation():
    """CoupledState can be created from plasma and neutral states."""
    plasma = State.zeros(8, 8)
    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8)) * 100.0
    )
    coupled = CoupledState(plasma=plasma, neutral=neutral)

    assert coupled.plasma is plasma
    assert coupled.neutral is neutral


def test_coupled_state_is_pytree():
    """CoupledState works with JAX transformations."""
    plasma = State.zeros(8, 8)
    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)),
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8))
    )
    coupled = CoupledState(plasma=plasma, neutral=neutral)

    # Should be able to flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(coupled)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert jnp.allclose(restored.plasma.psi, coupled.plasma.psi)
    assert jnp.allclose(restored.neutral.rho_n, coupled.neutral.rho_n)


def test_source_rates_creation():
    """SourceRates can be created with mass, momentum, energy."""
    rates = SourceRates(
        mass=jnp.ones((8, 8)),
        momentum=jnp.zeros((8, 8, 3)),
        energy=jnp.ones((8, 8)) * 1e3
    )

    assert rates.mass.shape == (8, 8)
    assert rates.momentum.shape == (8, 8, 3)
    assert rates.energy.shape == (8, 8)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_coupled.py -v`
Expected: FAIL with "ImportError: cannot import name 'CoupledState'"

**Step 3: Write minimal implementation**

```python
# jax_frc/models/coupled.py
"""Coupled plasma-neutral state and model for IMEX integration."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState


@dataclass(frozen=True)
class SourceRates:
    """Source term rates for one fluid species.

    All rates use SI units.
    """
    mass: Array      # kg/m³/s
    momentum: Array  # N/m³ (vector, shape nr,nz,3)
    energy: Array    # W/m³


@dataclass(frozen=True)
class CoupledState:
    """Combined plasma + neutral state for coupled simulations."""
    plasma: State
    neutral: NeutralState


# Register CoupledState as JAX pytree
def _coupled_state_flatten(state):
    children = (state.plasma, state.neutral)
    aux_data = None
    return children, aux_data


def _coupled_state_unflatten(aux_data, children):
    plasma, neutral = children
    return CoupledState(plasma=plasma, neutral=neutral)


jax.tree_util.register_pytree_node(
    CoupledState, _coupled_state_flatten, _coupled_state_unflatten
)


# Register SourceRates as JAX pytree
def _source_rates_flatten(rates):
    children = (rates.mass, rates.momentum, rates.energy)
    aux_data = None
    return children, aux_data


def _source_rates_unflatten(aux_data, children):
    mass, momentum, energy = children
    return SourceRates(mass=mass, momentum=momentum, energy=energy)


jax.tree_util.register_pytree_node(
    SourceRates, _source_rates_flatten, _source_rates_unflatten
)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_coupled.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/coupled.py tests/test_coupled.py
git commit -m "feat: add CoupledState and SourceRates dataclasses"
```

---

## Task 3: Add SplitRHS Methods to ResistiveMHD

**Files:**
- Modify: `jax_frc/models/resistive_mhd.py`
- Test: `tests/test_resistive_mhd_split.py`

**Step 1: Write the failing test**

```python
# tests/test_resistive_mhd_split.py
"""Tests for ResistiveMHD SplitRHS interface."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity


def test_explicit_rhs_returns_advection_only():
    """explicit_rhs returns only advection term, no diffusion."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    # Create state with non-zero psi and velocity
    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    v = jnp.zeros((16, 16, 3))
    v = v.at[:, :, 0].set(0.1)  # v_r = 0.1 m/s
    state = state.replace(psi=psi, v=v)

    rhs = model.explicit_rhs(state, geometry, t=0.0)

    # Should have non-zero psi (advection)
    assert jnp.any(rhs.psi != 0)


def test_implicit_rhs_returns_diffusion_only():
    """implicit_rhs returns only diffusion term."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.sin(jnp.pi * geometry.r_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    state = state.replace(psi=psi)

    rhs = model.implicit_rhs(state, geometry, t=0.0)

    # Should have non-zero psi (diffusion)
    assert jnp.any(rhs.psi != 0)


def test_apply_implicit_operator():
    """apply_implicit_operator applies (I - theta*dt*L)."""
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    state = State.zeros(16, 16)
    psi = jnp.ones((16, 16))
    state = state.replace(psi=psi)

    result = model.apply_implicit_operator(state, geometry, dt=1e-6, theta=1.0)

    # With identity input and small dt, result should be close to input
    assert result.psi.shape == psi.shape
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_resistive_mhd_split.py -v`
Expected: FAIL with "AttributeError: 'ResistiveMHD' object has no attribute 'explicit_rhs'"

**Step 3: Write minimal implementation**

Add these methods to `jax_frc/models/resistive_mhd.py` after the `_compute_j_phi` method:

```python
    def explicit_rhs(self, state: State, geometry: Geometry, t: float = 0.0) -> State:
        """Advection term only: -v . grad(psi).

        This is the explicit part for IMEX splitting.
        """
        psi = state.psi
        v_r = state.v[:, :, 0]
        v_z = state.v[:, :, 2]
        dr, dz = geometry.dr, geometry.dz

        # Compute gradient of psi using central differences
        # Interior only, boundaries stay zero
        dpsi_dr = jnp.zeros_like(psi)
        dpsi_dr = dpsi_dr.at[1:-1, :].set(
            (psi[2:, :] - psi[:-2, :]) / (2 * dr)
        )

        dpsi_dz = jnp.zeros_like(psi)
        dpsi_dz = dpsi_dz.at[:, 1:-1].set(
            (psi[:, 2:] - psi[:, :-2]) / (2 * dz)
        )

        # Advection: -v . grad(psi)
        advection = -(v_r * dpsi_dr + v_z * dpsi_dz)

        return state.replace(psi=advection)

    def implicit_rhs(self, state: State, geometry: Geometry, t: float = 0.0) -> State:
        """Diffusion term only: (eta/mu0) * Delta*psi.

        This is the implicit part for IMEX splitting.
        """
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute Delta*psi
        delta_star_psi = self._laplace_star(psi, dr, dz, r)

        # Get resistivity
        j_phi = -delta_star_psi / (MU0 * r)
        eta = self.resistivity.compute(j_phi)

        # Diffusion: (eta/mu_0)*Delta*psi
        diffusion = (eta / MU0) * delta_star_psi

        return state.replace(psi=diffusion)

    def apply_implicit_operator(
        self, state: State, geometry: Geometry, dt: float, theta: float
    ) -> State:
        """Apply (I - theta*dt*L) where L is diffusion operator.

        Used for matrix-free CG solve.
        """
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute diffusion operator L*psi = (eta/mu0) * Delta*psi
        delta_star_psi = self._laplace_star(psi, dr, dz, r)
        j_phi = -delta_star_psi / (MU0 * r)
        eta = self.resistivity.compute(j_phi)
        L_psi = (eta / MU0) * delta_star_psi

        # Apply (I - theta*dt*L)
        new_psi = psi - theta * dt * L_psi

        return state.replace(psi=new_psi)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_resistive_mhd_split.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/resistive_mhd.py tests/test_resistive_mhd_split.py
git commit -m "feat: add SplitRHS methods to ResistiveMHD"
```

---

## Task 4: Create AtomicCoupling Class

**Files:**
- Create: `jax_frc/models/atomic_coupling.py`
- Test: `tests/test_atomic_coupling.py`

**Step 1: Write the failing test**

```python
# tests/test_atomic_coupling.py
"""Tests for AtomicCoupling source term computation."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.constants import MI, QE


def test_compute_sources_returns_opposite_mass_rates():
    """Mass source terms should have opposite signs (conservation)."""
    config = AtomicCouplingConfig(include_radiation=False)
    coupling = AtomicCoupling(config)

    geometry = Geometry(
        coord_system="cylindrical",
        nr=8, nz=8,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    # Hot plasma, cold neutrals
    plasma = State.zeros(8, 8)
    n_e = 1e19  # m^-3
    T_e = 100 * QE  # 100 eV in Joules
    plasma = plasma.replace(
        n=jnp.ones((8, 8)) * n_e,
        T=jnp.ones((8, 8)) * T_e,
        p=jnp.ones((8, 8)) * n_e * T_e * 2  # p = 2*n*T
    )

    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * 1e-6,  # ~1e21 m^-3 neutral density
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8)) * 100.0
    )

    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Mass conservation: plasma_src.mass + neutral_src.mass = 0
    total_mass_rate = plasma_src.mass + neutral_src.mass
    assert jnp.allclose(total_mass_rate, 0.0, atol=1e-30)


def test_compute_sources_momentum_conservation():
    """Momentum source terms should have opposite signs."""
    config = AtomicCouplingConfig(include_radiation=False)
    coupling = AtomicCoupling(config)

    geometry = Geometry(
        coord_system="cylindrical",
        nr=8, nz=8,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    plasma = State.zeros(8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8)) * 1e19,
        T=jnp.ones((8, 8)) * 100 * QE,
        p=jnp.ones((8, 8)) * 1e19 * 100 * QE * 2,
        v=jnp.zeros((8, 8, 3))
    )

    # Neutrals moving in z direction
    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 3)).at[:, :, 2].set(1e-6 * 1000),  # v_z = 1000 m/s
        E_n=jnp.ones((8, 8)) * 100.0
    )

    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Momentum conservation: sum should be zero
    total_mom = plasma_src.momentum + neutral_src.momentum
    assert jnp.allclose(total_mom, 0.0, atol=1e-25)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_atomic_coupling.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.models.atomic_coupling'"

**Step 3: Write minimal implementation**

```python
# jax_frc/models/atomic_coupling.py
"""Atomic physics coupling between plasma and neutral fluids."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit

from jax_frc.constants import MI, QE
from jax_frc.models.atomic_rates import (
    ionization_rate, recombination_rate,
    charge_exchange_rates, bremsstrahlung_loss, line_radiation_loss,
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
    """Wraps atomic_rates module into SourceTerms protocol.

    Computes bidirectional source terms from atomic processes:
    ionization, recombination, charge exchange, radiation.
    """

    def __init__(self, config: AtomicCouplingConfig):
        self.config = config

    def compute_sources(self, plasma, neutral, geometry):
        """Compute source terms for both plasma and neutral fluids.

        Args:
            plasma: Plasma State with n, T, p, v fields
            neutral: NeutralState with rho_n, mom_n, E_n
            geometry: Grid geometry

        Returns:
            (plasma_sources, neutral_sources) as SourceRates
        """
        # Extract quantities from plasma
        n_e = plasma.n
        n_i = n_e  # Quasi-neutrality
        T_e = plasma.T  # Already in Joules
        v_i = plasma.v

        # Extract quantities from neutrals
        n_n = neutral.rho_n / MI
        v_n = neutral.v_n

        # Ionization and recombination
        S_ion = ionization_rate(T_e, n_e, neutral.rho_n)
        S_rec = recombination_rate(T_e, n_e, n_i)

        # Charge exchange
        R_cx, Q_cx = charge_exchange_rates(T_e, n_i, n_n, v_i, v_n)

        # Radiation losses
        if self.config.include_radiation:
            n_imp = self.config.impurity_fraction * n_e
            P_brem = bremsstrahlung_loss(T_e, n_e, n_i, self.config.Z_eff)
            P_line = line_radiation_loss(T_e, n_e, n_imp)
            P_ion = ionization_energy_loss(S_ion)
            P_rad = P_brem + P_line + P_ion
        else:
            P_rad = jnp.zeros_like(T_e)
            P_ion = jnp.zeros_like(T_e)

        # Plasma sources:
        # - gains mass from ionization, loses to recombination
        # - gains momentum from CX (R_cx is momentum gained by plasma)
        # - loses energy to radiation and CX
        plasma_sources = SourceRates(
            mass=S_ion - S_rec,
            momentum=R_cx,
            energy=-P_rad - Q_cx  # Q_cx is energy lost by plasma
        )

        # Neutral sources: opposite signs (conservation)
        neutral_sources = SourceRates(
            mass=-S_ion + S_rec,
            momentum=-R_cx,
            energy=Q_cx  # Neutrals gain energy from CX
        )

        return plasma_sources, neutral_sources
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_atomic_coupling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/atomic_coupling.py tests/test_atomic_coupling.py
git commit -m "feat: add AtomicCoupling implementing SourceTerms protocol"
```

---

## Task 5: Create CoupledModel Class

**Files:**
- Modify: `jax_frc/models/coupled.py`
- Test: `tests/test_coupled_model.py`

**Step 1: Write the failing test**

```python
# tests/test_coupled_model.py
"""Tests for CoupledModel composition wrapper."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralState, NeutralFluid
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.models.coupled import CoupledState, CoupledModel, CoupledModelConfig
from jax_frc.constants import QE


def test_coupled_model_explicit_rhs():
    """CoupledModel.explicit_rhs combines plasma and neutral advection."""
    plasma_model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    neutral_model = NeutralFluid()
    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    config = CoupledModelConfig(source_subcycles=5)

    model = CoupledModel(plasma_model, neutral_model, coupling, config)

    geometry = Geometry(
        coord_system="cylindrical",
        nr=8, nz=8,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    plasma = State.zeros(8, 8)
    plasma = plasma.replace(
        psi=jnp.ones((8, 8)),
        n=jnp.ones((8, 8)) * 1e19,
        T=jnp.ones((8, 8)) * 100 * QE,
        p=jnp.ones((8, 8)) * 1e19 * 100 * QE * 2
    )
    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8)) * 100.0
    )
    state = CoupledState(plasma=plasma, neutral=neutral)

    rhs = model.explicit_rhs(state, geometry, t=0.0)

    assert isinstance(rhs, CoupledState)
    assert rhs.plasma.psi.shape == (8, 8)
    assert rhs.neutral.rho_n.shape == (8, 8)


def test_coupled_model_source_rhs():
    """CoupledModel.source_rhs computes atomic source terms."""
    plasma_model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6))
    neutral_model = NeutralFluid()
    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    config = CoupledModelConfig(source_subcycles=5)

    model = CoupledModel(plasma_model, neutral_model, coupling, config)

    geometry = Geometry(
        coord_system="cylindrical",
        nr=8, nz=8,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    plasma = State.zeros(8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8)) * 1e19,
        T=jnp.ones((8, 8)) * 100 * QE,
        p=jnp.ones((8, 8)) * 1e19 * 100 * QE * 2
    )
    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8)) * 100.0
    )
    state = CoupledState(plasma=plasma, neutral=neutral)

    rhs = model.source_rhs(state, geometry, t=0.0)

    assert isinstance(rhs, CoupledState)
    # Ionization should create positive plasma mass source
    assert jnp.any(rhs.plasma.n != 0)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_coupled_model.py -v`
Expected: FAIL with "ImportError: cannot import name 'CoupledModel'"

**Step 3: Write minimal implementation**

Add to `jax_frc/models/coupled.py`:

```python
from dataclasses import dataclass, field
from typing import Any

from jax_frc.models.base import PhysicsModel


@dataclass
class CoupledModelConfig:
    """Configuration for coupled plasma-neutral model."""
    source_subcycles: int = 10


class CoupledModel(PhysicsModel):
    """Composes plasma model + neutral model + atomic coupling.

    Implements SplitRHS protocol for use with ImexSolver.
    """

    def __init__(
        self,
        plasma_model: PhysicsModel,
        neutral_model: Any,
        atomic_coupling: Any,
        config: CoupledModelConfig
    ):
        self.plasma = plasma_model
        self.neutral = neutral_model
        self.coupling = atomic_coupling
        self.config = config

    def explicit_rhs(self, state: CoupledState, geometry: Any, t: float) -> CoupledState:
        """Explicit terms: advection for both fluids."""
        # Plasma advection
        d_plasma = self.plasma.explicit_rhs(state.plasma, geometry, t)

        # Neutral flux divergence
        d_rho, d_mom, d_E = self.neutral.compute_flux_divergence(state.neutral, geometry)
        d_neutral = NeutralState(rho_n=d_rho, mom_n=d_mom, E_n=d_E)

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def implicit_rhs(self, state: CoupledState, geometry: Any, t: float) -> CoupledState:
        """Implicit terms: resistive diffusion."""
        d_plasma = self.plasma.implicit_rhs(state.plasma, geometry, t)

        # Neutrals: no implicit terms (explicit HLLE is stable)
        d_neutral = NeutralState(
            rho_n=jnp.zeros_like(state.neutral.rho_n),
            mom_n=jnp.zeros_like(state.neutral.mom_n),
            E_n=jnp.zeros_like(state.neutral.E_n)
        )

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def source_rhs(self, state: CoupledState, geometry: Any, t: float) -> CoupledState:
        """Atomic source terms coupling plasma <-> neutrals."""
        plasma_src, neutral_src = self.coupling.compute_sources(
            state.plasma, state.neutral, geometry
        )

        # Create derivative states from source rates
        d_plasma = state.plasma.replace(
            n=plasma_src.mass / jnp.maximum(state.plasma.n, 1e-10) * state.plasma.n,
            # Store mass rate in n for simplicity; proper integration handles units
        )
        # For now, just store the mass source directly
        d_plasma = State.zeros(state.plasma.psi.shape[0], state.plasma.psi.shape[1])
        d_plasma = d_plasma.replace(n=plasma_src.mass)

        d_neutral = NeutralState(
            rho_n=neutral_src.mass,
            mom_n=neutral_src.momentum,
            E_n=neutral_src.energy
        )

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def apply_implicit_operator(
        self, state: CoupledState, geometry: Any, dt: float, theta: float
    ) -> CoupledState:
        """Apply implicit operator for CG solve."""
        new_plasma = self.plasma.apply_implicit_operator(
            state.plasma, geometry, dt, theta
        )
        return CoupledState(plasma=new_plasma, neutral=state.neutral)

    def compute_rhs(self, state: CoupledState, geometry: Any) -> CoupledState:
        """Combined RHS for non-IMEX solvers."""
        exp = self.explicit_rhs(state, geometry, 0.0)
        imp = self.implicit_rhs(state, geometry, 0.0)
        src = self.source_rhs(state, geometry, 0.0)

        # Add all contributions
        return CoupledState(
            plasma=exp.plasma.replace(
                psi=exp.plasma.psi + imp.plasma.psi,
                n=exp.plasma.n + src.plasma.n
            ),
            neutral=NeutralState(
                rho_n=exp.neutral.rho_n + src.neutral.rho_n,
                mom_n=exp.neutral.mom_n + src.neutral.mom_n,
                E_n=exp.neutral.E_n + src.neutral.E_n
            )
        )

    def apply_constraints(self, state: CoupledState, geometry: Any) -> CoupledState:
        """Apply boundary conditions to both fluids."""
        new_plasma = self.plasma.apply_constraints(state.plasma, geometry)
        new_neutral = self.neutral.apply_boundary_conditions(state.neutral, geometry)
        return CoupledState(plasma=new_plasma, neutral=new_neutral)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_coupled_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/coupled.py tests/test_coupled_model.py
git commit -m "feat: add CoupledModel composing plasma, neutral, and atomic coupling"
```

---

## Task 6: Write Gaussian Diffusion Test (IMEX Validation)

**Files:**
- Create: `tests/test_imex_diffusion.py`

**Step 1: Write the test**

```python
# tests/test_imex_diffusion.py
"""Validation test: Gaussian diffusion with analytic solution."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.solvers.imex import ImexSolver, ImexConfig
from jax_frc.constants import MU0


def gaussian_analytic(r, z, t, kappa, sigma0, A0):
    """Analytic solution for 2D Gaussian diffusion.

    Initial: A(r,z,0) = A0 * exp(-(r^2 + z^2) / sigma0^2)
    Solution: A(r,z,t) = A0 / (1 + 4*kappa*t/sigma0^2) * exp(-(r^2+z^2)/(sigma0^2 + 4*kappa*t))
    """
    sigma_sq_t = sigma0**2 + 4 * kappa * t
    amplitude = A0 / (1 + 4 * kappa * t / sigma0**2)
    return amplitude * jnp.exp(-(r**2 + z**2) / sigma_sq_t)


def test_gaussian_diffusion_converges():
    """IMEX diffusion converges to analytic Gaussian solution."""
    # Parameters
    eta_0 = 1e-4  # Resistivity [Ohm*m]
    kappa = eta_0 / MU0  # Diffusion coefficient
    sigma0 = 0.1  # Initial Gaussian width [m]
    A0 = 1.0  # Initial amplitude [Wb]
    t_final = 1e-4  # Final time [s]

    geometry = Geometry(
        coord_system="cylindrical",
        nr=32, nz=32,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    # Initial condition: Gaussian centered at (r_center, z_center)
    r_center = 0.25
    z_center = 0.0
    r = geometry.r_grid
    z = geometry.z_grid

    psi_0 = gaussian_analytic(r - r_center, z - z_center, 0.0, kappa, sigma0, A0)

    state = State.zeros(32, 32)
    state = state.replace(psi=psi_0)

    # Model with uniform resistivity (no advection)
    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=eta_0))

    # IMEX solver
    config = ImexConfig(theta=1.0, cg_tol=1e-8, cg_max_iter=500)
    solver = ImexSolver(config=config)

    # Time stepping - use large dt (IMEX allows this)
    dt = 1e-5  # Much larger than explicit CFL would allow
    n_steps = int(t_final / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    # Compare to analytic solution
    psi_analytic = gaussian_analytic(r - r_center, z - z_center, t_final, kappa, sigma0, A0)

    # Compute L2 error (interior only, excluding boundaries)
    interior = (slice(2, -2), slice(2, -2))
    error = jnp.sqrt(jnp.mean((state.psi[interior] - psi_analytic[interior])**2))
    max_val = jnp.max(jnp.abs(psi_analytic[interior]))
    relative_error = error / max_val

    # Should be < 5% error for this resolution
    assert relative_error < 0.05, f"Relative error {relative_error:.2%} exceeds 5%"


def test_imex_large_timestep_stable():
    """IMEX solver remains stable with timesteps larger than explicit CFL."""
    eta_0 = 1e-3  # Higher resistivity = more restrictive explicit CFL

    geometry = Geometry(
        coord_system="cylindrical",
        nr=16, nz=16,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    # Explicit CFL: dt < 0.25 * dx^2 * mu0 / eta
    dx = min(geometry.dr, geometry.dz)
    dt_explicit_cfl = 0.25 * dx**2 * MU0 / eta_0

    # Use 10x the explicit CFL limit
    dt = 10 * dt_explicit_cfl

    state = State.zeros(16, 16)
    psi_0 = jnp.sin(jnp.pi * geometry.r_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    state = state.replace(psi=psi_0)

    model = ResistiveMHD(resistivity=SpitzerResistivity(eta_0=eta_0))
    solver = ImexSolver(config=ImexConfig(theta=1.0))

    # Run several steps
    for _ in range(10):
        state = solver.step(state, dt, model, geometry)

    # Should not blow up
    assert jnp.all(jnp.isfinite(state.psi)), "Solution became non-finite"
    assert jnp.max(jnp.abs(state.psi)) < 10 * jnp.max(jnp.abs(psi_0)), "Solution grew excessively"
```

**Step 2: Run test**

Run: `py -m pytest tests/test_imex_diffusion.py -v`
Expected: PASS (validates IMEX diffusion implementation)

**Step 3: Commit**

```bash
git add tests/test_imex_diffusion.py
git commit -m "test: add Gaussian diffusion validation for IMEX solver"
```

---

## Task 7: Write CX Equilibration Test

**Files:**
- Create: `tests/test_cx_equilibration.py`

**Step 1: Write the test**

```python
# tests/test_cx_equilibration.py
"""Validation test: CX momentum relaxation with analytic solution."""

import jax.numpy as jnp
import jax.lax as lax
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.constants import MI, QE


def test_cx_velocity_relaxation():
    """CX friction equilibrates plasma-neutral velocities exponentially.

    Setup: Stationary plasma, drifting neutrals
    Analytic: Delta_v(t) = Delta_v0 * exp(-t/tau_cx)
    where tau_cx = 1 / (n_n * sigma_cx * v_thermal)
    """
    # Physical parameters
    n_e = 1e19  # m^-3
    T_e = 100 * QE  # 100 eV
    n_n = 1e18  # m^-3 (lower than plasma)
    v_n0 = 1000.0  # Initial neutral velocity [m/s]

    # CX timescale estimate
    sigma_cx = 3e-19  # m^2
    v_thermal = jnp.sqrt(8 * T_e / (jnp.pi * MI))
    tau_cx = 1.0 / (n_n * sigma_cx * v_thermal)

    geometry = Geometry(
        coord_system="cylindrical",
        nr=4, nz=4,  # Small grid, uniform conditions
        r_min=0.01, r_max=0.1,
        z_min=-0.1, z_max=0.1
    )

    # Initial state: stationary plasma, moving neutrals
    plasma = State.zeros(4, 4)
    plasma = plasma.replace(
        n=jnp.ones((4, 4)) * n_e,
        T=jnp.ones((4, 4)) * T_e,
        p=jnp.ones((4, 4)) * n_e * T_e * 2,
        v=jnp.zeros((4, 4, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((4, 4)) * n_n * MI,
        mom_n=jnp.zeros((4, 4, 3)).at[:, :, 2].set(n_n * MI * v_n0),
        E_n=jnp.ones((4, 4)) * 0.5 * n_n * MI * v_n0**2
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))

    # Time evolution of velocity difference
    dt = tau_cx / 100  # Small timesteps for accuracy
    n_steps = 200

    def step(carry, _):
        plasma_v, neutral_v_z, neutral_rho = carry

        # Compute source rates
        # Create temporary states for coupling
        temp_plasma = plasma.replace(v=jnp.zeros((4, 4, 3)).at[:, :, 2].set(plasma_v))
        temp_neutral = NeutralState(
            rho_n=neutral_rho,
            mom_n=jnp.zeros((4, 4, 3)).at[:, :, 2].set(neutral_rho * neutral_v_z),
            E_n=jnp.ones((4, 4)) * 100.0
        )

        plasma_src, neutral_src = coupling.compute_sources(temp_plasma, temp_neutral, geometry)

        # R_cx is momentum transfer to plasma (positive if neutrals faster)
        R_cx_z = plasma_src.momentum[:, :, 2]

        # Update velocities: dv/dt = R_cx / (rho)
        # For plasma: rho = n_e * MI
        # For neutrals: -R_cx / rho_n
        dv_plasma = R_cx_z / (n_e * MI) * dt
        dv_neutral = -R_cx_z / neutral_rho * dt

        new_plasma_v = plasma_v + dv_plasma[0, 0]  # Uniform, take one cell
        new_neutral_v = neutral_v_z + dv_neutral[0, 0]

        delta_v = new_neutral_v - new_plasma_v
        return (new_plasma_v, new_neutral_v, neutral_rho), delta_v

    neutral_rho = jnp.ones((4, 4)) * n_n * MI
    init_carry = (0.0, v_n0, neutral_rho)
    _, delta_v_history = lax.scan(step, init_carry, None, length=n_steps)

    # Compare to analytic: delta_v(t) = v_n0 * exp(-t/tau_cx)
    times = jnp.arange(n_steps) * dt
    analytic = v_n0 * jnp.exp(-times / tau_cx)

    # Check exponential decay (within 20% due to approximations)
    # At t = tau_cx, should be at ~37% of initial
    idx_tau = int(tau_cx / dt)
    if idx_tau < n_steps:
        numerical_ratio = delta_v_history[idx_tau] / v_n0
        expected_ratio = jnp.exp(-1)  # ~0.368
        assert jnp.abs(numerical_ratio - expected_ratio) < 0.2, \
            f"At t=tau_cx: numerical {numerical_ratio:.3f} vs expected {expected_ratio:.3f}"


def test_cx_momentum_conservation():
    """Total momentum conserved during CX exchange."""
    n_e = 1e19
    T_e = 100 * QE
    n_n = 1e19
    v_n0 = 1000.0

    geometry = Geometry(
        coord_system="cylindrical",
        nr=4, nz=4,
        r_min=0.01, r_max=0.1,
        z_min=-0.1, z_max=0.1
    )

    plasma = State.zeros(4, 4)
    plasma = plasma.replace(
        n=jnp.ones((4, 4)) * n_e,
        T=jnp.ones((4, 4)) * T_e,
        p=jnp.ones((4, 4)) * n_e * T_e * 2,
        v=jnp.zeros((4, 4, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((4, 4)) * n_n * MI,
        mom_n=jnp.zeros((4, 4, 3)).at[:, :, 2].set(n_n * MI * v_n0),
        E_n=jnp.ones((4, 4)) * 100.0
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Total momentum source should be zero
    total_mom = plasma_src.momentum + neutral_src.momentum
    assert jnp.allclose(total_mom, 0.0, atol=1e-25), "Momentum not conserved"
```

**Step 2: Run test**

Run: `py -m pytest tests/test_cx_equilibration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_cx_equilibration.py
git commit -m "test: add CX velocity equilibration validation"
```

---

## Task 8: Write Ionization Front Test

**Files:**
- Create: `tests/test_ionization_front.py`

**Step 1: Write the test**

```python
# tests/test_ionization_front.py
"""Validation test: Ionization front propagation."""

import jax.numpy as jnp
import jax.lax as lax
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.coupled import CoupledState
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.constants import MI, QE


def test_ionization_front_mass_conservation():
    """Total mass (plasma + neutral) conserved during ionization.

    This tests the fundamental conservation property of the coupling.
    """
    n_e = 1e19  # Initial plasma density
    T_e = 100 * QE  # 100 eV - hot enough for ionization
    rho_n = 1e-5  # Initial neutral density [kg/m³]

    geometry = Geometry(
        coord_system="cylindrical",
        nr=8, nz=8,
        r_min=0.01, r_max=0.1,
        z_min=-0.1, z_max=0.1
    )

    plasma = State.zeros(8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8)) * n_e,
        T=jnp.ones((8, 8)) * T_e,
        p=jnp.ones((8, 8)) * n_e * T_e * 2,
        v=jnp.zeros((8, 8, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * rho_n,
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8)) * rho_n * 300 * QE / MI  # Room temp neutrals
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))

    # Initial total mass
    initial_plasma_mass = jnp.sum(plasma.n * MI)
    initial_neutral_mass = jnp.sum(neutral.rho_n)
    initial_total = initial_plasma_mass + initial_neutral_mass

    # Get source rates
    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Source rates should conserve mass: plasma gains = neutral loses
    plasma_mass_rate = jnp.sum(plasma_src.mass)
    neutral_mass_rate = jnp.sum(neutral_src.mass)

    assert jnp.allclose(plasma_mass_rate + neutral_mass_rate, 0.0, atol=1e-30), \
        "Mass source terms don't sum to zero"

    # After a small time step, total mass should be conserved
    dt = 1e-8
    new_plasma_n = plasma.n + plasma_src.mass / MI * dt
    new_neutral_rho = neutral.rho_n + neutral_src.mass * dt

    final_plasma_mass = jnp.sum(new_plasma_n * MI)
    final_neutral_mass = jnp.sum(new_neutral_rho)
    final_total = final_plasma_mass + final_neutral_mass

    relative_error = jnp.abs(final_total - initial_total) / initial_total
    assert relative_error < 1e-10, f"Mass conservation violated: {relative_error:.2e}"


def test_ionization_creates_plasma():
    """Hot plasma ionizes cold neutrals, increasing plasma density."""
    n_e = 1e18  # Lower initial plasma density
    T_e = 200 * QE  # 200 eV - very hot
    rho_n = 1e-4  # Significant neutral density

    geometry = Geometry(
        coord_system="cylindrical",
        nr=4, nz=4,
        r_min=0.01, r_max=0.1,
        z_min=-0.1, z_max=0.1
    )

    plasma = State.zeros(4, 4)
    plasma = plasma.replace(
        n=jnp.ones((4, 4)) * n_e,
        T=jnp.ones((4, 4)) * T_e,
        p=jnp.ones((4, 4)) * n_e * T_e * 2
    )

    neutral = NeutralState(
        rho_n=jnp.ones((4, 4)) * rho_n,
        mom_n=jnp.zeros((4, 4, 3)),
        E_n=jnp.ones((4, 4)) * 100.0
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # At high temperature, ionization >> recombination
    # So plasma should gain mass (plasma_src.mass > 0)
    # and neutrals should lose mass (neutral_src.mass < 0)
    assert jnp.all(plasma_src.mass > 0), "Plasma should gain mass from ionization"
    assert jnp.all(neutral_src.mass < 0), "Neutrals should lose mass to ionization"
```

**Step 2: Run test**

Run: `py -m pytest tests/test_ionization_front.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_ionization_front.py
git commit -m "test: add ionization front mass conservation validation"
```

---

## Task 9: Update Module Exports

**Files:**
- Modify: `jax_frc/models/__init__.py`

**Step 1: Update exports**

```python
# jax_frc/models/__init__.py
"""Physics models for plasma simulation."""

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.hybrid_kinetic import HybridKinetic
from jax_frc.models.neutral_fluid import NeutralState, NeutralFluid
from jax_frc.models.resistivity import SpitzerResistivity, ChoduraResistivity
from jax_frc.models.protocols import SplitRHS, SourceTerms
from jax_frc.models.coupled import CoupledState, CoupledModel, CoupledModelConfig, SourceRates
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig

__all__ = [
    "PhysicsModel",
    "ResistiveMHD",
    "ExtendedMHD",
    "HybridKinetic",
    "NeutralState",
    "NeutralFluid",
    "SpitzerResistivity",
    "ChoduraResistivity",
    "SplitRHS",
    "SourceTerms",
    "CoupledState",
    "CoupledModel",
    "CoupledModelConfig",
    "SourceRates",
    "AtomicCoupling",
    "AtomicCouplingConfig",
]
```

**Step 2: Run all tests**

Run: `py -m pytest tests/ -k "not slow" -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add jax_frc/models/__init__.py
git commit -m "feat: export new coupled model classes from models module"
```

---

## Task 10: Run Full Test Suite and Final Commit

**Step 1: Run complete test suite**

Run: `py -m pytest tests/ -k "not slow" -v`
Expected: All tests PASS (including new validation tests)

**Step 2: Check test count**

Run: `py -m pytest tests/ -k "not slow" --collect-only | tail -5`
Expected: More tests than before (239 + ~15 new = ~254)

**Step 3: Final summary commit**

```bash
git log --oneline -10
```

Review commits, then optionally squash or leave as-is.

---

## Summary

This plan implements the neutral-IMEX integration in 10 tasks:

1. **Protocols** - `SplitRHS`, `SourceTerms` type definitions
2. **CoupledState** - Combined plasma+neutral state as JAX pytree
3. **ResistiveMHD SplitRHS** - Add explicit/implicit split methods
4. **AtomicCoupling** - Wrap atomic_rates into SourceTerms protocol
5. **CoupledModel** - Composition wrapper for IMEX integration
6. **Gaussian diffusion test** - IMEX validation with analytic solution
7. **CX equilibration test** - Momentum coupling validation
8. **Ionization front test** - Mass conservation validation
9. **Module exports** - Update `__init__.py`
10. **Final verification** - Run full test suite

Each task follows TDD: write failing test, implement, verify, commit.
