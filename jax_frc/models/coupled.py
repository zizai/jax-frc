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
    mass: Array      # kg/m^3/s
    momentum: Array  # N/m^3 (vector, shape nx,ny,nz,3)
    energy: Array    # W/m^3


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
        # Plasma advection (fallback to compute_rhs for non-IMEX models)
        explicit_fn = getattr(self.plasma, "explicit_rhs", None)
        if callable(explicit_fn):
            d_plasma = explicit_fn(state.plasma, geometry, t)
        else:
            d_plasma = self.plasma.compute_rhs(state.plasma, geometry)

        # Neutral flux divergence
        d_rho, d_mom, d_E = self.neutral.compute_flux_divergence(state.neutral, geometry)
        d_neutral = NeutralState(rho_n=d_rho, mom_n=d_mom, E_n=d_E)

        return CoupledState(plasma=d_plasma, neutral=d_neutral)

    def implicit_rhs(self, state: CoupledState, geometry: Any, t: float) -> CoupledState:
        """Implicit terms: resistive diffusion."""
        implicit_fn = getattr(self.plasma, "implicit_rhs", None)
        if callable(implicit_fn):
            d_plasma = implicit_fn(state.plasma, geometry, t)
        else:
            nx, ny, nz = state.plasma.n.shape
            d_plasma = State.zeros(nx, ny, nz)

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
        # For now, just store the mass source directly in n field
        nx, ny, nz = state.plasma.n.shape
        d_plasma = State.zeros(nx, ny, nz)
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
        apply_fn = getattr(self.plasma, "apply_implicit_operator", None)
        if callable(apply_fn):
            new_plasma = apply_fn(state.plasma, geometry, dt, theta)
        else:
            new_plasma = state.plasma
        return CoupledState(plasma=new_plasma, neutral=state.neutral)

    def compute_rhs(self, state: CoupledState, geometry: Any) -> CoupledState:
        """Combined RHS for non-IMEX solvers."""
        exp = self.explicit_rhs(state, geometry, 0.0)
        imp = self.implicit_rhs(state, geometry, 0.0)
        src = self.source_rhs(state, geometry, 0.0)

        # Add all contributions
        return CoupledState(
            plasma=exp.plasma.replace(
                B=exp.plasma.B + imp.plasma.B,
                n=exp.plasma.n + src.plasma.n
            ),
            neutral=NeutralState(
                rho_n=exp.neutral.rho_n + src.neutral.rho_n,
                mom_n=exp.neutral.mom_n + src.neutral.mom_n,
                E_n=exp.neutral.E_n + src.neutral.E_n
            )
        )

    def compute_stable_dt(self, state: CoupledState, geometry: Any) -> float:
        """Return CFL-stable timestep for coupled system."""
        return self.plasma.compute_stable_dt(state.plasma, geometry)

    def apply_constraints(self, state: CoupledState, geometry: Any) -> CoupledState:
        """Apply boundary conditions to both fluids."""
        new_plasma = self.plasma.apply_constraints(state.plasma, geometry)
        new_neutral = self.neutral.apply_boundary_conditions(state.neutral, geometry)
        return CoupledState(plasma=new_plasma, neutral=new_neutral)
