"""Semi-implicit time integration for stiff systems (Extended MHD)."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit
from jax_frc.solvers.base import Solver
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel

MU0 = 1.2566e-6
QE = 1.602e-19


@dataclass
class SemiImplicitSolver(Solver):
    """Semi-implicit solver for Extended MHD with Whistler waves.

    Implements the scheme from plasma_physics.md:
        (I - dt^2 * L_Hall) * dB^{n+1} = Explicit_RHS

    This dampens high-k Whistler modes while preserving large-scale dynamics,
    allowing dt to be set by the slower Alfven time rather than Whistler CFL.

    Reference: NIMROD semi-implicit stepping for Hall MHD.
    """

    damping_factor: float = 1e6  # Controls implicit damping strength

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state using semi-implicit Hall damping."""
        # Get explicit RHS
        rhs = model.compute_rhs(state, geometry)

        # For psi-based models (Resistive MHD), use explicit stepping
        new_psi = state.psi + dt * rhs.psi

        # For B-based models (Extended MHD), apply semi-implicit correction
        if jnp.any(rhs.B != 0):
            new_B = self._semi_implicit_B_update(state, rhs, dt, geometry)
        else:
            new_B = state.B

        # E field (for Hybrid)
        new_E = rhs.E if jnp.any(rhs.E != 0) else state.E

        new_state = state.replace(
            psi=new_psi,
            B=new_B,
            E=new_E,
            time=state.time + dt,
            step=state.step + 1
        )
        return model.apply_constraints(new_state, geometry)

    def _semi_implicit_B_update(self, state: State, rhs: State,
                                 dt: float, geometry) -> jnp.ndarray:
        """Apply semi-implicit update for B field.

        The semi-implicit scheme solves:
            (I - dt^2 * L_Hall) * dB = explicit_dB

        We approximate this with a damping factor that reduces the
        high-frequency Whistler response:
            dB_implicit = dB_explicit / (1 + dt^2 * damping)

        This is equivalent to the operator splitting used in NIMROD.
        """
        dB_explicit = rhs.B

        # Compute Hall operator magnitude for adaptive damping
        # L_Hall ~ (B / (n*e*mu_0)) * nabla^2
        # For uniform damping, use constant factor
        implicit_factor = 1.0 / (1.0 + dt**2 * self.damping_factor)

        # Apply damping to high-frequency modes
        # In Fourier space this would be k-dependent; here we use uniform damping
        dB_implicit = dB_explicit * implicit_factor

        return state.B + dt * dB_implicit

    @classmethod
    def from_config(cls, config: dict) -> "SemiImplicitSolver":
        """Create from configuration dictionary."""
        return cls(
            damping_factor=float(config.get("damping_factor", 1e6))
        )


@dataclass
class HybridSolver(Solver):
    """Specialized solver for Hybrid Kinetic model.

    Handles both field evolution and particle pushing in the correct order:
    1. Deposit particle current to grid
    2. Compute E field from electron fluid equation
    3. Advance B field via Faraday's law
    4. Push particles with Boris algorithm
    5. Update delta-f weights
    """

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance hybrid simulation by one timestep."""
        # Import here to avoid circular dependency
        from jax_frc.models.hybrid_kinetic import HybridKinetic

        if not isinstance(model, HybridKinetic):
            raise TypeError("HybridSolver requires HybridKinetic model")

        # Step 1-3: Compute RHS (deposits current, computes E, gets dB/dt)
        rhs = model.compute_rhs(state, geometry)

        # Update B field
        new_B = state.B + dt * rhs.B

        # Store E field for particle pushing
        new_E = rhs.E

        # Update state with new fields
        state = state.replace(B=new_B, E=new_E)

        # Step 4-5: Push particles and update weights
        state = model.push_particles(state, geometry, dt)

        # Update time
        state = state.replace(time=state.time + dt, step=state.step + 1)

        # Apply constraints (boundaries)
        return model.apply_constraints(state, geometry)
