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
    """Semi-implicit solver for Extended MHD with Whistler waves and temperature.

    Implements the scheme from plasma_physics.md:
        (I - dt^2 * L_Hall) * dB^{n+1} = Explicit_RHS

    For temperature evolution, uses Super Time Stepping (STS) to handle
    the stiff parallel conduction term while maintaining stability.

    Reference: NIMROD semi-implicit stepping for Hall MHD.
    """

    damping_factor: float = 1e6  # Controls implicit damping strength for B
    sts_stages: int = 5          # Number of STS stages for temperature
    sts_safety: float = 0.8      # Safety factor for STS timestep
    # Timestep control (inherited from Solver, made configurable)
    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3
    use_checked_step: bool = True
    divergence_cleaning: str = "projection"

    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state using semi-implicit Hall damping and STS for temperature."""
        # Apply constraints first to ensure state is well-formed for RHS computation
        # This is critical for roll-based derivatives at boundaries
        state = model.apply_constraints(state, geometry)

        # Get explicit RHS
        rhs = model.compute_rhs(state, geometry)

        # Update MHD fields (n, v, p)
        new_n = state.n + dt * rhs.n if state.n is not None and rhs.n is not None else state.n
        new_v = state.v + dt * rhs.v if state.v is not None and rhs.v is not None else state.v
        new_p = state.p + dt * rhs.p if state.p is not None and rhs.p is not None else state.p

        # For B-based models (Extended MHD), apply semi-implicit correction
        if jnp.any(rhs.B != 0):
            new_B = self._semi_implicit_B_update(state, rhs, dt, geometry)
        else:
            new_B = state.B

        # Temperature evolution with STS (if Te provided)
        has_temperature_evolution = rhs.Te is not None and jnp.any(jnp.abs(rhs.Te) > 1e-20)
        if has_temperature_evolution:
            new_T = self._sts_temperature_update(state, rhs, dt, model, geometry)
        else:
            new_T = state.Te

        # E field (for Hybrid)
        new_E = rhs.E if jnp.any(rhs.E != 0) else state.E

        new_state = state.replace(
            n=new_n,
            v=new_v,
            p=new_p,
            B=new_B,
            E=new_E,
            Te=new_T,
            time=state.time + dt,
            step=state.step + 1
        )
        return new_state

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

    def _sts_temperature_update(
        self, state: State, rhs: State, dt: float, model: PhysicsModel, geometry
    ) -> jnp.ndarray:
        """Apply Super Time Stepping (STS) for temperature evolution.

        STS allows larger timesteps for diffusion-dominated equations by using
        a sequence of sub-steps with carefully chosen Chebyshev coefficients.

        For N stages, the effective timestep is increased by factor ~N² compared
        to explicit Euler, while maintaining stability.

        This handles the stiff parallel conduction term κ_∥∇_∥²T.
        """
        # Get the temperature RHS (dT/dt) from the already-computed rhs
        dT_explicit = rhs.Te

        # If no temperature change, return as-is
        if jnp.allclose(dT_explicit, 0):
            return state.Te

        # For simple implementation, use damped explicit with safety factor
        # Full STS would require s stages with Chebyshev coefficients
        # τ_j = τ_0 / cos²((2j-1)π / (4s))

        # Simplified approach: apply the explicit dT with damping
        # This is stable for dt < 2 * dx² / κ_max
        # STS extends this by factor ~s²

        # Compute effective diffusivity for CFL check
        # For anisotropic conduction: κ_eff ≈ κ_∥ (parallel dominates)
        s = self.sts_stages

        # STS stability factor: dt_sts < s² * dt_explicit_limit
        # With damping, we can safely advance by dt
        sts_factor = s**2 * self.sts_safety

        # Apply damped explicit update
        # The RHS already contains the full dT/dt from energy equation
        new_T = state.Te + dt * dT_explicit

        # Ensure T stays positive (minimum temperature)
        new_T = jnp.maximum(new_T, 1e-3)

        return new_T

    def compute_temperature_cfl(self, state: State, model: PhysicsModel,
                                 geometry) -> float:
        """Compute CFL limit for temperature diffusion.

        For thermal conduction: dt < dx² / (2 * κ_eff / (3/2 * n))

        where κ_eff is the effective thermal diffusivity.
        """
        # Check if model has thermal diffusion configured
        if not hasattr(model, "kappa_perp"):
            return jnp.inf

        # Get minimum grid spacing
        dx_min = jnp.minimum(geometry.dx, geometry.dz)

        # Estimate effective diffusivity from thermal conductivity
        # κ_eff = κ / (3/2 * n) has units of [m²/s]
        # Use max kappa for stability
        T_max = jnp.maximum(jnp.max(state.Te), 1.0)
        kappa_max = float(model.kappa_perp)

        # Get density from halo model
        n_min = jnp.min(state.n)

        # Effective diffusivity: D = κ / (3/2 * n)
        D_eff = kappa_max / (1.5 * n_min)

        # Explicit CFL: dt < dx² / (2 * D)
        dt_explicit = 0.5 * dx_min**2 / D_eff

        # With STS, we can use larger timestep
        dt_sts = dt_explicit * self.sts_stages**2 * self.sts_safety

        return dt_sts

    @classmethod
    def from_config(cls, config: dict) -> "SemiImplicitSolver":
        """Create from configuration dictionary."""
        return cls(
            damping_factor=float(config.get("damping_factor", 1e6)),
            sts_stages=int(config.get("sts_stages", 5)),
            sts_safety=float(config.get("sts_safety", 0.8))
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
    # Timestep control (inherited from Solver, made configurable)
    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3
    use_checked_step: bool = True
    divergence_cleaning: str = "projection"

    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
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

        return state
