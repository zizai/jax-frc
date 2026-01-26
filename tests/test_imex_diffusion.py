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
