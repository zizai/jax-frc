# tests/test_imex_diffusion.py
"""Validation test: Gaussian diffusion with analytic solution."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.imex import ImexSolver, ImexConfig
from jax_frc.constants import MU0
from tests.utils.cartesian import make_geometry


def gaussian_analytic(x, z, t, kappa, sigma0, A0):
    """Analytic solution for 2D Gaussian diffusion in x-z plane."""
    sigma_sq_t = sigma0**2 + 4 * kappa * t
    amplitude = A0 / (1 + 4 * kappa * t / sigma0**2)
    return amplitude * jnp.exp(-(x**2 + z**2) / sigma_sq_t)


def test_gaussian_diffusion_converges():
    """IMEX diffusion converges to analytic Gaussian solution."""
    # Parameters
    eta_0 = 1e-4  # Resistivity [Ohm*m]
    kappa = eta_0 / MU0  # Diffusion coefficient
    sigma0 = 0.1  # Initial Gaussian width [m]
    A0 = 1.0  # Initial amplitude [Wb]
    t_final = 1e-4  # Final time [s]

    geometry = make_geometry(nx=12, ny=1, nz=12, extent=0.5)

    # Initial condition: Gaussian centered at (r_center, z_center)
    r_center = 0.0
    z_center = 0.0
    x = geometry.x_grid
    z = geometry.z_grid

    B_z_init = gaussian_analytic(x - r_center, z - z_center, 0.0, kappa, sigma0, A0)

    state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
    B_init = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    B_init = B_init.at[:, :, :, 2].set(B_z_init)
    state = state.replace(B=B_init)

    # Model with uniform resistivity (no advection)
    model = ResistiveMHD(eta=eta_0, advection_scheme="ct")

    # IMEX solver
    config = ImexConfig(theta=1.0, cg_tol=1e-6, cg_max_iter=10)
    solver = ImexSolver(config=config)

    # Time stepping - use large dt (IMEX allows this)
    dt = 3e-5  # Much larger than explicit CFL would allow
    n_steps = max(1, int(t_final / dt))

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    # Compare to analytic solution
    B_analytic = gaussian_analytic(x - r_center, z - z_center, t_final, kappa, sigma0, A0)

    # Compute L2 error (interior only, excluding boundaries)
    interior = (slice(2, -2), 0, slice(2, -2))
    B_mid = state.B[:, :, :, 2]
    error = jnp.sqrt(jnp.mean((B_mid[interior] - B_analytic[interior])**2))
    max_val = jnp.max(jnp.abs(B_analytic[interior]))
    relative_error = error / max_val

    # Allow slightly larger error for 3D discretization on coarse grid
    assert relative_error < 0.1, f"Relative error {relative_error:.2%} exceeds 10%"


def test_imex_large_timestep_stable():
    """IMEX solver remains stable with timesteps larger than explicit CFL."""
    eta_0 = 1e-3  # Higher resistivity = more restrictive explicit CFL

    geometry = make_geometry(nx=6, ny=1, nz=6, extent=0.5)

    # Explicit CFL: dt < 0.25 * dx^2 * mu0 / eta
    dx = min(geometry.dx, geometry.dz)
    dt_explicit_cfl = 0.25 * dx**2 * MU0 / eta_0

    # Use 10x the explicit CFL limit
    dt = 10 * dt_explicit_cfl

    state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
    B_init = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    B_init = B_init.at[:, :, :, 2].set(
        jnp.sin(jnp.pi * geometry.x_grid / 0.5) * jnp.cos(jnp.pi * geometry.z_grid / 0.5)
    )
    state = state.replace(B=B_init)

    model = ResistiveMHD(eta=eta_0, advection_scheme="ct")
    solver = ImexSolver(config=ImexConfig(theta=1.0))

    # Run several steps
    for _ in range(3):
        state = solver.step(state, dt, model, geometry)

    # Should not blow up
    assert jnp.all(jnp.isfinite(state.B)), "Solution became non-finite"
    assert jnp.max(jnp.abs(state.B)) < 10 * jnp.max(jnp.abs(B_init)), "Solution grew excessively"
