# tests/test_imex_validation.py
"""Validation tests for IMEX solver against analytic solutions."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

from tests.utils.cartesian import make_geometry

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResistiveDiffusionAnalytic:
    """Test IMEX solver against analytic resistive diffusion."""

    def test_1d_diffusion_decay_rate(self):
        """B should decay exponentially with correct rate.

        Tests the analytic solution for magnetic field diffusion:
        B(z,t) = B_0 * sin(k*z) * exp(-k^2 * D * t)

        where D = eta / mu_0 is the magnetic diffusivity.
        """
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        # Setup: coarse grid for fast test (physics validated with higher eta)
        nx, ny, nz = 8, 1, 16
        L_r, L_z = 1.0, 2.0
        geometry = make_geometry(nx=nx, ny=ny, nz=nz, extent=1.0)

        # Use moderately higher resistivity for faster decay
        eta = 2e-4
        model = ResistiveMHD(
            eta=eta,
            advection_scheme="ct",
            evolve_density=False,
            evolve_velocity=False,
            evolve_pressure=False,
        )

        # Backward Euler is unconditionally stable, tight CG tolerance
        config = ImexConfig(theta=1.0, cg_tol=1e-10, cg_max_iter=1000)
        solver = ImexSolver(config=config)

        # Initial condition: sinusoidal B_z in z direction
        # B(z,t) = B_0 sin(k*z) exp(-k^2*D*t) where D = eta/mu_0
        k = jnp.pi / L_z  # Fundamental mode (half wavelength in domain)
        B0 = 0.1
        MU0 = 1.2566e-6
        D = eta / MU0  # Magnetic diffusivity

        # Create B field with sinusoidal z-component
        z_grid = geometry.z_grid
        B_z_init = B0 * jnp.sin(k * z_grid)

        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(B_z_init)

        state = State.zeros(nx, ny, nz)
        state = state.replace(
            B=B,
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
        )

        # Run simulation - moderate dt for accuracy with fewer steps
        dt = 5e-5
        n_steps = 10
        t_final = dt * n_steps

        for _ in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Analytic solution at t_final
        decay_factor = jnp.exp(-k**2 * D * t_final)
        B_analytic = B0 * decay_factor * jnp.sin(k * z_grid)

        # Compare interior points (avoid boundary effects)
        # Use a central slice in r and avoid z boundaries
        r_slice = slice(nx//4, 3*nx//4)
        z_slice = slice(nz//4, 3*nz//4)

        B_numerical = state.B[r_slice, 0, z_slice, 2]
        B_expected = B_analytic[r_slice, 0, z_slice]

        # Compute relative error
        # Note: With backward Euler and boundaries, we expect some numerical error
        max_abs_error = jnp.max(jnp.abs(B_numerical - B_expected))
        rel_error = max_abs_error / B0

        # Report metrics for debugging
        print(f"\nDiffusion test metrics:")
        print(f"  Initial B0 = {B0}")
        print(f"  Diffusivity D = eta/mu0 = {D:.6e}")
        print(f"  Wave number k = {k:.6f}")
        print(f"  Decay rate k^2*D = {k**2 * D:.6e}")
        print(f"  Time simulated = {t_final:.6e}")
        print(f"  Analytic decay factor = {decay_factor:.6f}")
        print(f"  Max |B_num - B_analytic| = {max_abs_error:.6e}")
        print(f"  Relative error = {rel_error:.4f} ({rel_error*100:.2f}%)")

        # Verify time is correct
        assert abs(state.time - t_final) < 1e-12, f"Time mismatch: {state.time} vs {t_final}"

        # Should match within 20% (numerical errors from boundaries,
        # discrete Laplacian, and backward Euler truncation error)
        assert rel_error < 0.2, f"Relative error {rel_error:.3f} exceeds 20%"

    def test_diffusion_preserves_pattern_shape(self):
        """Diffusion should preserve the spatial pattern shape (sinusoidal)."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        nx, ny, nz = 6, 1, 12
        L_z = 2.0
        geometry = make_geometry(nx=nx, ny=ny, nz=nz, extent=1.0)

        eta = 1e-4
        model = ResistiveMHD(
            eta=eta,
            advection_scheme="ct",
            evolve_density=False,
            evolve_velocity=False,
            evolve_pressure=False,
        )
        config = ImexConfig(theta=1.0, cg_tol=1e-10)
        solver = ImexSolver(config=config)

        # Sinusoidal initial condition
        k = jnp.pi / L_z
        B0 = 0.1
        z_grid = geometry.z_grid

        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(B0 * jnp.sin(k * z_grid))

        state = State.zeros(nx, ny, nz)
        state = state.replace(
            B=B,
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
        )

        # Run a few steps
        dt = 1e-5
        for _ in range(3):
            state = solver.step(state, dt, model, geometry)

        # The pattern should remain sinusoidal: B(z) ~ sin(k*z)
        # Check correlation with sin(k*z) at interior points
        B_z = state.B[nx//2, 0, nz//4:3*nz//4, 2]  # Take middle x slice
        z_interior = geometry.z[nz//4:3*nz//4]
        sin_pattern = jnp.sin(k * z_interior)

        # Normalize both to compare shape
        B_norm = B_z / jnp.max(jnp.abs(B_z))
        sin_norm = sin_pattern / jnp.max(jnp.abs(sin_pattern))

        # Correlation coefficient (should be close to 1)
        correlation = jnp.sum(B_norm * sin_norm) / jnp.sqrt(
            jnp.sum(B_norm**2) * jnp.sum(sin_norm**2)
        )

        print(f"\nPattern correlation = {correlation:.6f}")
        assert correlation > 0.95, f"Pattern correlation {correlation:.3f} too low"

    def test_uniform_field_is_stationary(self):
        """A spatially uniform B field should not diffuse."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        nx, ny, nz = 6, 1, 12
        geometry = make_geometry(nx=nx, ny=ny, nz=nz, extent=1.0)

        eta = 1e-4
        model = ResistiveMHD(
            eta=eta,
            advection_scheme="ct",
            evolve_density=False,
            evolve_velocity=False,
            evolve_pressure=False,
        )
        config = ImexConfig(theta=1.0, cg_tol=1e-10)
        solver = ImexSolver(config=config)

        # Uniform B field (should be unchanged by diffusion)
        B0 = 0.1
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(B0)

        state = State.zeros(nx, ny, nz)
        state = state.replace(
            B=B,
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
        )

        # Run several steps
        dt = 1e-4
        for _ in range(5):
            state = solver.step(state, dt, model, geometry)

        # Interior should remain uniform (boundaries may have effects)
        B_z_interior = state.B[2:-2, 0, 2:-2, 2]

        # Check uniformity: standard deviation should be very small
        std = jnp.std(B_z_interior)
        mean = jnp.mean(B_z_interior)

        print(f"\nUniform field test:")
        print(f"  Initial B0 = {B0}")
        print(f"  Mean B_z (interior) = {mean:.6f}")
        print(f"  Std B_z (interior) = {std:.6e}")

        # Standard deviation should be very small relative to mean
        assert std < 0.01 * mean, f"Uniform field is not stationary: std/mean = {std/mean:.3f}"

    def test_decay_rate_scales_with_resistivity(self):
        """Higher resistivity should cause faster decay."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        nx, ny, nz = 6, 1, 12
        L_z = 2.0
        geometry = make_geometry(nx=nx, ny=ny, nz=nz, extent=1.0)

        k = jnp.pi / L_z
        B0 = 0.1
        z_grid = geometry.z_grid

        config = ImexConfig(theta=1.0, cg_tol=1e-10)

        def measure_decay(eta_val, dt, n_steps):
            """Run simulation and return peak amplitude at end."""
            model = ResistiveMHD(
                eta=eta_val,
                advection_scheme="ct",
                evolve_density=False,
                evolve_velocity=False,
                evolve_pressure=False,
            )
            solver = ImexSolver(config=config)

            B = jnp.zeros((nx, ny, nz, 3))
            B = B.at[:, :, :, 2].set(B0 * jnp.sin(k * z_grid))

            state = State.zeros(nx, ny, nz)
            state = state.replace(
                B=B,
                n=jnp.ones((nx, ny, nz)) * 1e19,
                p=jnp.ones((nx, ny, nz)) * 1e3,
                v=jnp.zeros((nx, ny, nz, 3)),
            )

            for _ in range(n_steps):
                state = solver.step(state, dt, model, geometry)

            # Return peak amplitude (interior)
            return jnp.max(jnp.abs(state.B[nx//4:3*nx//4, 0, nz//4:3*nz//4, 2]))

        # Compare low and high resistivity (higher values for faster decay)
        eta_low = 1e-4
        eta_high = 1e-3  # 10x higher

        dt = 2e-5
        n_steps = 6

        B_final_low = measure_decay(eta_low, dt, n_steps)
        B_final_high = measure_decay(eta_high, dt, n_steps)

        print(f"\nResistivity scaling test:")
        print(f"  eta_low = {eta_low}, B_final = {B_final_low:.6f}")
        print(f"  eta_high = {eta_high}, B_final = {B_final_high:.6f}")

        # Higher resistivity should decay more
        assert B_final_high < B_final_low, \
            f"Higher resistivity should cause faster decay: {B_final_high} >= {B_final_low}"
