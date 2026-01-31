"""Validation test: 3D Gaussian magnetic diffusion (pseudo-2D grid)."""

import jax
import jax.numpy as jnp
from jax import lax
import pytest
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.constants import MU0


def gaussian_diffusion_analytic(x, y, z, t, B0, sigma0, eta, dim: int = 3):
    """Analytic solution for Gaussian diffusion in dim dimensions.

    B_z(x,y,z,t) = B0 * (sigma0^2 / sigma(t)^2)^(dim/2) * exp(-r^2 / (2*sigma(t)^2))
    where sigma(t)^2 = sigma0^2 + 2*eta/mu0*t, r^2 = x^2 + y^2 + z^2
    """
    diffusivity = eta / MU0
    sigma_sq = sigma0**2 + 2 * diffusivity * t
    r_sq = x**2 + y**2 + z**2
    amplitude = B0 * (sigma0**2 / sigma_sq) ** (0.5 * dim)
    return amplitude * jnp.exp(-r_sq / (2 * sigma_sq))


def rk4_step(state, dt, model, geometry):
    """Manual RK4 step for 3D state (avoiding incompatible solver).

    The existing RK4Solver references state.psi/time/step which don't exist
    in the 3D State class. This function performs RK4 on just the B field.
    """
    # k1
    k1 = model.compute_rhs(state, geometry)

    # k2
    state_k2 = state.replace(B=state.B + 0.5 * dt * k1.B)
    k2 = model.compute_rhs(state_k2, geometry)

    # k3
    state_k3 = state.replace(B=state.B + 0.5 * dt * k2.B)
    k3 = model.compute_rhs(state_k3, geometry)

    # k4
    state_k4 = state.replace(B=state.B + dt * k3.B)
    k4 = model.compute_rhs(state_k4, geometry)

    # Combine
    new_B = state.B + (dt / 6) * (k1.B + 2 * k2.B + 2 * k3.B + k4.B)

    return state.replace(B=new_B)


def effective_diffusion_dim(geom: Geometry) -> int:
    """Use 2D analytic scaling when any dimension is collapsed."""
    return 2 if 1 in (geom.nx, geom.ny, geom.nz) else 3


class TestGaussianDiffusion3D:
    """Test 3D Gaussian diffusion against analytic solution."""

    @pytest.mark.slow
    def test_diffusion_convergence(self):
        """Test convergence to analytic solution."""
        # Parameters
        eta = 1e-2
        B0 = 1.0
        sigma0 = 0.2
        t_final = 0.001

        # Grid (domain large enough for Gaussian to fit)
        geom = Geometry(
            nx=16,
            ny=16,
            nz=1,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            z_min=-1.0,
            z_max=1.0,
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )
        dim = effective_diffusion_dim(geom)

        # Initial condition
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid
        Bz_init = gaussian_diffusion_analytic(x, y, z, 0.0, B0, sigma0, eta, dim=dim)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[..., 2].set(Bz_init)

        state = State.zeros(geom.nx, geom.ny, geom.nz)
        state = state.replace(
            B=B_init,
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19
        )

        # Setup model
        model = ResistiveMHD(eta=eta)

        # Use stable timestep based on CFL condition
        # dt < dx^2 / (2*eta/mu0) * safety_factor
        dx_min = min(geom.dx, geom.dy, geom.dz)
        diffusivity = eta / MU0
        # 3D explicit diffusion is stable for dt <= O(dx^2 / (6*D)).
        # Use a conservative factor for RK4 to avoid long-time instability.
        dt_stable = 0.1 * dx_min**2 / diffusivity
        dt = min(dt_stable, 1e-5)  # Cap at 1e-5 for safety
        n_steps = int(t_final / dt)

        # Time evolution using a Python loop to avoid XLA instability in long runs.
        final_state = state
        for _ in range(n_steps):
            final_state = rk4_step(final_state, dt, model, geom)

        # Compare to analytic
        actual_time = n_steps * dt
        Bz_analytic = gaussian_diffusion_analytic(
            x, y, z, actual_time, B0, sigma0, eta, dim=dim
        )
        Bz_numeric = final_state.B[..., 2]

        # L2 error (exclude boundary where periodic BC causes issues)
        def interior_slice(n: int) -> slice:
            return slice(3, -3) if n > 6 else slice(None)

        xs = interior_slice(geom.nx)
        ys = interior_slice(geom.ny)
        zs = interior_slice(geom.nz)
        Bz_num_interior = Bz_numeric[xs, ys, zs]
        Bz_ana_interior = Bz_analytic[xs, ys, zs]
        error = jnp.sqrt(jnp.mean((Bz_num_interior - Bz_ana_interior) ** 2))
        rel_error = error / B0

        assert rel_error < 0.1, f"Relative error {rel_error:.3f} exceeds 10%"

    @pytest.mark.slow
    def test_diffusion_field_decay(self):
        """Test that peak field strength decays as expected."""
        # Parameters chosen so diffusion is moderate
        # diffusivity = eta/mu0, choose eta so diffusivity ~ 1
        # For sigma0=0.3 and t_final=0.01, sigma^2 goes from 0.09 to 0.11
        # Expected decay ratio ~ (0.09/0.11)^1.5 ~ 0.74
        eta = 1e-6  # Very low resistivity
        B0 = 1.0
        sigma0 = 0.3
        t_final = 0.01

        # Grid
        geom = Geometry(
            nx=16,
            ny=16,
            nz=1,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            z_min=-1.0,
            z_max=1.0,
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )
        dim = effective_diffusion_dim(geom)

        # Initial condition
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid
        Bz_init = gaussian_diffusion_analytic(x, y, z, 0.0, B0, sigma0, eta, dim=dim)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[..., 2].set(Bz_init)

        state = State.zeros(geom.nx, geom.ny, geom.nz)
        state = state.replace(
            B=B_init,
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19
        )

        # Model and timestep
        model = ResistiveMHD(eta=eta)
        dx_min = min(geom.dx, geom.dy, geom.dz)
        diffusivity = eta / MU0
        # CFL condition for diffusion: dt < dx^2 / (2*D) * safety
        dt = 0.2 * dx_min**2 / diffusivity
        n_steps = max(1, int(t_final / dt))

        # Time evolution
        def step_fn(i, state):
            return rk4_step(state, dt, model, geom)

        final_state = lax.fori_loop(0, n_steps, step_fn, state)

        # Check peak field decay
        initial_peak = jnp.max(Bz_init)
        final_peak = jnp.max(final_state.B[..., 2])

        # Expected decay from analytic formula
        actual_time = n_steps * dt
        sigma_sq_init = sigma0**2
        sigma_sq_final = sigma0**2 + 2 * diffusivity * actual_time
        expected_decay_ratio = (sigma_sq_init / sigma_sq_final) ** (0.5 * dim)

        actual_decay_ratio = float(final_peak / initial_peak)

        # For very slow diffusion (low eta), expect minimal decay
        # The field should remain close to initial
        # Check that both ratios are reasonably close (within 25% relative error)
        rel_error = abs(actual_decay_ratio - expected_decay_ratio) / max(expected_decay_ratio, 0.01)
        assert rel_error < 0.25, (
            f"Peak decay ratio {actual_decay_ratio:.3f} differs from expected "
            f"{expected_decay_ratio:.3f} by more than 25% (rel_error={rel_error:.3f})"
        )

    def test_analytic_solution_properties(self):
        """Test properties of the analytic solution itself."""
        # At t=0, should recover initial Gaussian
        x = jnp.array([0.0, 0.1, 0.2])
        y = jnp.array([0.0, 0.0, 0.0])
        z = jnp.array([0.0, 0.0, 0.0])
        B0 = 1.0
        sigma0 = 0.5
        eta = 1e-2

        Bz_t0 = gaussian_diffusion_analytic(x, y, z, 0.0, B0, sigma0, eta)

        # At r=0, t=0 should equal B0
        assert jnp.isclose(Bz_t0[0], B0, rtol=1e-10)

        # Should decay with r^2
        expected_decay = jnp.exp(-x**2 / (2 * sigma0**2))
        assert jnp.allclose(Bz_t0 / B0, expected_decay, rtol=1e-10)

        # At later time, peak should be lower
        Bz_t1 = gaussian_diffusion_analytic(x, y, z, 0.1, B0, sigma0, eta)
        assert Bz_t1[0] < Bz_t0[0]

    def test_compute_rhs_nonzero_for_gaussian(self):
        """Test that compute_rhs gives nonzero dB/dt for Gaussian field."""
        eta = 1e-2
        B0 = 1.0
        sigma0 = 0.2

        geom = Geometry(nx=16, ny=16, nz=1)
        dim = effective_diffusion_dim(geom)
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid
        Bz_init = gaussian_diffusion_analytic(x, y, z, 0.0, B0, sigma0, eta, dim=dim)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[..., 2].set(Bz_init)

        state = State.zeros(geom.nx, geom.ny, geom.nz)
        state = state.replace(
            B=B_init,
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19
        )

        model = ResistiveMHD(eta=eta)
        rhs = model.compute_rhs(state, geom)

        # Gaussian B field has nonzero curl => nonzero J => nonzero dB/dt
        # Actually for Bz only, curl(B) has x and y components
        # E = eta*J has x and y components
        # curl(E) has z component => dBz/dt != 0
        assert jnp.max(jnp.abs(rhs.B)) > 0, "RHS should be nonzero for Gaussian field"
