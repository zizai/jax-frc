"""Tests for 3D equilibrium solver and initializers."""

import jax.numpy as jnp
import pytest
from jax_frc.core.geometry import Geometry
from jax_frc.equilibrium.grad_shafranov import ForceBalanceSolver
from jax_frc.equilibrium.initializers import (
    harris_sheet_3d,
    uniform_field_3d,
    flux_rope_3d,
)


class TestForceBalanceSolver:
    """Test 3D force-balance equilibrium solver."""

    def test_solver_creation(self):
        """Test creating solver with custom parameters."""
        solver = ForceBalanceSolver(max_iterations=100, tolerance=1e-6)
        assert solver.max_iterations == 100
        assert solver.tolerance == 1e-6
        assert solver.relaxation == 0.1  # default value

    def test_solver_default_params(self):
        """Test solver default parameters."""
        solver = ForceBalanceSolver()
        assert solver.max_iterations == 1000
        assert solver.tolerance == 1e-6
        assert solver.relaxation == 0.1

    def test_uniform_pressure_zero_current(self):
        """Uniform pressure with uniform B_z should have zero J x B force."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        solver = ForceBalanceSolver()

        # Uniform pressure
        p = jnp.ones((16, 16, 16)) * 1e3

        # Uniform B_z field (no current)
        B_init = jnp.zeros((16, 16, 16, 3))
        B_init = B_init.at[..., 2].set(0.1)

        # Compute force imbalance
        result = solver.compute_force_imbalance(B_init, p, geom)

        # J x B - grad(p) should be small for uniform fields
        # J = curl(B)/mu0 = 0 for uniform B
        # grad(p) = 0 for uniform p
        assert result.shape == (16, 16, 16, 3)
        assert jnp.max(jnp.abs(result)) < 1e-10

    def test_compute_force_imbalance_shape(self):
        """Test force imbalance output shape."""
        geom = Geometry(nx=8, ny=10, nz=12)
        solver = ForceBalanceSolver()

        p = jnp.ones((8, 10, 12))
        B = jnp.zeros((8, 10, 12, 3))
        B = B.at[..., 2].set(0.1)

        result = solver.compute_force_imbalance(B, p, geom)
        assert result.shape == (8, 10, 12, 3)

    def test_solve_returns_state(self):
        """Test that solve returns a State object with correct structure."""
        geom = Geometry(
            nx=8, ny=8, nz=8,
            x_min=-0.5, x_max=0.5,
            y_min=-0.5, y_max=0.5,
            z_min=-0.5, z_max=0.5,
        )
        solver = ForceBalanceSolver(max_iterations=10, tolerance=1e-6)

        p = jnp.ones((8, 8, 8)) * 1e3
        B_init = jnp.zeros((8, 8, 8, 3))
        B_init = B_init.at[..., 2].set(0.1)

        state = solver.solve(geom, p, B_init)

        # Check state has expected fields
        assert state.B.shape == (8, 8, 8, 3)
        assert state.p.shape == (8, 8, 8)
        assert jnp.allclose(state.p, p)

    def test_solve_with_harris_initial(self):
        """Test solver with Harris sheet initialization."""
        geom = Geometry(
            nx=16, ny=16, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-0.5, z_max=0.5,
        )
        solver = ForceBalanceSolver(max_iterations=10, tolerance=1e-4)

        B_init = harris_sheet_3d(geom, B0=0.1, L=0.2)
        p = jnp.ones((16, 16, 8)) * 1e3

        state = solver.solve(geom, p, B_init)

        # Should return valid state
        assert state.B.shape == (16, 16, 8, 3)
        assert jnp.all(jnp.isfinite(state.B))


class TestHarrisSheetInitializer:
    """Test Harris current sheet initialization."""

    def test_harris_sheet_shape(self):
        """Test Harris sheet output shape."""
        geom = Geometry(nx=16, ny=16, nz=32)
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        assert B.shape == (16, 16, 32, 3)

    def test_harris_sheet_bx_variation(self):
        """Test Harris sheet has Bx variation across y."""
        geom = Geometry(
            nx=8, ny=32, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-2.0, y_max=2.0,
            z_min=-1.0, z_max=1.0,
        )
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)

        # Bx should vary significantly across y (tanh profile)
        assert jnp.max(jnp.abs(B[..., 0])) > 0.05

        # Bx should be antisymmetric about y=0
        # At center y=0: Bx = B0 * tanh(0) = 0
        # Note: With an even grid, y=0 falls between grid points
        # So we check that values near center are small relative to B0
        ny_mid = geom.ny // 2
        # Values near center should be much smaller than asymptotic B0
        assert jnp.abs(B[4, ny_mid, 4, 0]) < 0.05  # ~50% of B0 near center

    def test_harris_sheet_by_bz_zero(self):
        """Test Harris sheet has By = Bz = 0."""
        geom = Geometry(nx=8, ny=16, nz=8)
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)

        assert jnp.allclose(B[..., 1], 0.0, atol=1e-10)
        assert jnp.allclose(B[..., 2], 0.0, atol=1e-10)

    def test_harris_sheet_asymptotic_values(self):
        """Test Harris sheet approaches +/- B0 at large |y|."""
        geom = Geometry(
            nx=8, ny=64, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-5.0, y_max=5.0,  # Large y range
            z_min=-1.0, z_max=1.0,
        )
        B0 = 0.1
        L = 0.2
        B = harris_sheet_3d(geom, B0=B0, L=L)

        # At y >> L: Bx -> B0, at y << -L: Bx -> -B0
        # Check edges approach asymptotic values
        assert jnp.all(B[4, -4:, 4, 0] > 0.099)  # positive y edge
        assert jnp.all(B[4, :4, 4, 0] < -0.099)  # negative y edge


class TestUniformFieldInitializer:
    """Test uniform magnetic field initialization."""

    def test_uniform_field_shape(self):
        """Test uniform field output shape."""
        geom = Geometry(nx=8, ny=10, nz=12)
        B = uniform_field_3d(geom, B0=0.5, direction="z")
        assert B.shape == (8, 10, 12, 3)

    def test_uniform_field_z_direction(self):
        """Test uniform field in z direction."""
        geom = Geometry(nx=8, ny=8, nz=8)
        B0 = 0.2
        B = uniform_field_3d(geom, B0=B0, direction="z")

        assert jnp.allclose(B[..., 0], 0.0)
        assert jnp.allclose(B[..., 1], 0.0)
        assert jnp.allclose(B[..., 2], B0)

    def test_uniform_field_x_direction(self):
        """Test uniform field in x direction."""
        geom = Geometry(nx=8, ny=8, nz=8)
        B0 = 0.3
        B = uniform_field_3d(geom, B0=B0, direction="x")

        assert jnp.allclose(B[..., 0], B0)
        assert jnp.allclose(B[..., 1], 0.0)
        assert jnp.allclose(B[..., 2], 0.0)

    def test_uniform_field_y_direction(self):
        """Test uniform field in y direction."""
        geom = Geometry(nx=8, ny=8, nz=8)
        B0 = 0.15
        B = uniform_field_3d(geom, B0=B0, direction="y")

        assert jnp.allclose(B[..., 0], 0.0)
        assert jnp.allclose(B[..., 1], B0)
        assert jnp.allclose(B[..., 2], 0.0)


class TestFluxRopeInitializer:
    """Test flux rope (FRC-like) initialization."""

    def test_flux_rope_shape(self):
        """Test flux rope output shape."""
        geom = Geometry(nx=16, ny=16, nz=32)
        B = flux_rope_3d(geom, B0=0.1, a=0.3)
        assert B.shape == (16, 16, 32, 3)

    def test_flux_rope_bz_profile(self):
        """Test flux rope Bz profile peaks at center."""
        geom = Geometry(
            nx=32, ny=32, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-0.5, z_max=0.5,
        )
        B0 = 0.1
        a = 0.5
        B = flux_rope_3d(geom, B0=B0, a=a)

        # Bz should be maximum at r=0 (center of domain)
        nx_mid, ny_mid = geom.nx // 2, geom.ny // 2
        Bz_center = B[nx_mid, ny_mid, 4, 2]
        assert jnp.abs(Bz_center - B0) < 0.01  # Should be close to B0

        # Bz should decrease with r
        Bz_edge = B[-1, ny_mid, 4, 2]
        assert Bz_edge < Bz_center

    def test_flux_rope_bz_vanishes_outside(self):
        """Test flux rope Bz vanishes outside core radius."""
        geom = Geometry(
            nx=32, ny=32, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-0.5, z_max=0.5,
        )
        B0 = 0.1
        a = 0.3  # Core radius = 0.3
        B = flux_rope_3d(geom, B0=B0, a=a)

        # At corners (r > a), Bz should be zero
        # Corner is at roughly r = sqrt(2) * 1.0 >> 0.3
        assert jnp.allclose(B[0, 0, 4, 2], 0.0, atol=1e-10)
        assert jnp.allclose(B[-1, -1, 4, 2], 0.0, atol=1e-10)

    def test_flux_rope_theta_component(self):
        """Test flux rope has azimuthal field component."""
        geom = Geometry(
            nx=32, ny=32, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-0.5, z_max=0.5,
        )
        B0 = 0.1
        a = 0.5
        B = flux_rope_3d(geom, B0=B0, a=a)

        # Bx and By should be non-zero away from axis (azimuthal component)
        # Check that there's some azimuthal field in the interior
        Bx_abs = jnp.abs(B[..., 0])
        By_abs = jnp.abs(B[..., 1])

        # Should have some azimuthal component
        assert jnp.max(Bx_abs) > 0.001
        assert jnp.max(By_abs) > 0.001


class TestEquilibriumPhysics:
    """Test physical properties of equilibrium configurations."""

    def test_divergence_free_uniform(self):
        """Test uniform field is divergence-free."""
        from jax_frc.operators import divergence_3d

        geom = Geometry(nx=16, ny=16, nz=16)
        B = uniform_field_3d(geom, B0=0.1, direction="z")

        div_B = divergence_3d(B, geom)
        assert jnp.max(jnp.abs(div_B)) < 1e-10

    def test_divergence_free_flux_rope(self):
        """Test flux rope is approximately divergence-free.

        Note: The simple flux rope initializer is not exactly divergence-free
        due to the radial cutoff. This test checks the divergence is bounded.
        """
        from jax_frc.operators import divergence_3d

        geom = Geometry(
            nx=32, ny=32, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        B = flux_rope_3d(geom, B0=0.1, a=0.5)

        div_B = divergence_3d(B, geom)
        # Interior should have bounded divergence
        # The cutoff at r=a causes some divergence at the boundary
        assert jnp.max(jnp.abs(div_B[8:-8, 8:-8, 4:-4])) < 0.5
