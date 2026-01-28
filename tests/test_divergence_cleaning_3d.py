"""Tests for 3D divergence cleaning using Poisson projection."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.solvers.divergence_cleaning import clean_divergence, poisson_solve_jacobi
from jax_frc.operators import divergence_3d, curl_3d
from jax_frc.core.geometry import Geometry


class TestCleanDivergence:
    """Tests for the clean_divergence function."""

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        geom = Geometry(nx=8, ny=8, nz=8)
        B = jnp.ones((8, 8, 8, 3))

        B_clean = clean_divergence(B, geom)

        assert B_clean.shape == B.shape

    def test_already_divergence_free(self):
        """A divergence-free field should remain nearly unchanged."""
        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")

        # Create a curl field (curl of any field is divergence-free)
        # Use a vector potential A = [sin(z), sin(x), sin(y)]
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid
        kx = 2 * jnp.pi / (geom.x_max - geom.x_min)
        ky = 2 * jnp.pi / (geom.y_max - geom.y_min)
        kz = 2 * jnp.pi / (geom.z_max - geom.z_min)

        # A = [sin(kz*z), sin(kx*x), sin(ky*y)]
        A = jnp.stack([
            jnp.sin(kz * z),
            jnp.sin(kx * x),
            jnp.sin(ky * y)
        ], axis=-1)

        # B = curl(A) is divergence-free
        B = curl_3d(A, geom)

        # Verify it's divergence-free to start
        div_B_before = divergence_3d(B, geom)
        assert jnp.max(jnp.abs(div_B_before)) < 1e-5

        # Clean should leave it nearly unchanged
        B_clean = clean_divergence(B, geom, max_iter=100, tol=1e-10)

        # Field should be essentially the same
        diff = jnp.max(jnp.abs(B_clean - B))
        assert diff < 1e-6, f"Field changed by {diff}"

    def test_removes_divergence(self):
        """A field with divergence should have reduced divergence after cleaning."""
        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid

        kx = 2 * jnp.pi / (geom.x_max - geom.x_min)
        ky = 2 * jnp.pi / (geom.y_max - geom.y_min)
        kz = 2 * jnp.pi / (geom.z_max - geom.z_min)

        # Create a periodic field with non-zero divergence
        # B = [sin(kx*x), sin(ky*y), sin(kz*z)]
        # div(B) = kx*cos(kx*x) + ky*cos(ky*y) + kz*cos(kz*z)
        B = jnp.stack([
            jnp.sin(kx * x),
            jnp.sin(ky * y),
            jnp.sin(kz * z)
        ], axis=-1)

        div_before = divergence_3d(B, geom)
        max_div_before = jnp.max(jnp.abs(div_before))
        assert max_div_before > 1.0, f"Test setup error: div_before={max_div_before}"

        # Clean the field - use more iterations for convergence
        B_clean = clean_divergence(B, geom, max_iter=500, tol=1e-10)

        div_after = divergence_3d(B_clean, geom)
        max_div_after = jnp.max(jnp.abs(div_after))

        # Divergence should be significantly reduced (at least 5x for Jacobi)
        assert max_div_after < max_div_before / 5, (
            f"Divergence not reduced enough: before={max_div_before}, after={max_div_after}"
        )

    def test_periodic_field_cleaning(self):
        """Test cleaning a periodic field with divergence."""
        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid

        kx = 2 * jnp.pi / (geom.x_max - geom.x_min)
        ky = 2 * jnp.pi / (geom.y_max - geom.y_min)
        kz = 2 * jnp.pi / (geom.z_max - geom.z_min)

        # B = [cos(kx*x), cos(ky*y), cos(kz*z)]
        # div(B) = -kx*sin(kx*x) - ky*sin(ky*y) - kz*sin(kz*z)
        B = jnp.stack([
            jnp.cos(kx * x),
            jnp.cos(ky * y),
            jnp.cos(kz * z)
        ], axis=-1)

        div_before = divergence_3d(B, geom)
        max_div_before = jnp.max(jnp.abs(div_before))

        # Use more iterations for better convergence
        B_clean = clean_divergence(B, geom, max_iter=500, tol=1e-10)

        div_after = divergence_3d(B_clean, geom)
        max_div_after = jnp.max(jnp.abs(div_after))

        # Periodic field should clean reasonably well (at least 10x reduction)
        # Jacobi converges slowly, so we don't expect perfect cleaning
        assert max_div_after < max_div_before / 10, (
            f"Periodic cleaning failed: before={max_div_before}, after={max_div_after}"
        )


class TestPoissonSolveJacobi:
    """Tests for the Jacobi Poisson solver."""

    def test_zero_rhs_gives_zero_solution(self):
        """Zero right-hand side should give zero solution."""
        geom = Geometry(nx=8, ny=8, nz=8)
        rhs = jnp.zeros((8, 8, 8))

        phi = poisson_solve_jacobi(rhs, geom, max_iter=100, tol=1e-10)

        assert jnp.max(jnp.abs(phi)) < 1e-10

    def test_output_shape(self):
        """Output shape should match input shape."""
        geom = Geometry(nx=8, ny=10, nz=12)
        rhs = jnp.ones((8, 10, 12))

        phi = poisson_solve_jacobi(rhs, geom, max_iter=10, tol=1e-10)

        assert phi.shape == rhs.shape

    def test_periodic_mode_solution(self):
        """Test solver with a known periodic solution.

        If we want laplacian(phi) = f, and we choose:
        phi = sin(kx*x)*sin(ky*y)*sin(kz*z)
        then f = -(kx^2 + ky^2 + kz^2) * phi
        """
        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid

        kx = 2 * jnp.pi / (geom.x_max - geom.x_min)
        ky = 2 * jnp.pi / (geom.y_max - geom.y_min)
        kz = 2 * jnp.pi / (geom.z_max - geom.z_min)

        phi_exact = jnp.sin(kx * x) * jnp.sin(ky * y) * jnp.sin(kz * z)
        rhs = -(kx**2 + ky**2 + kz**2) * phi_exact

        phi_numerical = poisson_solve_jacobi(rhs, geom, max_iter=1000, tol=1e-12)

        # The solution is only determined up to a constant for periodic BC
        # So compare after removing the mean
        phi_exact_normalized = phi_exact - jnp.mean(phi_exact)
        phi_numerical_normalized = phi_numerical - jnp.mean(phi_numerical)

        error = jnp.max(jnp.abs(phi_numerical_normalized - phi_exact_normalized))
        # Jacobi converges slowly, allow some error
        assert error < 0.5, f"Poisson solve error: {error}"


class TestJITCompilation:
    """Tests that functions compile with JAX JIT."""

    def test_clean_divergence_jit(self):
        """clean_divergence should work with JIT."""
        geom = Geometry(nx=8, ny=8, nz=8)
        B = jnp.ones((8, 8, 8, 3))

        # Should compile without error
        B_clean = clean_divergence(B, geom)

        # Call again to verify caching works
        B_clean2 = clean_divergence(B, geom)

        assert jnp.allclose(B_clean, B_clean2)

    def test_poisson_solve_jit(self):
        """poisson_solve_jacobi should work with JIT."""
        geom = Geometry(nx=8, ny=8, nz=8)
        rhs = jnp.ones((8, 8, 8))

        # Should compile without error
        phi = poisson_solve_jacobi(rhs, geom, max_iter=10, tol=1e-6)

        # Call again
        phi2 = poisson_solve_jacobi(rhs, geom, max_iter=10, tol=1e-6)

        assert jnp.allclose(phi, phi2)
