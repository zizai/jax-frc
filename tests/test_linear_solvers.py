# tests/test_linear_solvers.py
"""Tests for linear solvers."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.linear.preconditioners import jacobi_preconditioner
from jax_frc.solvers.linear.cg import conjugate_gradient, CGResult
from tests.utils.cartesian import make_geometry


class TestJacobiPreconditioner:
    """Tests for Jacobi preconditioner."""

    def test_jacobi_divides_by_diagonal(self):
        """Jacobi preconditioner should divide residual by diagonal."""
        diag = jnp.array([2.0, 4.0, 8.0])
        precond = jacobi_preconditioner(diag)

        r = jnp.array([4.0, 8.0, 16.0])
        result = precond(r)

        expected = jnp.array([2.0, 2.0, 2.0])
        assert jnp.allclose(result, expected)

    def test_jacobi_handles_2d_arrays(self):
        """Jacobi should work with 2D arrays (field data)."""
        diag = jnp.ones((4, 4)) * 2.0
        precond = jacobi_preconditioner(diag)

        r = jnp.ones((4, 4)) * 6.0
        result = precond(r)

        expected = jnp.ones((4, 4)) * 3.0
        assert jnp.allclose(result, expected)

    def test_jacobi_jit_compatible(self):
        """Jacobi preconditioner should work under JIT."""
        from jax import jit

        diag = jnp.array([1.0, 2.0, 3.0])
        precond = jacobi_preconditioner(diag)
        precond_jit = jit(precond)

        r = jnp.array([2.0, 4.0, 6.0])
        result = precond_jit(r)

        expected = jnp.array([2.0, 2.0, 2.0])
        assert jnp.allclose(result, expected)


class TestConjugateGradient:
    """Tests for Conjugate Gradient solver."""

    def test_cg_solves_identity(self):
        """CG should solve Ix = b trivially."""
        def identity(x):
            return x

        b = jnp.array([1.0, 2.0, 3.0])
        result = conjugate_gradient(identity, b)

        assert jnp.allclose(result.x, b, rtol=1e-5)
        assert result.converged

    def test_cg_solves_diagonal_system(self):
        """CG should solve diagonal system Dx = b."""
        diag = jnp.array([2.0, 3.0, 4.0])

        def diag_op(x):
            return diag * x

        b = jnp.array([4.0, 9.0, 16.0])
        result = conjugate_gradient(diag_op, b)

        expected = jnp.array([2.0, 3.0, 4.0])
        assert jnp.allclose(result.x, expected, rtol=1e-5)
        assert result.converged

    def test_cg_solves_laplacian_1d(self):
        """CG should solve 1D Laplacian system."""
        n = 10

        # Tridiagonal: -1, 2, -1 (1D Laplacian with Dirichlet BC)
        def laplacian_1d(x):
            result = 2.0 * x
            result = result.at[:-1].add(-x[1:])
            result = result.at[1:].add(-x[:-1])
            return result

        # Known solution: quadratic
        x_true = jnp.arange(n, dtype=jnp.float32)
        b = laplacian_1d(x_true)

        result = conjugate_gradient(laplacian_1d, b, tol=1e-8, max_iter=100)

        assert jnp.allclose(result.x, x_true, rtol=1e-4, atol=1e-6)
        assert result.converged

    def test_cg_with_preconditioner(self):
        """CG should converge faster with preconditioner."""
        diag = jnp.array([10.0, 1.0, 10.0])

        def diag_op(x):
            return diag * x

        b = jnp.array([20.0, 2.0, 30.0])

        # Without preconditioner
        result_no_precond = conjugate_gradient(diag_op, b, tol=1e-10)

        # With Jacobi preconditioner
        precond = jacobi_preconditioner(diag)
        result_precond = conjugate_gradient(diag_op, b, preconditioner=precond, tol=1e-10)

        # Both should converge to same answer
        assert jnp.allclose(result_no_precond.x, result_precond.x, rtol=1e-5)
        # Preconditioner should take fewer or equal iterations
        assert result_precond.iterations <= result_no_precond.iterations

    def test_cg_returns_residual(self):
        """CG should return relative residual."""
        def identity(x):
            return x

        b = jnp.array([1.0, 2.0, 3.0])
        result = conjugate_gradient(identity, b)

        # For identity, residual should be ~0 at convergence
        assert result.residual < 1e-5

    def test_cg_respects_max_iter(self):
        """CG should stop at max_iter if not converged."""
        # Ill-conditioned system that won't converge in 2 iterations
        diag = jnp.array([1.0, 1000.0])

        def diag_op(x):
            return diag * x

        b = jnp.array([1.0, 1000.0])
        result = conjugate_gradient(diag_op, b, max_iter=2, tol=1e-15)

        assert result.iterations <= 2


class TestCGDiffusion:
    """Tests for CG on diffusion-like operators."""

    def test_cg_solves_2d_laplacian(self):
        """CG should solve 2D Laplacian on small grid."""
        nr, nz = 8, 8

        # 2D Laplacian with Dirichlet BC (zeros at boundary)
        # Standard 5-point stencil: center coefficient 4, neighbors -1
        # This is SPD for interior points with zero boundary conditions
        def laplacian_2d(x):
            # Create output array
            result = jnp.zeros_like(x)
            # Interior: standard 5-point stencil
            result = result.at[1:-1, 1:-1].set(
                4.0 * x[1:-1, 1:-1]
                - x[:-2, 1:-1]
                - x[2:, 1:-1]
                - x[1:-1, :-2]
                - x[1:-1, 2:]
            )
            return result

        # Create a smooth RHS (zero at boundaries as required by Dirichlet BC)
        r = jnp.linspace(0, 1, nr)[:, None]
        z = jnp.linspace(0, 1, nz)[None, :]
        b = jnp.sin(jnp.pi * r) * jnp.sin(jnp.pi * z)
        # Ensure boundary is exactly zero
        b = b.at[0, :].set(0.0)
        b = b.at[-1, :].set(0.0)
        b = b.at[:, 0].set(0.0)
        b = b.at[:, -1].set(0.0)

        result = conjugate_gradient(laplacian_2d, b, tol=1e-6, max_iter=200)

        # Verify Ax ≈ b for interior points
        Ax = laplacian_2d(result.x)
        # Check interior convergence (boundary is always zero for this operator)
        interior_close = jnp.allclose(
            Ax[1:-1, 1:-1], b[1:-1, 1:-1], rtol=1e-4, atol=1e-6
        )
        assert interior_close
        assert result.converged

    def test_cg_solves_implicit_diffusion_operator(self):
        """CG should solve (I - dt*D*lap)x = b for implicit diffusion.

        Note: The discrete Laplacian lap = (u[i+1] - 2*u[i] + u[i-1])/dx^2
        is negative semi-definite. So (I - dt*D*lap) has diagonal
        1 + 2*dt*D/dx^2 which is > 1, making it SPD.
        """
        nr, nz = 8, 8
        dr, dz = 1.0, 1.0  # Unit spacing for simplicity
        dt = 0.1
        D = 0.5  # Diffusion coefficient

        # Implicit diffusion operator: (I - dt*D*lap) on interior, identity on boundary
        def implicit_diffusion(x):
            result = x.copy()
            # Standard discrete Laplacian on interior only
            lap_interior = (
                (x[2:, 1:-1] - 2*x[1:-1, 1:-1] + x[:-2, 1:-1]) / dr**2 +
                (x[1:-1, 2:] - 2*x[1:-1, 1:-1] + x[1:-1, :-2]) / dz**2
            )
            # Apply (I - dt*D*lap) only to interior
            result = result.at[1:-1, 1:-1].set(
                x[1:-1, 1:-1] - dt * D * lap_interior
            )
            # Boundary stays as identity: result[boundary] = x[boundary]
            return result

        # RHS: construct b from a known x_true so A*x_true = b
        r = jnp.linspace(0, 1, nr)[:, None]
        z = jnp.linspace(0, 1, nz)[None, :]
        x_true = jnp.exp(-(r-0.5)**2 - (z-0.5)**2)
        b = implicit_diffusion(x_true)

        # Jacobi preconditioner: diagonal of (I - dt*D*lap)
        # Interior: 1 + dt*D*(2/dr^2 + 2/dz^2), Boundary: 1
        diag = jnp.ones((nr, nz))
        interior_diag = 1.0 + dt * D * (2.0/dr**2 + 2.0/dz**2)
        diag = diag.at[1:-1, 1:-1].set(interior_diag)
        precond = jacobi_preconditioner(diag)

        result = conjugate_gradient(
            implicit_diffusion, b,
            preconditioner=precond,
            tol=1e-6, max_iter=100
        )

        # Verify solution matches known x_true
        assert jnp.allclose(result.x, x_true, rtol=1e-4, atol=1e-5)
        assert result.converged


class TestImexSolver:
    """Tests for IMEX time integrator."""

    def test_imex_config_defaults(self):
        """ImexConfig should have sensible defaults."""
        from jax_frc.solvers.imex import ImexConfig

        config = ImexConfig()

        assert config.theta == 1.0  # Backward Euler
        assert config.cg_tol == 1e-6
        assert config.cg_max_iter == 500
        assert config.cfl_factor == 0.4

    def test_imex_solver_creates(self):
        """ImexSolver should instantiate."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig

        config = ImexConfig()
        solver = ImexSolver(config)

        assert solver.config.theta == 1.0


class TestImexDiffusion:
    """Tests for IMEX implicit diffusion."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=8, ny=2, nz=16)

    def test_build_diffusion_operator(self, geometry):
        """Should build implicit diffusion operator."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig

        config = ImexConfig(theta=1.0)
        solver = ImexSolver(config)

        dt = 0.001
        eta = jnp.ones((geometry.nx, geometry.nz)) * 1e-4  # Uniform resistivity

        # Build operator for B_z component
        operator, diag = solver._build_diffusion_operator(
            geometry, dt, eta, component='z'
        )

        # Test that operator is well-formed
        x_mid = geometry.ny // 2
        x = geometry.x_grid[:, x_mid, :]
        z = geometry.z_grid[:, x_mid, :]
        B_test = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * z)
        B_test_3d = jnp.repeat(B_test[:, None, :], geometry.ny, axis=1)
        result = operator(B_test_3d)

        assert result.shape == B_test_3d.shape
        assert jnp.all(jnp.isfinite(result))

    def test_diffusion_operator_identity_at_zero_dt(self, geometry):
        """At dt=0, operator should be identity."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig

        config = ImexConfig(theta=1.0)
        solver = ImexSolver(config)

        dt = 0.0
        eta = jnp.ones((geometry.nx, geometry.nz)) * 1e-4

        operator, _ = solver._build_diffusion_operator(geometry, dt, eta, component='z')

        x_mid = geometry.ny // 2
        x = geometry.x_grid[:, x_mid, :]
        B_test = jnp.sin(jnp.pi * x)
        B_test_3d = jnp.repeat(B_test[:, None, :], geometry.ny, axis=1)
        result = operator(B_test_3d)

        assert jnp.allclose(result, B_test_3d)

    def test_implicit_diffusion_step(self, geometry):
        """Implicit diffusion should solve and update B field."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        config = ImexConfig(theta=1.0, cg_tol=1e-8)
        solver = ImexSolver(config)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with non-zero B_z
        B = jnp.zeros((nx, ny, nz, 3))
        x_mid = geometry.ny // 2
        x = geometry.x_grid[:, x_mid, :]
        z = geometry.z_grid[:, x_mid, :]
        pattern = jnp.sin(jnp.pi * x / geometry.x_max) * jnp.cos(jnp.pi * z / (2 * geometry.z_max))
        B = B.at[:, :, :, 2].set(jnp.repeat(pattern[:, None, :], ny, axis=1))

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=None,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        # Create model with uniform resistivity
        model = ResistiveMHD(eta=1e-4)

        dt = 1e-3
        new_state = solver._implicit_diffusion(state, dt, model, geometry)

        # B should have changed (diffusion smooths it)
        assert not jnp.allclose(new_state.B, state.B)
        # B should remain finite
        assert jnp.all(jnp.isfinite(new_state.B))


class TestImexExplicit:
    """Tests for IMEX explicit step."""

    @pytest.fixture
    def geometry(self):
        return make_geometry(nx=8, ny=2, nz=16)

    def test_explicit_half_step_updates_B(self, geometry):
        """Explicit half step should update B."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        config = ImexConfig()
        solver = ImexSolver(config)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with non-zero B and velocity
        B = jnp.zeros((nx, ny, nz, 3))
        x_mid = geometry.ny // 2
        x = geometry.x_grid[:, x_mid, :]
        z = geometry.z_grid[:, x_mid, :]
        pattern = jnp.sin(jnp.pi * x / geometry.x_max) * jnp.cos(jnp.pi * z / geometry.z_max)
        B = B.at[:, :, :, 2].set(jnp.repeat(pattern[:, None, :], ny, axis=1))
        v = jnp.zeros((nx, ny, nz, 3))
        v = v.at[:, :, :, 0].set(0.1)

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=v,
            Te=None,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        model = ResistiveMHD(eta=1e-6)

        dt = 1e-5
        new_state = solver._explicit_half_step(state, dt, model, geometry)

        assert jnp.all(jnp.isfinite(new_state.B))
        assert not jnp.allclose(new_state.B, state.B)


class TestImexFullStep:
    """Tests for complete IMEX time step."""

    @pytest.fixture
    def geometry(self):
        return make_geometry(nx=6, ny=2, nz=12)

    def test_imex_step_advances_time(self, geometry):
        """IMEX step should advance time and step count."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        config = ImexConfig()
        solver = ImexSolver(config)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        x_mid = geometry.ny // 2
        x = geometry.x_grid[:, x_mid, :]
        z = geometry.z_grid[:, x_mid, :]
        pattern = jnp.sin(jnp.pi * x / geometry.x_max) * jnp.cos(jnp.pi * z / geometry.z_max)
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(jnp.repeat(pattern[:, None, :], ny, axis=1))

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=None,
            Ti=None,
            particles=None,
            time=0.0,
            step=0
        )

        model = ResistiveMHD(eta=1e-4)

        dt = 1e-4
        new_state = solver.step_with_dt(state, dt, model, geometry)

        assert float(new_state.time) == dt
        assert int(new_state.step) == 1
        assert jnp.all(jnp.isfinite(new_state.B))

    def test_imex_step_stable_with_moderate_dt(self, geometry):
        """IMEX should remain stable with moderate dt."""
        from jax_frc.solvers.imex import ImexSolver, ImexConfig
        from jax_frc.core.state import State
        from jax_frc.models.resistive_mhd import ResistiveMHD

        config = ImexConfig(theta=1.0)  # Backward Euler for stability
        solver = ImexSolver(config)

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        x_mid = geometry.ny // 2
        x = geometry.x_grid[:, x_mid, :]
        z = geometry.z_grid[:, x_mid, :]
        pattern = jnp.sin(jnp.pi * x / geometry.x_max) * jnp.cos(jnp.pi * z / geometry.z_max)
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(jnp.repeat(pattern[:, None, :], ny, axis=1))

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=None,
            Ti=None,
            particles=None,
            time=0.0,
            step=0
        )

        model = ResistiveMHD(
            eta=1e-3,
            evolve_density=False,
            evolve_velocity=False,
            evolve_pressure=False,
        )

        dt = 1e-5

        # Run a few steps
        for _ in range(5):
            state = solver.step_with_dt(state, dt, model, geometry)

        # Should remain bounded (not blow up)
        assert jnp.all(jnp.isfinite(state.B))
        assert jnp.max(jnp.abs(state.B)) < 100.0  # Reasonable bound
