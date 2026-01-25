# tests/test_linear_solvers.py
"""Tests for linear solvers."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.linear.preconditioners import jacobi_preconditioner
from jax_frc.solvers.linear.cg import conjugate_gradient, CGResult


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
