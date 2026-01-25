# tests/test_linear_solvers.py
"""Tests for linear solvers."""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.solvers.linear.preconditioners import jacobi_preconditioner


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
