"""Residual norms and helpers for RHS consistency tests."""
from __future__ import annotations

import jax.numpy as jnp


def max_abs(value: jnp.ndarray) -> float:
    """Return the maximum absolute value of an array."""
    return float(jnp.max(jnp.abs(value)))


def l2_norm(value: jnp.ndarray) -> float:
    """Return the L2 norm of an array."""
    return float(jnp.linalg.norm(value))


def relative_l2_norm(lhs: jnp.ndarray, rhs: jnp.ndarray, eps: float = 1e-12) -> float:
    """Return relative L2 norm of (lhs - rhs) with safe denominator."""
    scale = jnp.maximum(jnp.max(jnp.abs(rhs)), eps)
    lhs_scaled = lhs / scale
    rhs_scaled = rhs / scale
    denominator = jnp.maximum(jnp.linalg.norm(rhs_scaled), eps)
    return float(jnp.linalg.norm(lhs_scaled - rhs_scaled) / denominator)
