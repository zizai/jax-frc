"""Reconstruction methods with slope limiters.

Implements Piecewise Linear Method (PLM) reconstruction with various
slope limiters for achieving second-order accuracy while preventing
spurious oscillations near discontinuities.

Available Limiters:
    - minmod: Most diffusive, TVD
    - mc_limiter: Monotonized Central with beta parameter (default 1.3)
    - superbee: Least diffusive, can be oscillatory

Reference:
    van Leer (1977) "Towards the ultimate conservative difference scheme"
    AGATE uses mcbeta=1.3 as default
"""

import jax.numpy as jnp
from jax import jit
from typing import Tuple


from functools import partial


@jit
def minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Minmod limiter: returns 0 if signs differ, else min magnitude.

    Args:
        a: First slope
        b: Second slope

    Returns:
        Limited slope
    """
    return jnp.where(
        a * b > 0,
        jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)),
        0.0,
    )


@jit
def mc_limiter(
    a: jnp.ndarray, b: jnp.ndarray, beta: float = 1.3
) -> jnp.ndarray:
    """Monotonized Central (MC) limiter with beta parameter.

    The MC limiter provides a good balance between accuracy and stability.
    AGATE uses beta=1.3 as default.

    Args:
        a: Left slope (q[i] - q[i-1])
        b: Right slope (q[i+1] - q[i])
        beta: Limiter parameter (1 < beta < 2)
            - beta=1: equivalent to minmod (most diffusive)
            - beta=2: superbee (least diffusive, can be oscillatory)
            - beta=1.3: AGATE default (good balance)

    Returns:
        Limited slope
    """
    # Central slope
    c = 0.5 * (a + b)

    # Apply minmod to beta-scaled slopes and central slope
    # slope = minmod(minmod(beta*a, beta*b), c)
    beta_a = beta * a
    beta_b = beta * b

    # First minmod: between beta*a and beta*b
    mm_ab = minmod(beta_a, beta_b)

    # Second minmod: between result and central slope
    return minmod(mm_ab, c)


@partial(jit, static_argnums=(1,))
def reconstruct_plm(
    q: jnp.ndarray,
    axis: int,
    beta: float = 1.3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Piecewise Linear reconstruction at cell interfaces.

    Reconstructs left and right states at cell interfaces using
    PLM with MC-beta limiter for second-order accuracy.

    Args:
        q: Cell-centered values, shape (..., nx, ny, nz, ...)
        axis: Direction of reconstruction (0=x, 1=y, 2=z)
        beta: MC limiter parameter (default 1.3, AGATE default)

    Returns:
        Tuple of (q_L, q_R):
            q_L: Left state at i+1/2 interface (extrapolated from cell i)
            q_R: Right state at i+1/2 interface (extrapolated from cell i+1)
    """
    # Compute slopes
    dq_minus = q - jnp.roll(q, 1, axis=axis)  # q[i] - q[i-1]
    dq_plus = jnp.roll(q, -1, axis=axis) - q  # q[i+1] - q[i]

    # Apply MC limiter
    dq = mc_limiter(dq_minus, dq_plus, beta)

    # Reconstruct at interfaces
    # q_L at interface i+1/2 is extrapolated from cell i to the right
    # q_R at interface i+1/2 is extrapolated from cell i+1 to the left
    q_L = q + 0.5 * dq  # Right edge of cell i
    q_R = jnp.roll(q - 0.5 * dq, -1, axis=axis)  # Left edge of cell i+1

    return q_L, q_R


@partial(jit, static_argnums=(1,))
def reconstruct_plm_component(
    q: jnp.ndarray,
    axis: int,
    beta: float = 1.3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """PLM reconstruction for a single field component.

    Same as reconstruct_plm but optimized for scalar fields.

    Args:
        q: Cell-centered scalar field
        axis: Direction of reconstruction
        beta: MC limiter parameter

    Returns:
        Tuple of (q_L, q_R) at interfaces
    """
    return reconstruct_plm(q, axis, beta)


@partial(jit, static_argnums=(1, 3))
def reconstruct_plm_bc(
    q: jnp.ndarray,
    axis: int,
    beta: float = 1.3,
    bc: str = "periodic",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """PLM reconstruction with boundary handling.

    Args:
        q: Cell-centered values
        axis: Direction of reconstruction
        beta: MC limiter parameter
        bc: Boundary type ("periodic" or zero-gradient for others)

    Returns:
        Tuple of (q_L, q_R) at interfaces
    """
    if bc == "periodic":
        return reconstruct_plm(q, axis, beta)

    pad_width = [(0, 0)] * q.ndim
    pad_width[axis] = (1, 1)
    q_pad = jnp.pad(q, pad_width, mode="edge")

    slicer_mid = [slice(None)] * q.ndim
    slicer_lo = [slice(None)] * q.ndim
    slicer_hi = [slice(None)] * q.ndim
    slicer_mid[axis] = slice(1, -1)
    slicer_lo[axis] = slice(0, -2)
    slicer_hi[axis] = slice(2, None)

    q_mid = q_pad[tuple(slicer_mid)]
    dq_minus = q_mid - q_pad[tuple(slicer_lo)]
    dq_plus = q_pad[tuple(slicer_hi)] - q_mid

    dq = mc_limiter(dq_minus, dq_plus, beta)

    q_L = q_mid + 0.5 * dq
    q_R = q_mid - 0.5 * dq

    # Shift right state to the i+1/2 interface with edge padding
    q_R_pad = jnp.pad(q_R, pad_width, mode="edge")
    slicer_r = [slice(None)] * q.ndim
    slicer_r[axis] = slice(2, None)
    q_R = q_R_pad[tuple(slicer_r)]

    return q_L, q_R
