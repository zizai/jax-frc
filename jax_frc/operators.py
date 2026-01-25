"""Numerical differential operators for plasma simulations.

All operators use central finite differences on regular grids.
Boundary conditions are handled by the calling code (these operators use periodic rollover).
"""

from typing import Tuple
import jax.numpy as jnp
from jax import jit

Array = jnp.ndarray


@jit
def gradient_2d(f: Array, dx: float, dy: float) -> Tuple[Array, Array]:
    """Compute 2D gradient using central differences.

    Args:
        f: 2D scalar field of shape (nx, ny)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction

    Returns:
        Tuple of (df/dx, df/dy), each with same shape as f
    """
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dy)
    return df_dx, df_dy


@jit
def gradient_3d(
    f: Array, dx: float, dy: float, dz: float
) -> Tuple[Array, Array, Array]:
    """Compute 3D gradient using central differences.

    Args:
        f: 3D scalar field of shape (nx, ny, nz)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        dz: Grid spacing in z direction

    Returns:
        Tuple of (df/dx, df/dy, df/dz), each with same shape as f
    """
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dy)
    df_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2 * dz)
    return df_dx, df_dy, df_dz


@jit
def laplacian_2d(f: Array, dx: float, dy: float) -> Array:
    """Compute 2D Laplacian using central differences.

    Args:
        f: 2D scalar field of shape (nx, ny)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction

    Returns:
        Laplacian field with same shape as f
    """
    d2f_dx2 = (jnp.roll(f, -1, axis=0) - 2 * f + jnp.roll(f, 1, axis=0)) / (dx**2)
    d2f_dy2 = (jnp.roll(f, -1, axis=1) - 2 * f + jnp.roll(f, 1, axis=1)) / (dy**2)
    return d2f_dx2 + d2f_dy2


@jit
def laplacian_3d(f: Array, dx: float, dy: float, dz: float) -> Array:
    """Compute 3D Laplacian using central differences.

    Args:
        f: 3D scalar field of shape (nx, ny, nz)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        dz: Grid spacing in z direction

    Returns:
        Laplacian field with same shape as f
    """
    d2f_dx2 = (jnp.roll(f, -1, axis=0) - 2 * f + jnp.roll(f, 1, axis=0)) / (dx**2)
    d2f_dy2 = (jnp.roll(f, -1, axis=1) - 2 * f + jnp.roll(f, 1, axis=1)) / (dy**2)
    d2f_dz2 = (jnp.roll(f, -1, axis=2) - 2 * f + jnp.roll(f, 1, axis=2)) / (dz**2)
    return d2f_dx2 + d2f_dy2 + d2f_dz2


@jit
def laplace_star(psi: Array, dr: float, dz: float, r: Array) -> Array:
    """Compute Grad-Shafranov operator Delta* in (r, z) cylindrical coordinates.

    Delta* psi = d^2 psi / dr^2 - (1/r) * d psi / dr + d^2 psi / dz^2

    This operator appears in the Grad-Shafranov equation for axisymmetric MHD equilibria.

    Args:
        psi: Poloidal flux function of shape (nr, nz)
        dr: Grid spacing in radial direction
        dz: Grid spacing in axial direction
        r: Radial coordinate array, broadcastable to psi shape

    Returns:
        Delta* psi with same shape as psi
    """
    psi_rr = (jnp.roll(psi, -1, axis=0) - 2 * psi + jnp.roll(psi, 1, axis=0)) / (dr**2)
    psi_zz = (jnp.roll(psi, -1, axis=1) - 2 * psi + jnp.roll(psi, 1, axis=1)) / (dz**2)
    psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)

    return psi_rr - (1.0 / r) * psi_r + psi_zz


@jit
def curl_2d(f_x: Array, f_y: Array, dx: float, dy: float) -> Array:
    """Compute 2D curl (z-component of curl of a 2D vector field).

    curl_z = df_y/dx - df_x/dy

    Args:
        f_x: x-component of vector field, shape (nx, ny)
        f_y: y-component of vector field, shape (nx, ny)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction

    Returns:
        z-component of curl with same shape as inputs
    """
    dfy_dx = (jnp.roll(f_y, -1, axis=0) - jnp.roll(f_y, 1, axis=0)) / (2 * dx)
    dfx_dy = (jnp.roll(f_x, -1, axis=1) - jnp.roll(f_x, 1, axis=1)) / (2 * dy)
    return dfy_dx - dfx_dy


@jit
def curl_3d(
    f_x: Array, f_y: Array, f_z: Array, dx: float, dy: float, dz: float
) -> Tuple[Array, Array, Array]:
    """Compute 3D curl of a vector field.

    Args:
        f_x, f_y, f_z: Components of vector field, each shape (nx, ny, nz)
        dx, dy, dz: Grid spacings

    Returns:
        Tuple of (curl_x, curl_y, curl_z), each with same shape as inputs
    """
    dfz_dy = (jnp.roll(f_z, -1, axis=1) - jnp.roll(f_z, 1, axis=1)) / (2 * dy)
    dfy_dz = (jnp.roll(f_y, -1, axis=2) - jnp.roll(f_y, 1, axis=2)) / (2 * dz)
    curl_x = dfz_dy - dfy_dz

    dfx_dz = (jnp.roll(f_x, -1, axis=2) - jnp.roll(f_x, 1, axis=2)) / (2 * dz)
    dfz_dx = (jnp.roll(f_z, -1, axis=0) - jnp.roll(f_z, 1, axis=0)) / (2 * dx)
    curl_y = dfx_dz - dfz_dx

    dfy_dx = (jnp.roll(f_y, -1, axis=0) - jnp.roll(f_y, 1, axis=0)) / (2 * dx)
    dfx_dy = (jnp.roll(f_x, -1, axis=1) - jnp.roll(f_x, 1, axis=1)) / (2 * dy)
    curl_z = dfy_dx - dfx_dy

    return curl_x, curl_y, curl_z


@jit
def divergence_2d(f_x: Array, f_y: Array, dx: float, dy: float) -> Array:
    """Compute 2D divergence of a vector field.

    Args:
        f_x: x-component of vector field, shape (nx, ny)
        f_y: y-component of vector field, shape (nx, ny)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction

    Returns:
        Divergence with same shape as inputs
    """
    dfx_dx = (jnp.roll(f_x, -1, axis=0) - jnp.roll(f_x, 1, axis=0)) / (2 * dx)
    dfy_dy = (jnp.roll(f_y, -1, axis=1) - jnp.roll(f_y, 1, axis=1)) / (2 * dy)
    return dfx_dx + dfy_dy


@jit
def divergence_3d(
    f_x: Array, f_y: Array, f_z: Array, dx: float, dy: float, dz: float
) -> Array:
    """Compute 3D divergence of a vector field.

    Args:
        f_x, f_y, f_z: Components of vector field, each shape (nx, ny, nz)
        dx, dy, dz: Grid spacings

    Returns:
        Divergence with same shape as inputs
    """
    dfx_dx = (jnp.roll(f_x, -1, axis=0) - jnp.roll(f_x, 1, axis=0)) / (2 * dx)
    dfy_dy = (jnp.roll(f_y, -1, axis=1) - jnp.roll(f_y, 1, axis=1)) / (2 * dy)
    dfz_dz = (jnp.roll(f_z, -1, axis=2) - jnp.roll(f_z, 1, axis=2)) / (2 * dz)
    return dfx_dx + dfy_dy + dfz_dz


@jit
def cross_product_2d(
    a_x: Array, a_y: Array, a_z: Array, b_x: Array, b_y: Array, b_z: Array
) -> Tuple[Array, Array, Array]:
    """Compute 3D cross product for 2D arrays with z-components.

    Args:
        a_x, a_y, a_z: Components of vector a
        b_x, b_y, b_z: Components of vector b

    Returns:
        Tuple of (c_x, c_y, c_z) where c = a × b
    """
    c_x = a_y * b_z - a_z * b_y
    c_y = a_z * b_x - a_x * b_z
    c_z = a_x * b_y - a_y * b_x
    return c_x, c_y, c_z


@jit
def apply_boundary_dirichlet(field: Array, value: float = 0.0) -> Array:
    """Apply Dirichlet boundary conditions to a 2D field.

    Args:
        field: 2D field of shape (nx, ny)
        value: Boundary value (default 0)

    Returns:
        Field with boundary values set
    """
    field = field.at[0, :].set(value)
    field = field.at[-1, :].set(value)
    field = field.at[:, 0].set(value)
    field = field.at[:, -1].set(value)
    return field


@jit
def apply_boundary_neumann(field: Array) -> Array:
    """Apply Neumann (zero gradient) boundary conditions to a 2D field.

    Args:
        field: 2D field of shape (nx, ny)

    Returns:
        Field with zero-gradient boundaries (extrapolated from interior)
    """
    field = field.at[0, :].set(field[1, :])
    field = field.at[-1, :].set(field[-2, :])
    field = field.at[:, 0].set(field[:, 1])
    field = field.at[:, -1].set(field[:, -2])
    return field
