"""Numerical differential operators for plasma simulations.

All operators use central finite differences on regular grids.
Includes both Cartesian (periodic) and cylindrical (non-periodic) operators.
Cylindrical operators handle the 1/r singularity at the axis using L'Hopital's rule.
"""

from typing import TYPE_CHECKING, Tuple
import jax.numpy as jnp
from jax import jit

if TYPE_CHECKING:
    from jax_frc.core.geometry import Geometry

Array = jnp.ndarray


# ============================================================================
# Non-periodic gradient operators (for cylindrical coordinates)
# ============================================================================

def gradient_1d_nonperiodic(f: Array, dh: float, axis: int) -> Array:
    """Compute 1D gradient with non-periodic boundaries.

    Uses central differences in the interior and 2nd-order one-sided
    differences at boundaries.

    Note: Not JIT-decorated because axis must be static. Caller should
    use specific axis versions or call from within a JIT-compiled function.

    Args:
        f: N-dimensional array
        dh: Grid spacing
        axis: Axis along which to differentiate (must be 0 or 1 for 2D)

    Returns:
        Gradient with same shape as f
    """
    # Central difference for interior
    df_central = (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * dh)

    # One-sided differences at boundaries (2nd order)
    # Forward: (-3f_0 + 4f_1 - f_2) / (2*dh)
    # Backward: (3f_n - 4f_{n-1} + f_{n-2}) / (2*dh)

    if axis == 0:
        # Radial direction
        forward_diff = (-3 * f[0, :] + 4 * f[1, :] - f[2, :]) / (2 * dh)
        backward_diff = (3 * f[-1, :] - 4 * f[-2, :] + f[-3, :]) / (2 * dh)
        df = df_central.at[0, :].set(forward_diff)
        df = df.at[-1, :].set(backward_diff)
    elif axis == 1:
        # Axial direction
        forward_diff = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * dh)
        backward_diff = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dh)
        df = df_central.at[:, 0].set(forward_diff)
        df = df.at[:, -1].set(backward_diff)
    else:
        raise ValueError(f"axis must be 0 or 1 for 2D arrays, got {axis}")

    return df


@jit
def gradient_r(f: Array, dr: float) -> Array:
    """Compute gradient in radial direction with non-periodic boundaries."""
    return gradient_1d_nonperiodic(f, dr, axis=0)


@jit
def gradient_z(f: Array, dz: float) -> Array:
    """Compute gradient in axial direction with non-periodic boundaries."""
    return gradient_1d_nonperiodic(f, dz, axis=1)


# ============================================================================
# Cylindrical coordinate operators (with axis singularity handling)
# ============================================================================

@jit
def curl_cylindrical_axisymmetric(
    B_r: Array, B_theta: Array, B_z: Array, dr: float, dz: float, r: Array
) -> Tuple[Array, Array, Array]:
    """Compute curl(B)/mu0 = J in cylindrical (r,theta,z) with axisymmetry.

    Curl in cylindrical coordinates (axisymmetric, d/dtheta = 0):
        J_r = -dB_theta/dz
        J_theta = dB_r/dz - dB_z/dr
        J_z = (1/r)*d(r*B_theta)/dr = B_theta/r + dB_theta/dr

    At r=0, uses L'Hopital's rule:
        lim(r->0) (1/r)*d(r*B_theta)/dr = 2*dB_theta/dr

    Args:
        B_r, B_theta, B_z: Magnetic field components, shape (nr, nz)
        dr, dz: Grid spacings
        r: Radial coordinate array, shape (nr, 1) or broadcastable

    Returns:
        Tuple of (J_r, J_theta, J_z) current density components (not divided by mu0)
    """
    # J_r = -dB_theta/dz
    dB_theta_dz = gradient_z(B_theta, dz)
    J_r = -dB_theta_dz

    # J_theta = dB_r/dz - dB_z/dr
    dB_r_dz = gradient_z(B_r, dz)
    dB_z_dr = gradient_r(B_z, dr)
    J_theta = dB_r_dz - dB_z_dr

    # J_z = (1/r)*d(r*B_theta)/dr = B_theta/r + dB_theta/dr
    dB_theta_dr = gradient_r(B_theta, dr)

    # Regular formula away from axis
    # Avoid division by zero with safe divide
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    J_z = jnp.where(r > 1e-10, B_theta / r_safe + dB_theta_dr, 0.0)

    # L'Hopital at r=0: J_z[0,:] = 2*dB_theta/dr[0,:]
    J_z = J_z.at[0, :].set(2.0 * dB_theta_dr[0, :])

    return J_r, J_theta, J_z


@jit
def divergence_cylindrical(f_r: Array, f_z: Array, dr: float, dz: float, r: Array) -> Array:
    """Compute divergence in cylindrical coordinates (axisymmetric).

    div(f) = (1/r)*d(r*f_r)/dr + df_z/dz
           = f_r/r + df_r/dr + df_z/dz

    At r=0, uses L'Hopital's rule:
        lim(r->0) (1/r)*d(r*f_r)/dr = 2*df_r/dr

    Args:
        f_r, f_z: Vector field components, shape (nr, nz)
        dr, dz: Grid spacings
        r: Radial coordinate array, shape (nr, 1) or broadcastable

    Returns:
        Divergence field with same shape as inputs
    """
    df_r_dr = gradient_r(f_r, dr)
    df_z_dz = gradient_z(f_z, dz)

    # Regular formula away from axis
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    div_f = jnp.where(r > 1e-10, f_r / r_safe + df_r_dr + df_z_dz, 0.0)

    # L'Hopital at r=0: div[0,:] = 2*df_r/dr[0,:] + df_z/dz[0,:]
    div_f = div_f.at[0, :].set(2.0 * df_r_dr[0, :] + df_z_dz[0, :])

    return div_f


@jit
def laplace_star_safe(psi: Array, dr: float, dz: float, r: Array) -> Array:
    """Compute Grad-Shafranov operator Delta* with axis singularity handling.

    Delta* psi = d^2 psi / dr^2 - (1/r) * d psi / dr + d^2 psi / dz^2

    At r=0, uses L'Hopital's rule:
        lim(r->0) (1/r)*dpsi/dr = d^2psi/dr^2
        So Delta*[0,:] = 2*psi_rr[0,:] + psi_zz[0,:]

    Args:
        psi: Poloidal flux function of shape (nr, nz)
        dr: Grid spacing in radial direction
        dz: Grid spacing in axial direction
        r: Radial coordinate array, shape (nr, 1) or broadcastable

    Returns:
        Delta* psi with same shape as psi
    """
    # Second derivatives using non-periodic stencils at boundaries
    # Interior: standard 3-point stencil
    psi_rr_interior = (jnp.roll(psi, -1, axis=0) - 2 * psi + jnp.roll(psi, 1, axis=0)) / (dr**2)
    psi_zz_interior = (jnp.roll(psi, -1, axis=1) - 2 * psi + jnp.roll(psi, 1, axis=1)) / (dz**2)

    # Fix boundary values for second derivatives using one-sided stencils
    # For d2f/dx2 at x=0: (2f_0 - 5f_1 + 4f_2 - f_3) / dx^2
    # For d2f/dx2 at x=n: (2f_n - 5f_{n-1} + 4f_{n-2} - f_{n-3}) / dx^2

    # Radial boundaries
    psi_rr = psi_rr_interior
    psi_rr = psi_rr.at[0, :].set(
        (2*psi[0,:] - 5*psi[1,:] + 4*psi[2,:] - psi[3,:]) / (dr**2)
    )
    psi_rr = psi_rr.at[-1, :].set(
        (2*psi[-1,:] - 5*psi[-2,:] + 4*psi[-3,:] - psi[-4,:]) / (dr**2)
    )

    # Axial boundaries
    psi_zz = psi_zz_interior
    psi_zz = psi_zz.at[:, 0].set(
        (2*psi[:,0] - 5*psi[:,1] + 4*psi[:,2] - psi[:,3]) / (dz**2)
    )
    psi_zz = psi_zz.at[:, -1].set(
        (2*psi[:,-1] - 5*psi[:,-2] + 4*psi[:,-3] - psi[:,-4]) / (dz**2)
    )

    # First derivative for -(1/r)*dpsi/dr term
    psi_r = gradient_r(psi, dr)

    # Compute Delta* away from axis
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    delta_star = jnp.where(
        r > 1e-10,
        psi_rr - (1.0 / r_safe) * psi_r + psi_zz,
        0.0
    )

    # L'Hopital at r=0: Delta*[0,:] = 2*psi_rr[0,:] + psi_zz[0,:]
    delta_star = delta_star.at[0, :].set(2.0 * psi_rr[0, :] + psi_zz[0, :])

    return delta_star


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


def _gradient_3d_tuple(
    f: Array, dx: float, dy: float, dz: float
) -> Tuple[Array, Array, Array]:
    """Compute 3D gradient using central differences (internal helper).

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


def _derivative_with_bc(f: Array, dx: float, axis: int, bc: str, order: int = 4) -> Array:
    """Compute derivative along axis respecting boundary condition.

    Args:
        f: Field array
        dx: Grid spacing
        axis: Axis to differentiate along (0=x, 1=y, 2=z)
        bc: Boundary condition type ("periodic", "dirichlet", "neumann")
        order: Finite difference order (2 or 4, default 4)

    Returns:
        Derivative array with same shape as f
    """
    n = f.shape[axis]

    if bc == "periodic":
        if order == 4 and n >= 5:
            # 4th-order central differences with periodic wrap.
            return (
                -jnp.roll(f, -2, axis=axis)
                + 8 * jnp.roll(f, -1, axis=axis)
                - 8 * jnp.roll(f, 1, axis=axis)
                + jnp.roll(f, 2, axis=axis)
            ) / (12 * dx)
        # 2nd-order central differences with periodic wrap.
        return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * dx)

    # Non-periodic: use 4th-order ghost-cell padding when requested,
    # otherwise central differences with one-sided boundary stencils.
    if order == 4 and n >= 5:
        return _derivative_with_bc_pad(f, axis, dx, bc, order=4)

    ndim = f.ndim
    df = jnp.zeros_like(f)

    # Interior: 2nd-order central differences (no wrap).
    center = [slice(None)] * ndim
    plus = [slice(None)] * ndim
    minus = [slice(None)] * ndim
    center[axis] = slice(1, -1)
    plus[axis] = slice(2, None)
    minus[axis] = slice(None, -2)
    df = df.at[tuple(center)].set((f[tuple(plus)] - f[tuple(minus)]) / (2 * dx))

    # Boundaries: one-sided differences.
    left = [slice(None)] * ndim
    left[axis] = 0
    left_next = [slice(None)] * ndim
    left_next[axis] = 1
    df = df.at[tuple(left)].set((f[tuple(left_next)] - f[tuple(left)]) / dx)

    right = [slice(None)] * ndim
    right[axis] = -1
    right_prev = [slice(None)] * ndim
    right_prev[axis] = -2
    df = df.at[tuple(right)].set((f[tuple(right)] - f[tuple(right_prev)]) / dx)

    return df


def _second_derivative_with_bc(f: Array, dx: float, axis: int, bc: str) -> Array:
    """Compute second derivative along axis respecting boundary condition."""
    if bc == "periodic":
        f_plus = jnp.roll(f, -1, axis=axis)
        f_minus = jnp.roll(f, 1, axis=axis)
        return (f_plus - 2*f + f_minus) / (dx**2)

    # Non-periodic: use ghost cell approach
    # For Neumann: ghost = interior (zero gradient)
    # For Dirichlet: ghost = -interior (zero value)
    ndim = f.ndim

    # Interior: standard central difference
    f_plus = jnp.roll(f, -1, axis=axis)
    f_minus = jnp.roll(f, 1, axis=axis)
    d2f = (f_plus - 2*f + f_minus) / (dx**2)

    # Left boundary correction
    left_slice = [slice(None)] * ndim
    left_slice[axis] = 0
    next_slice = [slice(None)] * ndim
    next_slice[axis] = 1

    if bc == "neumann":
        # Ghost cell = f[0] (zero gradient), so f[-1] = f[0]
        # d2f = (f[1] - 2*f[0] + f[0]) / dx^2 = (f[1] - f[0]) / dx^2
        d2f_left = (f[tuple(next_slice)] - f[tuple(left_slice)]) / (dx**2)
    else:  # dirichlet
        # Ghost cell = -f[0] (zero value at boundary)
        # d2f = (f[1] - 2*f[0] + (-f[0])) / dx^2 = (f[1] - 3*f[0]) / dx^2
        d2f_left = (f[tuple(next_slice)] - 3*f[tuple(left_slice)]) / (dx**2)
    d2f = d2f.at[tuple(left_slice)].set(d2f_left)

    # Right boundary correction
    right_slice = [slice(None)] * ndim
    right_slice[axis] = -1
    prev_slice = [slice(None)] * ndim
    prev_slice[axis] = -2

    if bc == "neumann":
        d2f_right = (f[tuple(prev_slice)] - f[tuple(right_slice)]) / (dx**2)
    else:  # dirichlet
        d2f_right = (f[tuple(prev_slice)] - 3*f[tuple(right_slice)]) / (dx**2)
    d2f = d2f.at[tuple(right_slice)].set(d2f_right)

    return d2f


def _reflect_pad(f: Array, axis: int, n_ghost: int, negate: bool) -> Array:
    """Pad field with reflected ghost cells along a single axis."""
    left = jnp.take(f, jnp.arange(n_ghost), axis=axis)
    left = jnp.flip(left, axis=axis)
    right = jnp.take(f, -1 - jnp.arange(n_ghost), axis=axis)
    if negate:
        left = -left
        right = -right
    return jnp.concatenate([left, f, right], axis=axis)


def _pad_with_bc(f: Array, axis: int, bc: str, n_ghost: int) -> Array:
    """Pad field with ghost cells based on boundary condition."""
    if bc == "periodic":
        left = jnp.take(f, jnp.arange(-n_ghost, 0), axis=axis)
        right = jnp.take(f, jnp.arange(n_ghost), axis=axis)
        return jnp.concatenate([left, f, right], axis=axis)
    if bc == "dirichlet":
        return _reflect_pad(f, axis, n_ghost, negate=True)
    if bc == "neumann":
        return _reflect_pad(f, axis, n_ghost, negate=False)
    raise ValueError(f"Unknown boundary condition: {bc}")


def _derivative_with_bc_pad(f: Array, axis: int, h: float, bc: str, order: int) -> Array:
    """Compute first derivative with ghost-cell padding for non-periodic BCs."""
    if bc == "periodic":
        if order == 4 and f.shape[axis] >= 5:
            return (
                -jnp.roll(f, -2, axis=axis)
                + 8 * jnp.roll(f, -1, axis=axis)
                - 8 * jnp.roll(f, 1, axis=axis)
                + jnp.roll(f, 2, axis=axis)
            ) / (12 * h)
        return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * h)

    use_4th = order == 4 and f.shape[axis] >= 5
    n_ghost = 2 if use_4th else 1
    f_pad = _pad_with_bc(f, axis, bc, n_ghost)
    if use_4th:
        df_pad = (
            -jnp.roll(f_pad, -2, axis=axis)
            + 8 * jnp.roll(f_pad, -1, axis=axis)
            - 8 * jnp.roll(f_pad, 1, axis=axis)
            + jnp.roll(f_pad, 2, axis=axis)
        ) / (12 * h)
    else:
        df_pad = (jnp.roll(f_pad, -1, axis=axis) - jnp.roll(f_pad, 1, axis=axis)) / (2 * h)

    slicer = [slice(None)] * f.ndim
    slicer[axis] = slice(n_ghost, -n_ghost)
    return df_pad[tuple(slicer)]


@jit(static_argnums=(1,))
def gradient_3d(f: Array, geometry: "Geometry") -> Array:
    """Compute gradient of scalar field in 3D Cartesian coordinates.

    Uses central differences in interior, one-sided at non-periodic boundaries.

    Args:
        f: Scalar field, shape (nx, ny, nz)
        geometry: 3D Cartesian geometry with bc_x, bc_y, bc_z settings

    Returns:
        Gradient vector field, shape (nx, ny, nz, 3) with [df/dx, df/dy, df/dz]
    """
    order_x = 4 if geometry.bc_x == "periodic" else 2
    order_y = 4 if geometry.bc_y == "periodic" else 2
    order_z = 4 if geometry.bc_z == "periodic" else 2
    df_dx = _derivative_with_bc(f, geometry.dx, 0, geometry.bc_x, order=order_x)
    df_dy = _derivative_with_bc(f, geometry.dy, 1, geometry.bc_y, order=order_y)
    df_dz = _derivative_with_bc(f, geometry.dz, 2, geometry.bc_z, order=order_z)

    return jnp.stack([df_dx, df_dy, df_dz], axis=-1)


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
def laplacian_3d_explicit(f: Array, dx: float, dy: float, dz: float) -> Array:
    """Compute 3D Laplacian using central differences (explicit grid spacing).

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


@jit(static_argnums=(1,))
def laplacian_3d(f: Array, geometry: "Geometry") -> Array:
    """Compute Laplacian of scalar field in 3D Cartesian coordinates.

    Uses second-derivative stencils with boundary-aware ghost cells.
    """
    d2f_dx2 = _second_derivative_with_bc(f, geometry.dx, 0, geometry.bc_x)
    d2f_dy2 = _second_derivative_with_bc(f, geometry.dy, 1, geometry.bc_y)
    d2f_dz2 = _second_derivative_with_bc(f, geometry.dz, 2, geometry.bc_z)
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
def curl_3d_components(
    f_x: Array, f_y: Array, f_z: Array, dx: float, dy: float, dz: float
) -> Tuple[Array, Array, Array]:
    """Compute 3D curl of a vector field from separate components.

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
def divergence_3d_components(
    f_x: Array, f_y: Array, f_z: Array, dx: float, dy: float, dz: float
) -> Array:
    """Compute 3D divergence of a vector field from separate components.

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


@jit(static_argnums=(1, 2))
def divergence_3d(F: Array, geometry: "Geometry", order: int = 4) -> Array:
    """Compute divergence of vector field in 3D Cartesian coordinates.

    div(F) = dFx/dx + dFy/dy + dFz/dz

    Uses central differences with periodic wrapping via jnp.roll.
    Supports 2nd-order (3-point stencil) or 4th-order (5-point stencil).

    Args:
        F: Vector field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry
        order: Finite difference order (2 or 4, default 4)

    Returns:
        Divergence scalar field, shape (nx, ny, nz)
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
    bc_x, bc_y, bc_z = geometry.bc_x, geometry.bc_y, geometry.bc_z

    dFx_dx = _derivative_with_bc(F[..., 0], dx, 0, bc_x, order=order)
    dFy_dy = _derivative_with_bc(F[..., 1], dy, 1, bc_y, order=order)
    dFz_dz = _derivative_with_bc(F[..., 2], dz, 2, bc_z, order=order)

    return dFx_dx + dFy_dy + dFz_dz


@jit(static_argnums=(1, 2))
def curl_3d(F: Array, geometry: "Geometry", order: int = 4) -> Array:
    """Compute curl of vector field in 3D Cartesian coordinates.

    curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)

    Uses central differences with periodic wrapping via jnp.roll.
    Supports 2nd-order (3-point stencil) or 4th-order (5-point stencil).

    Args:
        F: Vector field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry
        order: Finite difference order (2 or 4, default 4)

    Returns:
        Curl vector field, shape (nx, ny, nz, 3)
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
    bc_x, bc_y, bc_z = geometry.bc_x, geometry.bc_y, geometry.bc_z
    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]

    use_2nd_y = geometry.ny < 5
    use_2nd_z = geometry.nz < 5
    order_y = 2 if use_2nd_y else order
    order_z = 2 if use_2nd_z else order

    dFx_dy = _derivative_with_bc(Fx, dy, 1, bc_y, order=order_y)
    dFx_dz = _derivative_with_bc(Fx, dz, 2, bc_z, order=order_z)
    dFy_dx = _derivative_with_bc(Fy, dx, 0, bc_x, order=order)
    dFy_dz = _derivative_with_bc(Fy, dz, 2, bc_z, order=order_z)
    dFz_dx = _derivative_with_bc(Fz, dx, 0, bc_x, order=order)
    dFz_dy = _derivative_with_bc(Fz, dy, 1, bc_y, order=order_y)

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy

    return jnp.stack([curl_x, curl_y, curl_z], axis=-1)


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
