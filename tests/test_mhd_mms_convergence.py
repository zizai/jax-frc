"""Manufactured-solution convergence tests for MHD RHS operators."""
from __future__ import annotations

from dataclasses import dataclass
import jax
from jax import jacfwd, vmap
import jax.numpy as jnp
from jax_frc.constants import MI, MU0, QE
from jax_frc.models.resistive_mhd import GAMMA
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.resistive_mhd import ResistiveMHD
from tests.utils.residuals import relative_l2_norm


@dataclass(frozen=True)
class MmsFields:
    """Container for MMS fields and derivatives."""
    n: jnp.ndarray
    p: jnp.ndarray
    v: jnp.ndarray
    B: jnp.ndarray
    dn_dx: jnp.ndarray
    dn_dy: jnp.ndarray
    dn_dz: jnp.ndarray
    dp_dx: jnp.ndarray
    dp_dy: jnp.ndarray
    dp_dz: jnp.ndarray
    dvx_dx: jnp.ndarray
    dvx_dy: jnp.ndarray
    dvx_dz: jnp.ndarray
    dvy_dx: jnp.ndarray
    dvy_dy: jnp.ndarray
    dvy_dz: jnp.ndarray
    dvz_dx: jnp.ndarray
    dvz_dy: jnp.ndarray
    dvz_dz: jnp.ndarray
    dBx_dx: jnp.ndarray
    dBx_dy: jnp.ndarray
    dBx_dz: jnp.ndarray
    dBy_dx: jnp.ndarray
    dBy_dy: jnp.ndarray
    dBy_dz: jnp.ndarray
    dBz_dx: jnp.ndarray
    dBz_dy: jnp.ndarray
    dBz_dz: jnp.ndarray
    k2: float


def make_mms_geometry(n: int) -> Geometry:
    """Create a periodic cube geometry for MMS tests."""
    return Geometry(
        nx=n,
        ny=n,
        nz=n,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        bc_x="periodic",
        bc_y="periodic",
        bc_z="periodic",
    )


def build_mms_fields(geometry: Geometry) -> MmsFields:
    """Generate manufactured fields with analytic derivatives."""
    x = geometry.x_grid
    y = geometry.y_grid
    z = geometry.z_grid
    kx = 2.0 * jnp.pi / (geometry.x_max - geometry.x_min)
    ky = 2.0 * jnp.pi / (geometry.y_max - geometry.y_min)
    kz = 2.0 * jnp.pi / (geometry.z_max - geometry.z_min)
    k2 = float(kx**2 + ky**2 + kz**2)

    sx = jnp.sin(kx * x)
    cx = jnp.cos(kx * x)
    sy = jnp.sin(ky * y)
    cy = jnp.cos(ky * y)
    sz = jnp.sin(kz * z)
    cz = jnp.cos(kz * z)

    n0, n1 = 1.0e19, 1.0e18
    p0, p1 = 1.0e3, 1.0e2
    v1, v2, v3 = 1.0e3, -0.7e3, 0.5e3
    b1, b2, b3 = 0.05, -0.03, 0.04

    n = n0 + n1 * sx * sy * sz
    p = p0 + p1 * cx * sy * cz
    v_x = v1 * sx * cy * cz
    v_y = v2 * cx * sy * cz
    v_z = v3 * cx * cy * sz
    v = jnp.stack([v_x, v_y, v_z], axis=-1)

    B_x = b1 * cx * sy * sz
    B_y = b2 * sx * cy * sz
    B_z = b3 * sx * sy * cz
    B = jnp.stack([B_x, B_y, B_z], axis=-1)

    dn_dx = n1 * kx * cx * sy * sz
    dn_dy = n1 * ky * sx * cy * sz
    dn_dz = n1 * kz * sx * sy * cz

    dp_dx = -p1 * kx * sx * sy * cz
    dp_dy = p1 * ky * cx * cy * cz
    dp_dz = -p1 * kz * cx * sy * sz

    dvx_dx = v1 * kx * cx * cy * cz
    dvx_dy = -v1 * ky * sx * sy * cz
    dvx_dz = -v1 * kz * sx * cy * sz

    dvy_dx = -v2 * kx * sx * sy * cz
    dvy_dy = v2 * ky * cx * cy * cz
    dvy_dz = -v2 * kz * cx * sy * sz

    dvz_dx = -v3 * kx * sx * cy * sz
    dvz_dy = -v3 * ky * cx * sy * sz
    dvz_dz = v3 * kz * cx * cy * cz

    dBx_dx = -b1 * kx * sx * sy * sz
    dBx_dy = b1 * ky * cx * cy * sz
    dBx_dz = b1 * kz * cx * sy * cz

    dBy_dx = b2 * kx * cx * cy * sz
    dBy_dy = -b2 * ky * sx * sy * sz
    dBy_dz = b2 * kz * sx * cy * cz

    dBz_dx = b3 * kx * cx * sy * cz
    dBz_dy = b3 * ky * sx * cy * cz
    dBz_dz = -b3 * kz * sx * sy * sz

    return MmsFields(
        n=n,
        p=p,
        v=v,
        B=B,
        dn_dx=dn_dx,
        dn_dy=dn_dy,
        dn_dz=dn_dz,
        dp_dx=dp_dx,
        dp_dy=dp_dy,
        dp_dz=dp_dz,
        dvx_dx=dvx_dx,
        dvx_dy=dvx_dy,
        dvx_dz=dvx_dz,
        dvy_dx=dvy_dx,
        dvy_dy=dvy_dy,
        dvy_dz=dvy_dz,
        dvz_dx=dvz_dx,
        dvz_dy=dvz_dy,
        dvz_dz=dvz_dz,
        dBx_dx=dBx_dx,
        dBx_dy=dBx_dy,
        dBx_dz=dBx_dz,
        dBy_dx=dBy_dx,
        dBy_dy=dBy_dy,
        dBy_dz=dBy_dz,
        dBz_dx=dBz_dx,
        dBz_dy=dBz_dy,
        dBz_dz=dBz_dz,
        k2=k2,
    )


def analytic_rhs(fields: MmsFields, eta: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute analytic RHS for the manufactured fields."""
    n = fields.n
    p = fields.p
    v = fields.v
    B = fields.B

    rho = MI * n
    rho_safe = jnp.maximum(rho, 1e-20)

    # Continuity: dn/dt = -div(n*v)
    div_nv = (
        fields.dn_dx * v[..., 0] + n * fields.dvx_dx
        + fields.dn_dy * v[..., 1] + n * fields.dvy_dy
        + fields.dn_dz * v[..., 2] + n * fields.dvz_dz
    )
    dn_dt = -div_nv

    # Momentum: dv/dt = -grad(p)/rho + JxB/rho - (v·grad)v
    grad_p = jnp.stack([fields.dp_dx, fields.dp_dy, fields.dp_dz], axis=-1)
    pressure_force = -grad_p / rho_safe[..., None]

    J_x = (fields.dBz_dy - fields.dBy_dz) / MU0
    J_y = (fields.dBx_dz - fields.dBz_dx) / MU0
    J_z = (fields.dBy_dx - fields.dBx_dy) / MU0
    J = jnp.stack([J_x, J_y, J_z], axis=-1)

    JxB = jnp.stack(
        [
            J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
            J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
            J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
        ],
        axis=-1,
    )
    lorentz_force = JxB / rho_safe[..., None]

    v_dot_grad_v = jnp.stack(
        [
            v[..., 0] * fields.dvx_dx + v[..., 1] * fields.dvx_dy + v[..., 2] * fields.dvx_dz,
            v[..., 0] * fields.dvy_dx + v[..., 1] * fields.dvy_dy + v[..., 2] * fields.dvy_dz,
            v[..., 0] * fields.dvz_dx + v[..., 1] * fields.dvz_dy + v[..., 2] * fields.dvz_dz,
        ],
        axis=-1,
    )

    dv_dt = pressure_force + lorentz_force - v_dot_grad_v

    # Pressure: dp/dt = -gamma*p*div(v)
    div_v = fields.dvx_dx + fields.dvy_dy + fields.dvz_dz
    dp_dt = -GAMMA * p * div_v

    # Induction: dB/dt = curl(vxB) + eta/mu0 * laplacian(B)
    v_cross_B = jnp.stack(
        [
            v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
            v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
            v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
        ],
        axis=-1,
    )

    dCx_dx = (
        fields.dvy_dx * B[..., 2]
        + v[..., 1] * fields.dBz_dx
        - fields.dvz_dx * B[..., 1]
        - v[..., 2] * fields.dBy_dx
    )
    dCx_dy = (
        fields.dvy_dy * B[..., 2]
        + v[..., 1] * fields.dBz_dy
        - fields.dvz_dy * B[..., 1]
        - v[..., 2] * fields.dBy_dy
    )
    dCx_dz = (
        fields.dvy_dz * B[..., 2]
        + v[..., 1] * fields.dBz_dz
        - fields.dvz_dz * B[..., 1]
        - v[..., 2] * fields.dBy_dz
    )

    dCy_dx = (
        fields.dvz_dx * B[..., 0]
        + v[..., 2] * fields.dBx_dx
        - fields.dvx_dx * B[..., 2]
        - v[..., 0] * fields.dBz_dx
    )
    dCy_dy = (
        fields.dvz_dy * B[..., 0]
        + v[..., 2] * fields.dBx_dy
        - fields.dvx_dy * B[..., 2]
        - v[..., 0] * fields.dBz_dy
    )
    dCy_dz = (
        fields.dvz_dz * B[..., 0]
        + v[..., 2] * fields.dBx_dz
        - fields.dvx_dz * B[..., 2]
        - v[..., 0] * fields.dBz_dz
    )

    dCz_dx = (
        fields.dvx_dx * B[..., 1]
        + v[..., 0] * fields.dBy_dx
        - fields.dvy_dx * B[..., 0]
        - v[..., 1] * fields.dBx_dx
    )
    dCz_dy = (
        fields.dvx_dy * B[..., 1]
        + v[..., 0] * fields.dBy_dy
        - fields.dvy_dy * B[..., 0]
        - v[..., 1] * fields.dBx_dy
    )
    dCz_dz = (
        fields.dvx_dz * B[..., 1]
        + v[..., 0] * fields.dBy_dz
        - fields.dvy_dz * B[..., 0]
        - v[..., 1] * fields.dBx_dz
    )

    curl_v_cross_B = jnp.stack(
        [
            dCz_dy - dCy_dz,
            dCx_dz - dCz_dx,
            dCy_dx - dCx_dy,
        ],
        axis=-1,
    )

    lap_B = -fields.k2 * B
    dB_dt = curl_v_cross_B + (eta / MU0) * lap_B

    return dn_dt, dv_dt, dp_dt, dB_dt


def compute_errors(model, geometry: Geometry, eta: float) -> dict[str, float]:
    """Compute relative errors for MMS RHS."""
    fields = build_mms_fields(geometry)
    state = State(B=fields.B, E=jnp.zeros_like(fields.B), n=fields.n, p=fields.p, v=fields.v)
    rhs_num = model.compute_rhs(state, geometry)
    dn_dt, dv_dt, dp_dt, dB_dt = analytic_rhs(fields, eta)

    return {
        "n": relative_l2_norm(rhs_num.n, dn_dt),
        "v": relative_l2_norm(rhs_num.v, dv_dt),
        "p": relative_l2_norm(rhs_num.p, dp_dt),
        "B": relative_l2_norm(rhs_num.B, dB_dt),
    }


def assert_convergence(errors_coarse: dict[str, float], errors_fine: dict[str, float]) -> None:
    """Require errors to decrease when grid is refined."""
    for key in ("n", "v", "p", "B"):
        assert errors_fine[key] < errors_coarse[key] * 0.6


def test_resistive_mhd_mms_convergence():
    """MMS RHS errors should decrease with resolution for ResistiveMHD."""
    eta = 1e-4
    geom_coarse = make_mms_geometry(8)
    geom_fine = make_mms_geometry(16)
    model = ResistiveMHD(eta=eta, advection_scheme="central")

    errors_coarse = compute_errors(model, geom_coarse, eta)
    errors_fine = compute_errors(model, geom_fine, eta)

    assert_convergence(errors_coarse, errors_fine)


def test_extended_mhd_mms_convergence():
    """MMS RHS errors should decrease with resolution for ExtendedMHD."""
    eta = 1e-4
    geom_coarse = make_mms_geometry(8)
    geom_fine = make_mms_geometry(16)
    model = ExtendedMHD(
        eta=eta,
        include_hall=False,
        include_electron_pressure=False,
        apply_divergence_cleaning=False,
    )

    errors_coarse = compute_errors(model, geom_coarse, eta)
    errors_fine = compute_errors(model, geom_fine, eta)

    assert_convergence(errors_coarse, errors_fine)


def make_mms_geometry_2d(n: int) -> Geometry:
    """Create periodic 2D geometry (single z plane) for Hall/Te MMS."""
    return Geometry(
        nx=n,
        ny=n,
        nz=1,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
        bc_x="periodic",
        bc_y="periodic",
        bc_z="periodic",
    )


def hall_ep_analytic_dB_dt(geometry: Geometry, eta: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute analytic B RHS with Hall + electron pressure using AD."""
    kx = 2.0 * jnp.pi / (geometry.x_max - geometry.x_min)
    ky = 2.0 * jnp.pi / (geometry.y_max - geometry.y_min)

    n0, n1 = 1.0e19, 2.0e18
    t0, t1 = 200.0 * 1.602e-19, 50.0 * 1.602e-19
    b1, b2, b3 = 0.06, -0.04, 0.05

    def n_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return n0 + n1 * jnp.sin(kx * x) * jnp.sin(ky * y)

    def te_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return t0 + t1 * jnp.cos(kx * x) * jnp.sin(ky * y)

    def b_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        bx = b1 * jnp.sin(kx * x) * jnp.cos(ky * y)
        by = b2 * jnp.cos(kx * x) * jnp.sin(ky * y)
        bz = b3 * jnp.sin(kx * x) * jnp.sin(ky * y)
        return jnp.stack([bx, by, bz])

    def pe_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return n_func(x, y) * te_func(x, y)

    def e_func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        n_val = n_func(x, y)
        b_val = b_func(x, y)
        db_dx = jacfwd(b_func, 0)(x, y)
        db_dy = jacfwd(b_func, 1)(x, y)
        jx = db_dy[2] / MU0
        jy = -db_dx[2] / MU0
        jz = (db_dx[1] - db_dy[0]) / MU0
        j_vec = jnp.stack([jx, jy, jz])

        jxb = jnp.cross(j_vec, b_val)
        dp_dx = jacfwd(pe_func, 0)(x, y)
        dp_dy = jacfwd(pe_func, 1)(x, y)
        grad_pe = jnp.array([dp_dx, dp_dy, 0.0])

        return eta * j_vec + jxb / (n_val * QE) - grad_pe / (n_val * QE)

    def db_dt_point(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        de_dx = jacfwd(e_func, 0)(x, y)
        de_dy = jacfwd(e_func, 1)(x, y)
        dB_x = -de_dy[2]
        dB_y = de_dx[2]
        dB_z = de_dy[0] - de_dx[1]
        return jnp.stack([dB_x, dB_y, dB_z])

    x2d = geometry.x_grid[:, :, 0]
    y2d = geometry.y_grid[:, :, 0]

    b_vals = vmap(vmap(b_func))(x2d, y2d)
    n_vals = vmap(vmap(n_func))(x2d, y2d)
    te_vals = vmap(vmap(te_func))(x2d, y2d)
    db_dt = vmap(vmap(db_dt_point))(x2d, y2d)

    B = b_vals[:, :, None, :]
    n = n_vals[:, :, None]
    Te = te_vals[:, :, None]
    dB_dt = db_dt[:, :, None, :]

    return B, n, Te, dB_dt


def test_extended_mhd_mms_hall_ep_convergence():
    """Hall+electron pressure B RHS should converge with refinement."""
    eta = 1e-4
    geom_coarse = make_mms_geometry_2d(8)
    geom_fine = make_mms_geometry_2d(16)

    model = ExtendedMHD(
        eta=eta,
        include_hall=True,
        include_electron_pressure=True,
        apply_divergence_cleaning=False,
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )

    def compute_b_error(geometry: Geometry) -> float:
        B, n, Te, dB_dt = hall_ep_analytic_dB_dt(geometry, eta)
        state = State(
            B=B,
            E=jnp.zeros_like(B),
            n=n,
            p=jnp.zeros_like(n),
            v=jnp.zeros_like(B),
            Te=Te,
        )
        rhs_num = model.compute_rhs(state, geometry)
        return relative_l2_norm(rhs_num.B, dB_dt)

    error_coarse = compute_b_error(geom_coarse)
    error_fine = compute_b_error(geom_fine)

    assert error_fine < error_coarse * 0.6
