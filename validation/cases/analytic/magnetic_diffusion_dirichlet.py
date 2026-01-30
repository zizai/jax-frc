"""2D Magnetic Field Diffusion Validation with Dirichlet Boundaries.

Physics:
    ∂B/∂t = η∇²B with B=0 at all boundaries (conducting walls).

Analytic solution (Fourier sine series on [0, L] × [0, L]):
    B(x,y,t) = Σ A_nm sin(nπx/L) sin(mπy/L) exp(-λ_nm η t)
    λ_nm = π²(n² + m²) / L²
"""

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.explicit import RK4Solver
from jax_frc.operators import apply_boundary_dirichlet
from jax_frc.validation.metrics import l2_error
from validation.utils.reporting import ValidationReport


project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


NAME = "magnetic_diffusion_dirichlet"
DESCRIPTION = "Magnetic diffusion with Dirichlet (conducting wall) boundaries"

MODES = [
    (1, 1, 1.0),
    (2, 1, 0.3),
    (1, 2, 0.3),
    (2, 2, 0.1),
]


def _analytic_dirichlet(x: jnp.ndarray, y: jnp.ndarray, t: float, L: float, eta: float) -> jnp.ndarray:
    B = jnp.zeros_like(x)
    for n, m, A in MODES:
        lambda_nm = (jnp.pi**2) * (n**2 + m**2) / (L**2)
        B = B + A * jnp.sin(n * jnp.pi * x / L) * jnp.sin(m * jnp.pi * y / L) * jnp.exp(
            -lambda_nm * eta * t
        )
    return B


def setup_configuration() -> dict:
    eta = 1.26e-10
    nx = 64
    ny = 64
    nz = 1
    L = 1.0
    dx = L / nx
    dt = 0.2 * dx * dx / (eta / 1.2566e-6)
    t_end = 5 * dt
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "L": L,
        "eta": eta,
        "dt": dt,
        "t_end": t_end,
    }


def run_simulation(cfg: dict) -> tuple[State, Geometry]:
    geom = Geometry(
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        x_min=0.0,
        x_max=cfg["L"],
        y_min=0.0,
        y_max=cfg["L"],
        z_min=0.0,
        z_max=cfg["L"],
        bc_x="dirichlet",
        bc_y="dirichlet",
        bc_z="dirichlet",
    )
    x = geom.x_grid
    y = geom.y_grid
    Bz = _analytic_dirichlet(x, y, t=0.0, L=cfg["L"], eta=cfg["eta"])
    B = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
    B = B.at[:, :, :, 2].set(Bz)

    state = State(
        B=B,
        E=jnp.zeros((geom.nx, geom.ny, geom.nz, 3)),
        n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e23,
        p=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e23 * 100.0,
        v=jnp.zeros((geom.nx, geom.ny, geom.nz, 3)),
    )

    model = ResistiveMHD(
        eta=cfg["eta"],
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )
    solver = RK4Solver()

    t_end = cfg["t_end"]
    dt = cfg["dt"]
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        state = solver.step(state, dt, model, geom)
        # Enforce Dirichlet B=0 at boundaries (conducting walls).
        B = state.B
        for comp in range(3):
            B_slice = apply_boundary_dirichlet(B[:, :, 0, comp], value=0.0)
            B = B.at[:, :, 0, comp].set(B_slice)
        state = state.replace(B=B)

    return state, geom


def run_validation() -> bool:
    cfg = setup_configuration()
    t_start = time.time()
    final_state, geometry = run_simulation(cfg)
    t_sim = time.time() - t_start

    z_idx = geometry.nz // 2
    Bz_sim = final_state.B[:, :, z_idx, 2]
    x_2d = geometry.x_grid[:, :, z_idx]
    y_2d = geometry.y_grid[:, :, z_idx]
    Bz_analytic = _analytic_dirichlet(x_2d, y_2d, cfg["t_end"], cfg["L"], cfg["eta"])

    l2_err = float(l2_error(Bz_sim, Bz_analytic))
    boundary_max = float(
        jnp.max(
            jnp.abs(
                jnp.concatenate(
                    [
                        Bz_sim[0, :],
                        Bz_sim[-1, :],
                        Bz_sim[:, 0],
                        Bz_sim[:, -1],
                    ]
                )
            )
        )
    )

    metrics = {
        "l2_error": {
            "value": l2_err,
            "threshold": 0.05,
            "passed": l2_err < 0.05,
            "description": "L2 error against Dirichlet analytic solution",
        },
        "boundary_max": {
            "value": boundary_max,
            "threshold": 1e-2,
            "passed": boundary_max < 1e-2,
            "description": "Max |B| on boundaries",
        },
    }
    overall_pass = metrics["l2_error"]["passed"] and metrics["boundary_max"]["passed"]

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=metrics,
        overall_pass=overall_pass,
        timing={"total_simulation": t_sim},
    )
    report.save()
    return overall_pass


def main() -> bool:
    """Entry point for validation runner."""
    return run_validation()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
