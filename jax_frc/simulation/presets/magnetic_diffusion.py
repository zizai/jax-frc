"""Magnetic diffusion preset simulation."""
import jax.numpy as jnp
from jax_frc.simulation import Simulation, State, Geometry
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers.explicit import RK4Solver


def create_magnetic_diffusion(
    nx: int = 64,
    ny: int = 64,
    nz: int = 1,
    extent: float = 1.0,
    eta: float = 1.26e-10,
    B_peak: float = 1.0,
    sigma: float = 0.1,
) -> Simulation:
    """Create magnetic diffusion test simulation.
    
    Physics: ∂B/∂t = η∇²B (resistive diffusion, no flow)
    
    Args:
        nx: X resolution
        ny: Y resolution
        nz: Z resolution
        extent: Domain extent [-extent, extent]
        eta: Magnetic resistivity [Ω·m]
        B_peak: Peak B_z [T]
        sigma: Initial Gaussian width [m]
    
    Returns:
        Configured Simulation ready to run
    """
    geometry = Geometry(
        nx=nx, ny=ny, nz=nz,
        x_min=-extent, x_max=extent,
        y_min=-extent, y_max=extent,
        z_min=-extent, z_max=extent,
        bc_x="neumann", bc_y="neumann", bc_z="neumann",
    )
    
    model = ExtendedMHD(
        eta=eta,
        include_hall=False,
        include_electron_pressure=False,
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )
    
    solver = RK4Solver()
    
    # Initial Gaussian B_z profile
    x = geometry.x_grid
    y = geometry.y_grid
    r_sq = x**2 + y**2
    B_z = B_peak * jnp.exp(-r_sq / (2 * sigma**2))
    
    B = jnp.zeros((nx, ny, nz, 3))
    B = B.at[:, :, :, 2].set(B_z)
    
    state = State.zeros(nx, ny, nz).replace(B=B)
    
    return Simulation.builder() \
        .geometry(geometry) \
        .model(model) \
        .solver(solver) \
        .initial_state(state) \
        .build()
