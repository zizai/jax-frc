from jax_frc.core.geometry import Geometry


def make_geometry(nx=8, ny=1, nz=8, extent=1.0):
    return Geometry(
        nx=nx,
        ny=ny,
        nz=nz,
        x_min=-extent,
        x_max=extent,
        y_min=-extent,
        y_max=extent,
        z_min=-extent,
        z_max=extent,
        bc_x="neumann",
        bc_y="periodic",
        bc_z="neumann",
    )
