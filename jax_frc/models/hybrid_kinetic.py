"""Hybrid kinetic physics model with delta-f PIC ions for 3D Cartesian geometry."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
import jax.random as random
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.models.particle_pusher import (
    boris_push,
    interpolate_field_to_particles_3d,
    deposit_particles_to_grid_3d,
)
from jax_frc.operators import curl_3d, gradient_3d
from jax_frc.core.state import State, ParticleState
from jax_frc.core.geometry import Geometry
from jax_frc.constants import MU0, QE, MI


@dataclass
class RigidRotorEquilibrium:
    """Rigid rotor equilibrium distribution for delta-f method.

    For 3D Cartesian, the rotation is in the xy-plane (about z-axis).
    """
    n0: float = 1e19           # Reference density (m^-3)
    T0: float = 1000.0         # Temperature (eV converted to J internally)
    Omega: float = 1e5         # Rotation frequency (rad/s)

    def f0(self, r, vr, vtheta, vz):
        """Evaluate equilibrium distribution function.

        Args:
            r: Radial distance from z-axis (sqrt(x^2 + y^2))
            vr: Radial velocity component
            vtheta: Azimuthal velocity component
            vz: Axial velocity component
        """
        T_joules = self.T0 * QE  # Convert eV to Joules
        v_sq = vr**2 + (vtheta - self.Omega * r)**2 + vz**2
        thermal_factor = (MI / (2 * jnp.pi * T_joules)) ** 1.5
        return self.n0 * thermal_factor * jnp.exp(-MI * v_sq / (2 * T_joules))

    def d_ln_f0_dt(self, r, vr, vtheta, vz, ar, atheta, az):
        """Compute d(ln f0)/dt for weight evolution.

        d(ln f0)/dt = (1/f0) * df0/dt
                    = (df0/dr * dr/dt + df0/dvr * dvr/dt + ...) / f0
        """
        T_joules = self.T0 * QE
        v_rot = vtheta - self.Omega * r

        # df0/dr contributes via chain rule
        # d/dr[exp(-m*v_rot^2/(2T))] = exp(...) * (-m/(2T)) * 2*v_rot * (-Omega)
        dlnf0_dr = MI * self.Omega * v_rot / T_joules

        # df0/dv terms
        dlnf0_dvr = -MI * vr / T_joules
        dlnf0_dvtheta = -MI * v_rot / T_joules
        dlnf0_dvz = -MI * vz / T_joules

        # Chain rule: d(ln f0)/dt = dlnf0_dr * vr + dlnf0_dv * a
        return (dlnf0_dr * vr +
                dlnf0_dvr * ar +
                dlnf0_dvtheta * atheta +
                dlnf0_dvz * az)


@dataclass
class HybridKinetic(PhysicsModel):
    """Hybrid kinetic model: kinetic ions + fluid electrons in 3D Cartesian.

    Uses delta-f PIC method for ions to reduce statistical noise.
    Electrons are treated as a massless neutralizing fluid.

    Attributes:
        equilibrium: Background distribution for delta-f
        eta: Resistivity (Ohm-m)
        collision_frequency: Ion-ion collision frequency (1/s) for Krook operator.
            Typical FRC values: 1e3-1e5 s^-1. Set to 0 to disable collisions.
    """

    equilibrium: RigidRotorEquilibrium
    eta: float = 1e-6  # Resistivity
    collision_frequency: float = 0.0  # Krook collision frequency (1/s)

    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for hybrid model in 3D Cartesian.

        This computes the RHS for the electromagnetic fields using Faraday's law
        with E from the electron fluid equation (generalized Ohm's law):
            E = (J x B)/(ne) + eta*J - grad(p_e)/(ne)
        """
        B = state.B
        n = jnp.maximum(state.n, 1e16)

        # Current density from curl(B) / mu_0
        J = curl_3d(B, geometry) / MU0

        # Hall term: (J x B) / (ne)
        ne = n * QE
        J_cross_B = jnp.stack([
            J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
            J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
            J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
        ], axis=-1)

        # E = eta*J + (J x B) / (ne)
        E = self.eta * J + J_cross_B / ne[..., None]

        # Pressure gradient term: -grad(p_e) / (ne)
        if state.p is not None:
            grad_p = gradient_3d(state.p, geometry)
            E = E - grad_p / ne[..., None]

        # Faraday's law: dB/dt = -curl(E)
        dB = -curl_3d(E, geometry)

        return state.replace(B=dB, E=E)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Ion cyclotron CFL constraint: dt < 0.1 / omega_ci."""
        B_mag = jnp.sqrt(jnp.sum(state.B**2, axis=-1))
        B_max = jnp.maximum(jnp.max(B_mag), 1e-10)

        # Cyclotron frequency: omega_ci = q*B/m
        omega_ci = QE * B_max / MI

        # Also check particle CFL: dt < dx / v_max
        dx_min = jnp.minimum(jnp.minimum(geometry.dx, geometry.dy), geometry.dz)
        T_joules = self.equilibrium.T0 * QE
        v_thermal = jnp.sqrt(2 * T_joules / MI)
        dt_cfl = dx_min / (3 * v_thermal)  # Factor of 3 for safety

        return jnp.minimum(0.1 / omega_ci, dt_cfl)

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions and particle constraints for 3D.

        For 3D Cartesian with periodic BC in x/y and Dirichlet in z,
        we apply zero-field at z boundaries.
        """
        B = state.B

        # For 3D array: shape is (nx, ny, nz, 3)
        # Apply Dirichlet BC at z boundaries (periodic in x, y by default)
        if geometry.bc_z == "dirichlet":
            B = B.at[:, :, 0, :].set(0)
            B = B.at[:, :, -1, :].set(0)

        # Apply particle boundary conditions if particles exist
        if state.particles is not None:
            particles = self._apply_particle_boundaries(
                state.particles, geometry
            )
            state = state.replace(B=B, particles=particles)
        else:
            state = state.replace(B=B)

        return state

    def push_particles(self, state: State, geometry: Geometry, dt: float) -> State:
        """Advance particles using Boris pusher and update weights."""
        if state.particles is None:
            return state

        particles = state.particles
        x = particles.x
        v = particles.v
        w = particles.w

        # Interpolate E and B fields to particle positions using 3D interpolation
        E_p = interpolate_field_to_particles_3d(state.E, x, geometry)
        B_p = interpolate_field_to_particles_3d(state.B, x, geometry)

        # Boris push
        x_new, v_new = boris_push(x, v, E_p, B_p, QE, MI, dt)

        # Compute acceleration for weight evolution
        a = QE / MI * (E_p + jnp.cross(v, B_p))

        # For weight evolution, convert to cylindrical-like coordinates
        # x, y -> r, theta (for equilibrium that depends on r)
        r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
        r_safe = jnp.where(r > 1e-10, r, 1.0)

        # Compute vr and vtheta from vx, vy
        cos_theta = jnp.where(r > 1e-10, x[:, 0] / r_safe, 1.0)
        sin_theta = jnp.where(r > 1e-10, x[:, 1] / r_safe, 0.0)
        vr = v[:, 0] * cos_theta + v[:, 1] * sin_theta
        vtheta = -v[:, 0] * sin_theta + v[:, 1] * cos_theta

        # Similar for acceleration
        ar = a[:, 0] * cos_theta + a[:, 1] * sin_theta
        atheta = -a[:, 0] * sin_theta + a[:, 1] * cos_theta
        az = a[:, 2]
        vz = v[:, 2]

        dlnf0_dt = self.equilibrium.d_ln_f0_dt(r, vr, vtheta, vz, ar, atheta, az)
        dw = -(1 - w) * dlnf0_dt
        w_new = w + dw * dt

        # Krook collision operator: dw/dt = -nu*w
        # Exact solution: w(t+dt) = w(t) * exp(-nu*dt)
        if self.collision_frequency > 0:
            collision_decay = jnp.exp(-self.collision_frequency * dt)
            w_new = w_new * collision_decay

        w_new = jnp.clip(w_new, -1.0, 1.0)

        # Create new particle state
        new_particles = ParticleState(
            x=x_new,
            v=v_new,
            w=w_new,
            species=particles.species
        )

        return state.replace(particles=new_particles)

    def deposit_current(self, state: State, geometry: Geometry) -> jnp.ndarray:
        """Deposit ion current from particles to 3D grid."""
        if state.particles is None:
            return jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))

        particles = state.particles

        # Current density: J = n * q * v, weighted by delta-f weights
        # For delta-f: J = J_0 + delta_J, where delta_J comes from weights
        J = deposit_particles_to_grid_3d(
            particles.v * QE,  # q*v per particle
            particles.w,       # delta-f weights
            particles.x,
            geometry
        )

        # Normalize by cell volume
        cell_vol = geometry.cell_volumes
        J = J / cell_vol[..., None]

        return J

    def _apply_particle_boundaries(self, particles: ParticleState,
                                   geometry: Geometry) -> ParticleState:
        """Apply periodic boundaries for particles in 3D Cartesian."""
        x = particles.x
        v = particles.v

        # Periodic wrapping in x
        Lx = geometry.x_max - geometry.x_min
        x = x.at[:, 0].set(
            geometry.x_min + jnp.mod(x[:, 0] - geometry.x_min, Lx)
        )

        # Periodic wrapping in y
        Ly = geometry.y_max - geometry.y_min
        x = x.at[:, 1].set(
            geometry.y_min + jnp.mod(x[:, 1] - geometry.y_min, Ly)
        )

        # Periodic or reflecting in z depending on BC
        if geometry.bc_z == "periodic":
            Lz = geometry.z_max - geometry.z_min
            x = x.at[:, 2].set(
                geometry.z_min + jnp.mod(x[:, 2] - geometry.z_min, Lz)
            )
        else:
            # Reflecting at z boundaries
            z = x[:, 2]
            mask_z_min = z < geometry.z_min
            mask_z_max = z > geometry.z_max
            x = x.at[:, 2].set(jnp.where(mask_z_min, 2*geometry.z_min - z, x[:, 2]))
            x = x.at[:, 2].set(jnp.where(mask_z_max, 2*geometry.z_max - z, x[:, 2]))
            v = v.at[:, 2].set(jnp.where(mask_z_min | mask_z_max, -v[:, 2], v[:, 2]))

        return ParticleState(x=x, v=v, w=particles.w, species=particles.species)

    @classmethod
    def from_config(cls, config: dict) -> "HybridKinetic":
        """Create from configuration dictionary."""
        eq_config = config.get("equilibrium", {})
        equilibrium = RigidRotorEquilibrium(
            n0=float(eq_config.get("n0", 1e19)),
            T0=float(eq_config.get("T0", 1000.0)),
            Omega=float(eq_config.get("Omega", 1e5))
        )

        eta = float(config.get("eta", 1e-6))
        collision_frequency = float(config.get("collision_frequency", 0.0))

        return cls(
            equilibrium=equilibrium,
            eta=eta,
            collision_frequency=collision_frequency
        )

    @staticmethod
    def initialize_particles(n_particles: int, geometry: Geometry,
                            equilibrium: RigidRotorEquilibrium,
                            key: jnp.ndarray) -> ParticleState:
        """Initialize particles in 3D Cartesian coordinates.

        Particles are uniformly distributed in the domain with
        Maxwellian velocity distribution plus rigid rotation.

        Args:
            n_particles: Number of particles to create
            geometry: Computational geometry
            equilibrium: Equilibrium distribution parameters
            key: JAX random key

        Returns:
            ParticleState with initialized particles
        """
        keys = random.split(key, 6)

        # Uniform positions in Cartesian coordinates
        x_pos = random.uniform(keys[0], (n_particles,),
                               minval=geometry.x_min, maxval=geometry.x_max)
        y_pos = random.uniform(keys[1], (n_particles,),
                               minval=geometry.y_min, maxval=geometry.y_max)
        z_pos = random.uniform(keys[2], (n_particles,),
                               minval=geometry.z_min, maxval=geometry.z_max)

        # Velocities: Maxwellian + rotation about z-axis
        T_joules = equilibrium.T0 * QE
        v_thermal = jnp.sqrt(T_joules / MI)

        vx_thermal = random.normal(keys[3], (n_particles,)) * v_thermal
        vy_thermal = random.normal(keys[4], (n_particles,)) * v_thermal
        vz = random.normal(keys[5], (n_particles,)) * v_thermal

        # Add rotation: v_rot = Omega x r (rotation about z-axis)
        # v_x += -Omega * y, v_y += Omega * x
        vx = vx_thermal - equilibrium.Omega * y_pos
        vy = vy_thermal + equilibrium.Omega * x_pos

        # Stack into arrays
        x = jnp.stack([x_pos, y_pos, z_pos], axis=-1)
        v = jnp.stack([vx, vy, vz], axis=-1)
        w = jnp.zeros(n_particles)  # Delta-f weights start at zero

        return ParticleState(x=x, v=v, w=w, species="ion")
