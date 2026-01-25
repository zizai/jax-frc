"""Hybrid kinetic physics model with delta-f PIC ions."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
import jax.random as random
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.models.particle_pusher import boris_push, interpolate_field_to_particles, deposit_particles_to_grid
from jax_frc.core.state import State, ParticleState
from jax_frc.core.geometry import Geometry

MU0 = 1.2566e-6
QE = 1.602e-19
MI = 1.673e-27
KB = 1.381e-23


@dataclass
class RigidRotorEquilibrium:
    """Rigid rotor equilibrium distribution for delta-f method."""
    n0: float = 1e19           # Reference density (m^-3)
    T0: float = 1000.0         # Temperature (eV converted to J internally)
    Omega: float = 1e5         # Rotation frequency (rad/s)

    def f0(self, r, vr, vtheta, vz):
        """Evaluate equilibrium distribution function."""
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
    """Hybrid kinetic model: kinetic ions + fluid electrons.

    Uses delta-f PIC method for ions to reduce statistical noise.
    Electrons are treated as a massless neutralizing fluid.
    """

    equilibrium: RigidRotorEquilibrium
    eta: float = 1e-6  # Resistivity

    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for hybrid model.

        This computes the RHS for the electromagnetic fields using Faraday's law
        with E from the electron fluid equation (generalized Ohm's law):
            E = (J × B)/(ne) + ηJ - ∇p_e/(ne)
        """
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute current density from curl(B) / mu_0 for field evolution
        B_r = state.B[:, :, 0]
        B_phi = state.B[:, :, 1]
        B_z = state.B[:, :, 2]
        J_r, J_phi, J_z = self._compute_current(B_r, B_phi, B_z, dr, dz, r)

        # Safe density to avoid division by zero
        n = jnp.maximum(state.n, 1e16)
        ne = n * QE

        # Hall term: (J × B) / (ne)
        hall_r = (J_phi * B_z - J_z * B_phi) / ne
        hall_phi = (J_z * B_r - J_r * B_z) / ne
        hall_z = (J_r * B_phi - J_phi * B_r) / ne

        # Electron pressure gradient term: -∇p_e/(ne)
        dp_dr = (jnp.roll(state.p, -1, axis=0) - jnp.roll(state.p, 1, axis=0)) / (2*dr)
        dp_dz = (jnp.roll(state.p, -1, axis=1) - jnp.roll(state.p, 1, axis=1)) / (2*dz)

        # Combined E-field: E = Hall + η*J - ∇p_e/(ne)
        E_r = hall_r + self.eta * J_r - dp_dr / ne
        E_phi = hall_phi + self.eta * J_phi
        E_z = hall_z + self.eta * J_z - dp_dz / ne

        E = jnp.stack([E_r, E_phi, E_z], axis=-1)

        # Faraday's law: dB/dt = -curl(E)
        dB_r, dB_phi, dB_z = self._compute_curl_E(E_r, E_phi, E_z, dr, dz)
        dB = jnp.stack([-dB_r, -dB_phi, -dB_z], axis=-1)

        # Store E in state for particle pushing
        return state.replace(B=dB, E=E)

    def _compute_current(self, B_r, B_phi, B_z, dr, dz, r):
        """Compute J = curl(B) / mu_0 in cylindrical coordinates.

        Curl in cylindrical (r, theta, z) with axisymmetry (d/dtheta = 0):
            J_r = -(1/mu_0) * dB_phi/dz
            J_phi = (1/mu_0) * (dB_r/dz - dB_z/dr)
            J_z = (1/mu_0) * (1/r) * d(r*B_phi)/dr
                = (1/mu_0) * (B_phi/r + dB_phi/dr)

        At r=0, uses L'Hopital's rule:
            J_z[0,:] = (1/mu_0) * 2 * dB_phi/dr[0,:]
        """
        # J_r = -(1/mu_0) * dB_phi/dz
        dB_phi_dz = (jnp.roll(B_phi, -1, axis=1) - jnp.roll(B_phi, 1, axis=1)) / (2 * dz)
        J_r = -(1.0 / MU0) * dB_phi_dz

        # J_phi = (1/mu_0) * (dB_r/dz - dB_z/dr)
        dB_r_dz = (jnp.roll(B_r, -1, axis=1) - jnp.roll(B_r, 1, axis=1)) / (2 * dz)
        dB_z_dr = (jnp.roll(B_z, -1, axis=0) - jnp.roll(B_z, 1, axis=0)) / (2 * dr)
        J_phi = (1.0 / MU0) * (dB_r_dz - dB_z_dr)

        # J_z = (1/mu_0) * (B_phi/r + dB_phi/dr)
        dB_phi_dr = (jnp.roll(B_phi, -1, axis=0) - jnp.roll(B_phi, 1, axis=0)) / (2 * dr)

        # Handle r=0 singularity
        r_safe = jnp.where(r > 1e-10, r, 1.0)
        J_z = (1.0 / MU0) * jnp.where(
            r > 1e-10,
            B_phi / r_safe + dB_phi_dr,
            2.0 * dB_phi_dr  # L'Hopital at r=0
        )

        return J_r, J_phi, J_z

    def _compute_curl_E(self, E_r, E_phi, E_z, dr, dz):
        """Compute curl(E) for Faraday's law in cylindrical coordinates."""
        # curl_r = (1/r)*dE_z/dphi - dE_phi/dz ~ -dE_phi/dz (axisymmetric)
        dE_phi_dz = (jnp.roll(E_phi, -1, axis=1) - jnp.roll(E_phi, 1, axis=1)) / (2*dz)
        curl_r = -dE_phi_dz

        # curl_phi = dE_r/dz - dE_z/dr
        dE_r_dz = (jnp.roll(E_r, -1, axis=1) - jnp.roll(E_r, 1, axis=1)) / (2*dz)
        dE_z_dr = (jnp.roll(E_z, -1, axis=0) - jnp.roll(E_z, 1, axis=0)) / (2*dr)
        curl_phi = dE_r_dz - dE_z_dr

        # curl_z = (1/r)*d(r*E_phi)/dr ~ dE_phi/dr (simplified)
        dE_phi_dr = (jnp.roll(E_phi, -1, axis=0) - jnp.roll(E_phi, 1, axis=0)) / (2*dr)
        curl_z = dE_phi_dr

        return curl_r, curl_phi, curl_z

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Ion cyclotron CFL constraint: dt < 0.1 / omega_ci."""
        B_mag = jnp.sqrt(jnp.sum(state.B**2, axis=-1))
        B_max = jnp.maximum(jnp.max(B_mag), 1e-10)

        # Cyclotron frequency: omega_ci = q*B/m
        omega_ci = QE * B_max / MI

        # Also check particle CFL: dt < dx / v_max
        dx_min = jnp.minimum(geometry.dr, geometry.dz)
        T_joules = self.equilibrium.T0 * QE
        v_thermal = jnp.sqrt(2 * T_joules / MI)
        dt_cfl = dx_min / (3 * v_thermal)  # Factor of 3 for safety

        return jnp.minimum(0.1 / omega_ci, dt_cfl)

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions and particle constraints."""
        # Apply field boundary conditions
        B = state.B
        B = B.at[0, :, :].set(B[1, :, :])  # Neumann at r=0
        B = B.at[-1, :, :].set(0)  # Dirichlet at r_max
        B = B.at[:, 0, :].set(0)   # Dirichlet at z_min
        B = B.at[:, -1, :].set(0)  # Dirichlet at z_max

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

        # Geometry parameters for interpolation
        geom_params = (geometry.r_min, geometry.r_max,
                       geometry.z_min, geometry.z_max,
                       geometry.nr, geometry.nz)

        # Interpolate E and B fields to particle positions
        E_p = interpolate_field_to_particles(state.E, x, geom_params)
        B_p = interpolate_field_to_particles(state.B, x, geom_params)

        # Boris push
        x_new, v_new = boris_push(x, v, E_p, B_p, QE, MI, dt)

        # Compute acceleration for weight evolution
        a = QE / MI * (E_p + jnp.cross(v, B_p))

        # Weight evolution: dw/dt = -(1-w) * d(ln f0)/dt
        r = x[:, 0]
        vr, vtheta, vz = v[:, 0], v[:, 1], v[:, 2]
        ar, atheta, az = a[:, 0], a[:, 1], a[:, 2]

        dlnf0_dt = self.equilibrium.d_ln_f0_dt(r, vr, vtheta, vz, ar, atheta, az)
        dw = -(1 - w) * dlnf0_dt
        w_new = jnp.clip(w + dw * dt, -1.0, 1.0)

        # Create new particle state
        new_particles = ParticleState(
            x=x_new,
            v=v_new,
            w=w_new,
            species=particles.species
        )

        return state.replace(particles=new_particles)

    def deposit_current(self, state: State, geometry: Geometry) -> jnp.ndarray:
        """Deposit ion current from particles to grid."""
        if state.particles is None:
            return jnp.zeros((geometry.nr, geometry.nz, 3))

        particles = state.particles
        geom_params = (geometry.r_min, geometry.r_max,
                       geometry.z_min, geometry.z_max,
                       geometry.nr, geometry.nz)

        # Current density: J = n * q * v, weighted by delta-f weights
        # For delta-f: J = J_0 + delta_J, where delta_J comes from weights
        J = deposit_particles_to_grid(
            particles.v * QE,  # q*v per particle
            particles.w,       # delta-f weights
            particles.x,
            geom_params
        )

        # Normalize by cell volume
        cell_vol = geometry.cell_volumes
        J = J / cell_vol[:, :, None]

        return J

    def _apply_particle_boundaries(self, particles: ParticleState,
                                   geometry: Geometry) -> ParticleState:
        """Apply reflecting boundaries for particles."""
        x = particles.x
        v = particles.v

        # Reflect at r boundaries
        r = x[:, 0]
        mask_r_min = r < geometry.r_min
        mask_r_max = r > geometry.r_max
        x = x.at[:, 0].set(jnp.where(mask_r_min, 2*geometry.r_min - r, x[:, 0]))
        x = x.at[:, 0].set(jnp.where(mask_r_max, 2*geometry.r_max - r, x[:, 0]))
        v = v.at[:, 0].set(jnp.where(mask_r_min | mask_r_max, -v[:, 0], v[:, 0]))

        # Reflect at z boundaries
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

        return cls(equilibrium=equilibrium, eta=eta)

    @staticmethod
    def initialize_particles(n_particles: int, geometry: Geometry,
                            equilibrium: RigidRotorEquilibrium,
                            key: jnp.ndarray) -> ParticleState:
        """Initialize particles from equilibrium distribution.

        Args:
            n_particles: Number of particles to create
            geometry: Computational geometry
            equilibrium: Equilibrium distribution parameters
            key: JAX random key

        Returns:
            ParticleState with initialized particles
        """
        keys = random.split(key, 5)

        # Positions: uniform in r^2 (for uniform density in cylindrical)
        r_sq = random.uniform(keys[0], (n_particles,),
                              minval=geometry.r_min**2,
                              maxval=geometry.r_max**2)
        r = jnp.sqrt(r_sq)
        theta = random.uniform(keys[1], (n_particles,),
                               minval=0, maxval=2*jnp.pi)
        z = random.uniform(keys[2], (n_particles,),
                          minval=geometry.z_min, maxval=geometry.z_max)

        # Velocities: Maxwellian + rotation
        T_joules = equilibrium.T0 * QE
        v_thermal = jnp.sqrt(T_joules / MI)

        vr = random.normal(keys[3], (n_particles,)) * v_thermal
        vz = random.normal(keys[4], (n_particles,)) * v_thermal
        vtheta = random.normal(random.fold_in(keys[0], 1), (n_particles,)) * v_thermal
        vtheta = vtheta + equilibrium.Omega * r  # Add rotation

        # Position in Cartesian-like (r, theta, z)
        x = jnp.stack([r, theta, z], axis=-1)
        v = jnp.stack([vr, vtheta, vz], axis=-1)
        w = jnp.zeros(n_particles)  # Delta-f weights start at zero

        return ParticleState(x=x, v=v, w=w, species="ion")
