"""Thermal transport and energy equation for Extended MHD.

Implements anisotropic thermal conduction with parallel and perpendicular
conductivities relative to the magnetic field direction.
"""

from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ThermalTransport:
    """Anisotropic thermal transport model for magnetized plasma.

    Computes heat flux: q = -κ_∥ (b·∇T) b - κ_⊥ (∇T - (b·∇T) b)

    Where:
        - b = B/|B| is the unit vector along the magnetic field
        - κ_∥ is the parallel thermal conductivity
        - κ_⊥ is the perpendicular thermal conductivity

    For Spitzer conductivity: κ_∥ = κ_0 T^(5/2) / ln(Λ)

    Attributes:
        kappa_parallel_0: Base parallel conductivity [W/(m·eV^(7/2))]
        kappa_perp_ratio: Ratio κ_⊥/κ_∥ (typically 1e-6 for FRCs)
        use_spitzer: If True, κ_∥ = κ_0 T^(5/2) / ln(Λ)
        coulomb_log: Coulomb logarithm (default 15)
        min_temperature: Minimum temperature to avoid singularities [eV]
    """
    kappa_parallel_0: float = 3.16e20
    kappa_perp_ratio: float = 1e-6
    use_spitzer: bool = True
    coulomb_log: float = 15.0
    min_temperature: float = 1e-3

    def compute_kappa_parallel(self, T: Array) -> Array:
        """Compute parallel thermal conductivity κ_∥.

        For Spitzer: κ_∥ = κ_0 T^(5/2) / ln(Λ)
        For constant: κ_∥ = κ_0

        Args:
            T: Temperature field [eV] with shape (nr, nz)

        Returns:
            Parallel thermal conductivity with same shape as T
        """
        if self.use_spitzer:
            # Spitzer: κ_∥ = κ_0 T^(5/2) / ln(Λ)
            T_safe = jnp.maximum(T, self.min_temperature)
            return self.kappa_parallel_0 * T_safe**2.5 / self.coulomb_log
        return jnp.full_like(T, self.kappa_parallel_0)

    def compute_kappa_perp(self, T: Array) -> Array:
        """Compute perpendicular thermal conductivity κ_⊥.

        κ_⊥ = κ_∥ * kappa_perp_ratio

        Args:
            T: Temperature field [eV] with shape (nr, nz)

        Returns:
            Perpendicular thermal conductivity with same shape as T
        """
        return self.compute_kappa_parallel(T) * self.kappa_perp_ratio

    def compute_heat_flux(self, T: Array, B: Array, dr: float, dz: float,
                          r: Array) -> tuple[Array, Array]:
        """Compute anisotropic heat flux components (q_r, q_z).

        The heat flux in magnetized plasma is:
            q = -κ_∥ (b·∇T) b - κ_⊥ (∇T - (b·∇T) b)

        Where:
            - b = B/|B| is the unit vector along B
            - (b·∇T) is the parallel gradient component
            - (∇T - (b·∇T) b) is the perpendicular gradient

        Args:
            T: Temperature field [eV] with shape (nr, nz)
            B: Magnetic field with shape (nr, nz, 3) for (B_r, B_phi, B_z)
            dr: Radial grid spacing [m]
            dz: Axial grid spacing [m]
            r: Radial coordinate grid with shape (nr, nz)

        Returns:
            Tuple (q_r, q_z) of heat flux components, each shape (nr, nz)
        """
        # Compute temperature gradients using central differences
        dT_dr = (jnp.roll(T, -1, axis=0) - jnp.roll(T, 1, axis=0)) / (2 * dr)
        dT_dz = (jnp.roll(T, -1, axis=1) - jnp.roll(T, 1, axis=1)) / (2 * dz)

        # Magnetic field components and magnitude
        B_r = B[:, :, 0]
        B_phi = B[:, :, 1]
        B_z = B[:, :, 2]
        B_mag = jnp.sqrt(B_r**2 + B_phi**2 + B_z**2)
        B_mag_safe = jnp.maximum(B_mag, 1e-10)

        # Unit vector along B (only r and z components for 2D heat flux)
        b_r = B_r / B_mag_safe
        b_z = B_z / B_mag_safe

        # Parallel gradient: b·∇T (in r-z plane, ignoring phi)
        grad_T_parallel = b_r * dT_dr + b_z * dT_dz

        # Perpendicular gradient: ∇T - (b·∇T) b
        grad_T_perp_r = dT_dr - grad_T_parallel * b_r
        grad_T_perp_z = dT_dz - grad_T_parallel * b_z

        # Conductivities
        kappa_par = self.compute_kappa_parallel(T)
        kappa_perp = self.compute_kappa_perp(T)

        # Heat flux: q = -κ_∥ (b·∇T) b - κ_⊥ (∇_⊥T)
        # Parallel contribution: -κ_∥ (b·∇T) b
        q_par_r = -kappa_par * grad_T_parallel * b_r
        q_par_z = -kappa_par * grad_T_parallel * b_z

        # Perpendicular contribution: -κ_⊥ (∇T - (b·∇T) b)
        q_perp_r = -kappa_perp * grad_T_perp_r
        q_perp_z = -kappa_perp * grad_T_perp_z

        # Total heat flux
        q_r = q_par_r + q_perp_r
        q_z = q_par_z + q_perp_z

        return q_r, q_z

    def compute_heat_flux_divergence(self, T: Array, B: Array, dr: float,
                                      dz: float, r: Array) -> Array:
        """Compute divergence of heat flux ∇·q in cylindrical coordinates.

        In cylindrical coords: ∇·q = (1/r) d(r q_r)/dr + dq_z/dz

        Args:
            T: Temperature field [eV] with shape (nr, nz)
            B: Magnetic field with shape (nr, nz, 3)
            dr: Radial grid spacing [m]
            dz: Axial grid spacing [m]
            r: Radial coordinate grid with shape (nr, nz)

        Returns:
            Divergence of heat flux ∇·q with shape (nr, nz)
        """
        q_r, q_z = self.compute_heat_flux(T, B, dr, dz, r)

        # Cylindrical divergence: (1/r) d(r*q_r)/dr + dq_z/dz
        rq_r = r * q_r
        d_rq_r_dr = (jnp.roll(rq_r, -1, axis=0) - jnp.roll(rq_r, 1, axis=0)) / (2 * dr)
        dq_z_dz = (jnp.roll(q_z, -1, axis=1) - jnp.roll(q_z, 1, axis=1)) / (2 * dz)

        # Handle r=0 singularity with L'Hopital's rule
        # lim(r→0) (1/r) d(r*q_r)/dr = 2 * dq_r/dr
        dq_r_dr = (jnp.roll(q_r, -1, axis=0) - jnp.roll(q_r, 1, axis=0)) / (2 * dr)
        r_safe = jnp.where(r > 1e-10, r, 1.0)

        div_q = jnp.where(
            r > 1e-10,
            d_rq_r_dr / r_safe + dq_z_dz,
            2.0 * dq_r_dr + dq_z_dz  # L'Hopital at r=0
        )

        return div_q
