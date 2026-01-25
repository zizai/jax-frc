"""Physical constants for plasma physics simulations.

All values are from CODATA 2018 recommended values unless otherwise noted.
SI units are used throughout.
"""

from typing import Final

# Electromagnetic constants
MU0: Final[float] = 1.25663706212e-6  # Permeability of free space [H/m]
EPSILON0: Final[float] = 8.8541878128e-12  # Permittivity of free space [F/m]
C: Final[float] = 299792458.0  # Speed of light [m/s]

# Particle properties
QE: Final[float] = 1.602176634e-19  # Elementary charge [C]
ME: Final[float] = 9.1093837015e-31  # Electron mass [kg]
MI: Final[float] = 1.67262192369e-27  # Proton mass [kg]
MP: Final[float] = MI  # Alias for proton mass

# Thermodynamic constants
KB: Final[float] = 1.380649e-23  # Boltzmann constant [J/K]

# Derived quantities
ELECTRON_CHARGE_MASS_RATIO: Final[float] = QE / ME  # [C/kg]
ION_CHARGE_MASS_RATIO: Final[float] = QE / MI  # [C/kg]

# Mass ratios
MI_ME_RATIO: Final[float] = MI / ME  # Proton-to-electron mass ratio (~1836)


def get_cyclotron_frequency(B: float, q: float = QE, m: float = MI) -> float:
    """Compute cyclotron frequency omega_c = q * B / m.

    Args:
        B: Magnetic field strength [T]
        q: Particle charge [C], defaults to proton charge
        m: Particle mass [kg], defaults to proton mass

    Returns:
        Cyclotron frequency [rad/s]
    """
    return q * B / m


def get_plasma_frequency(n: float, q: float = QE, m: float = MI) -> float:
    """Compute plasma frequency omega_p = sqrt(n * q^2 / (m * epsilon0)).

    Args:
        n: Number density [m^-3]
        q: Particle charge [C], defaults to proton charge
        m: Particle mass [kg], defaults to proton mass

    Returns:
        Plasma frequency [rad/s]
    """
    import jax.numpy as jnp
    return jnp.sqrt(n * q**2 / (m * EPSILON0))


def get_debye_length(n: float, T: float) -> float:
    """Compute Debye length lambda_D = sqrt(epsilon0 * kB * T / (n * q^2)).

    Args:
        n: Number density [m^-3]
        T: Temperature [K]

    Returns:
        Debye length [m]
    """
    import jax.numpy as jnp
    return jnp.sqrt(EPSILON0 * KB * T / (n * QE**2))


def get_skin_depth(n: float, m: float = MI) -> float:
    """Compute ion/electron skin depth d = c / omega_p.

    Args:
        n: Number density [m^-3]
        m: Particle mass [kg], defaults to proton mass

    Returns:
        Skin depth [m]
    """
    import jax.numpy as jnp
    return jnp.sqrt(m / (MU0 * n * QE**2))
