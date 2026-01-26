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


# =============================================================================
# Nuclear Reaction Data
# =============================================================================

# Particle masses for fusion products
M_DEUTERIUM: Final[float] = 2.014102 * 1.66053906660e-27  # [kg]
M_TRITIUM: Final[float] = 3.016049 * 1.66053906660e-27    # [kg]
M_HELIUM3: Final[float] = 3.016029 * 1.66053906660e-27    # [kg]
M_HELIUM4: Final[float] = 4.002603 * 1.66053906660e-27    # [kg]
M_PROTON: Final[float] = 1.007276 * 1.66053906660e-27     # [kg]
M_NEUTRON: Final[float] = 1.008665 * 1.66053906660e-27    # [kg]

# Fusion reaction energies [J]
E_DT: Final[float] = 17.6e6 * QE      # D + T -> He4 + n
E_DD_T: Final[float] = 4.03e6 * QE    # D + D -> T + p
E_DD_HE3: Final[float] = 3.27e6 * QE  # D + D -> He3 + n
E_DHE3: Final[float] = 18.3e6 * QE    # D + He3 -> He4 + p

# Charged particle energy fractions
F_CHARGED_DT: Final[float] = 3.5 / 17.6       # Alpha only
F_CHARGED_DD_T: Final[float] = 1.0            # T + p both charged
F_CHARGED_DD_HE3: Final[float] = 0.82 / 3.27  # He3 only
F_CHARGED_DHE3: Final[float] = 1.0            # He4 + p both charged

# Bosch-Hale parameterization coefficients (NF 1992)
# <sigma*v> = C1 * theta * sqrt(xi / (m_rc2 * T^3)) * exp(-3*xi)
# where theta = T / (1 - (T*(C2 + T*(C4 + T*C6))) / (1 + T*(C3 + T*(C5 + T*C7))))
#       xi = (B_G^2 / (4*theta))^(1/3)
# T in keV, <sigma*v> in cm^3/s

BOSCH_HALE_DT: Final[dict] = {
    "B_G": 34.3827,  # Gamow constant [keV^0.5]
    "m_rc2": 1124656,  # Reduced mass * c^2 [keV]
    "C1": 1.17302e-9,
    "C2": 1.51361e-2,
    "C3": 7.51886e-2,
    "C4": 4.60643e-3,
    "C5": 1.35000e-2,
    "C6": -1.06750e-4,
    "C7": 1.36600e-5,
}

BOSCH_HALE_DD_T: Final[dict] = {
    "B_G": 31.3970,
    "m_rc2": 937814,
    "C1": 5.65718e-12,
    "C2": 3.41267e-3,
    "C3": 1.99167e-3,
    "C4": 0.0,
    "C5": 1.05060e-5,
    "C6": 0.0,
    "C7": 0.0,
}

BOSCH_HALE_DD_HE3: Final[dict] = {
    "B_G": 31.3970,
    "m_rc2": 937814,
    "C1": 5.43360e-12,
    "C2": 5.85778e-3,
    "C3": 7.68222e-3,
    "C4": 0.0,
    "C5": -2.96400e-6,
    "C6": 0.0,
    "C7": 0.0,
}

BOSCH_HALE_DHE3: Final[dict] = {
    "B_G": 68.7508,
    "m_rc2": 1124572,
    "C1": 5.51036e-10,
    "C2": 6.41918e-3,
    "C3": -2.02896e-3,
    "C4": -1.91080e-5,
    "C5": 1.35776e-4,
    "C6": 0.0,
    "C7": 0.0,
}
