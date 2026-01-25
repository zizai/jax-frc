# examples/merging_examples.py
"""Example merging scenarios from Belova et al. paper."""

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.scenarios import Scenario, timeout
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import separation_below, any_of
import jax.numpy as jnp


def create_default_geometry(rc: float = 1.0, zc: float = 4.0,
                           nr: int = 64, nz: int = 256) -> Geometry:
    """Create default cylindrical geometry for merging.

    Args:
        rc: Flux conserver radius
        zc: Half-length of domain
        nr: Radial grid points
        nz: Axial grid points

    Returns:
        Geometry object
    """
    return Geometry(
        coord_system="cylindrical",
        nr=nr,
        nz=nz,
        r_min=0.01 * rc,
        r_max=rc,
        z_min=-zc,
        z_max=zc,
    )


def create_initial_frc(geometry: Geometry,
                       s_star: float = 20.0,
                       elongation: float = 2.0,
                       xs: float = 0.6,
                       beta_s: float = 0.2) -> State:
    """Create initial single-FRC equilibrium.

    Uses simplified Gaussian profile (proper G-S solve in production).

    Args:
        geometry: Computational geometry
        s_star: Kinetic parameter Rs/di
        elongation: E = Zs/Rs
        xs: Normalized separatrix radius Rs/Rc
        beta_s: Separatrix beta

    Returns:
        Single FRC equilibrium state
    """
    state = State.zeros(nr=geometry.nr, nz=geometry.nz)

    r = geometry.r_grid
    z = geometry.z_grid

    # Compute FRC dimensions from parameters
    Rc = geometry.r_max
    Rs = xs * Rc
    Zs = elongation * Rs

    # Create Gaussian FRC profile
    psi = jnp.exp(-((r - 0.5*Rs)**2 / (0.3*Rs)**2 + z**2 / Zs**2))

    # Pressure proportional to psi, with separatrix beta
    p = beta_s * psi

    # Uniform density
    n = jnp.ones_like(psi)

    return state.replace(psi=psi, p=p, n=n)


def belova_case1() -> Scenario:
    """Large FRC merging without compression (paper Fig. 1-2).

    Parameters:
        S* = 25.6, E = 2.9, xs = 0.69, beta_s = 0.2
        Initial separation: dZ = 180 (normalized)
        Initial velocity: Vz = 0.2 vA

    Expected outcome: Partial merge, doublet configuration
    """
    geometry = create_default_geometry(rc=1.0, zc=5.0, nr=64, nz=512)
    initial_state = create_initial_frc(
        geometry,
        s_star=25.6,
        elongation=2.9,
        xs=0.69,
        beta_s=0.2
    )

    # Merge phase with velocity drive only (no compression)
    merge_phase = MergingPhase(
        name="merge_no_compression",
        transition=any_of(
            separation_below(0.5, geometry),  # Complete merge
            timeout(30.0)  # tA
        ),
        separation=3.0,  # Normalized units
        initial_velocity=0.2,
        compression=None,
    )

    return Scenario(
        name="belova_case1_large_frc",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        dt=0.01,
    )


def belova_case2() -> Scenario:
    """Small FRC merging without compression (paper Fig. 3-4).

    Parameters:
        S* = 20, E = 1.5, xs = 0.53, beta_s = 0.2
        Initial separation: dZ = 75 (normalized)
        Initial velocity: Vz = 0.1 vA

    Expected outcome: Complete merge by ~5-7 tA
    """
    geometry = create_default_geometry(rc=1.0, zc=3.0, nr=64, nz=256)
    initial_state = create_initial_frc(
        geometry,
        s_star=20.0,
        elongation=1.5,
        xs=0.53,
        beta_s=0.2
    )

    merge_phase = MergingPhase(
        name="merge_small_frc",
        transition=any_of(
            separation_below(0.3, geometry),
            timeout(15.0)
        ),
        separation=1.5,
        initial_velocity=0.1,
        compression=None,
    )

    return Scenario(
        name="belova_case2_small_frc",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        dt=0.01,
    )


def belova_case4() -> Scenario:
    """Large FRC with compression (paper Fig. 6-7).

    Parameters: Same as case1 but with compression
        Mirror ratio: 1.5
        Ramp time: 19 tA

    Expected outcome: Complete merge by ~20-25 tA
    """
    geometry = create_default_geometry(rc=1.0, zc=5.0, nr=64, nz=512)
    initial_state = create_initial_frc(
        geometry,
        s_star=25.6,
        elongation=2.9,
        xs=0.69,
        beta_s=0.2
    )

    merge_phase = MergingPhase(
        name="merge_with_compression",
        transition=any_of(
            separation_below(0.3, geometry),
            timeout(40.0)
        ),
        separation=3.0,
        initial_velocity=0.0,  # Compression drives merging
        compression={
            "base_field": 1.0,
            "mirror_ratio": 1.5,
            "ramp_time": 19.0,
            "profile": "cosine",
        },
    )

    return Scenario(
        name="belova_case4_compression",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        dt=0.01,
    )


if __name__ == "__main__":
    # Run case 2 as a quick test
    print("Creating Belova Case 2 scenario...")
    scenario = belova_case2()

    print(f"Geometry: {scenario.geometry.nr}x{scenario.geometry.nz}")
    print(f"Phases: {[p.name for p in scenario.phases]}")
    print(f"dt: {scenario.dt}")

    print("\nRunning scenario...")
    result = scenario.run()

    print(f"\nResult: {result.success}")
    for pr in result.phase_results:
        print(f"  {pr.name}: {pr.termination} at t={pr.end_time:.2f}")
