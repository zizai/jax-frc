# examples/merging_examples.py
"""Example merging scenarios from Belova et al. paper.

This module provides factory functions for creating FRC merging configurations
using the validation/configuration infrastructure. The configurations are based
on simulations from Belova et al.

Usage:
    # Using configuration classes directly (preferred)
    from jax_frc.configurations import BelovaCase2Configuration

    config = BelovaCase2Configuration()
    result = config.run()

    # Using factory functions (backward-compatible API)
    from examples.merging_examples import belova_case2

    config = belova_case2()
    result = config.run()
"""

from typing import Optional, Dict, Any

from jax_frc.configurations.frc_merging import (
    BelovaMergingConfiguration,
    BelovaCase1Configuration,
    BelovaCase2Configuration,
    BelovaCase4Configuration,
)

# Re-export configuration classes for backward compatibility
__all__ = [
    "BelovaMergingConfiguration",
    "BelovaCase1Configuration",
    "BelovaCase2Configuration",
    "BelovaCase4Configuration",
    "belova_case1",
    "belova_case2",
    "belova_case3",
    "belova_case4",
    "create_custom_merging",
]


def belova_case1(model_type: str = "resistive_mhd") -> BelovaCase1Configuration:
    """Large FRC merging without compression (paper Fig. 1-2).

    Parameters:
        S* = 25.6, E = 2.9, xs = 0.69, beta_s = 0.2
        Initial separation: dZ = 180 (normalized)
        Initial velocity: Vz = 0.2 vA

    Expected outcome: Partial merge, doublet configuration

    Args:
        model_type: "resistive_mhd", "extended_mhd", or "hybrid_kinetic"

    Returns:
        BelovaCase1Configuration ready to run
    """
    return BelovaCase1Configuration(model_type=model_type)


def belova_case2(model_type: str = "resistive_mhd") -> BelovaCase2Configuration:
    """Small FRC merging without compression (paper Fig. 3-4).

    Parameters:
        S* = 20, E = 1.5, xs = 0.53, beta_s = 0.2
        Initial separation: dZ = 75 (normalized)
        Initial velocity: Vz = 0.1 vA

    Expected outcome: Complete merge by ~5-7 tA

    Args:
        model_type: "resistive_mhd", "extended_mhd", or "hybrid_kinetic"

    Returns:
        BelovaCase2Configuration ready to run
    """
    return BelovaCase2Configuration(model_type=model_type)


def belova_case3(
    separation: float = 1.5, model_type: str = "resistive_mhd"
) -> BelovaMergingConfiguration:
    """Small FRC with variable separation (paper Section 2.3).

    Args:
        separation: Initial separation (1.5~dZ=75, 2.2~dZ=110, 2.5~dZ=125, 3.7~dZ=185)
        model_type: "resistive_mhd", "extended_mhd", or "hybrid_kinetic"

    Returns:
        BelovaMergingConfiguration with custom separation
    """
    # Adjust domain size based on separation
    domain_half_length = max(3.0, separation * 1.5)
    timeout = 50.0 if separation > 2.5 else 25.0

    return BelovaMergingConfiguration(
        name=f"belova_case3_sep{separation}",
        description=f"Small FRC with separation {separation}",
        # Geometry
        flux_conserver_radius=1.0,
        domain_half_length=domain_half_length,
        nr=64,
        nz=256,
        # FRC parameters (same as case 2)
        s_star=20.0,
        elongation=1.5,
        xs=0.53,
        beta_s=0.2,
        # Merging parameters
        separation=separation,
        initial_velocity=0.1,
        compression=None,
        # Model
        model_type=model_type,
        # Runtime
        timeout=timeout,
        separation_threshold=0.3,
    )


def belova_case4(model_type: str = "resistive_mhd") -> BelovaCase4Configuration:
    """Large FRC with compression (paper Fig. 6-7).

    Parameters: Same as case1 but with compression
        Mirror ratio: 1.5
        Ramp time: 19 tA

    Expected outcome: Complete merge by ~20-25 tA

    Args:
        model_type: "resistive_mhd", "extended_mhd", or "hybrid_kinetic"

    Returns:
        BelovaCase4Configuration ready to run
    """
    return BelovaCase4Configuration(model_type=model_type)


def create_custom_merging(
    s_star: float = 20.0,
    elongation: float = 2.0,
    xs: float = 0.6,
    beta_s: float = 0.2,
    separation: float = 1.5,
    initial_velocity: float = 0.1,
    compression: Optional[Dict[str, Any]] = None,
    model_type: str = "resistive_mhd",
    nr: int = 64,
    nz: int = 256,
    flux_conserver_radius: float = 1.0,
    domain_half_length: float = 4.0,
    timeout: float = 15.0,
    dt: float = 0.001,
) -> BelovaMergingConfiguration:
    """Create a custom FRC merging configuration.

    This function allows full customization of all FRC and merging parameters
    for parameter scans and studies.

    Args:
        s_star: Kinetic parameter S* = Rs/di
        elongation: FRC elongation E = Zs/Rs
        xs: Normalized separatrix radius Rs/Rc
        beta_s: Separatrix beta
        separation: Initial separation between FRC nulls
        initial_velocity: Initial axial velocity (vA units)
        compression: Optional compression BC config dict
        model_type: Physics model type
        nr: Number of radial grid points
        nz: Number of axial grid points
        flux_conserver_radius: Flux conserver radius
        domain_half_length: Half-length of domain
        timeout: Maximum simulation time
        dt: Timestep

    Returns:
        BelovaMergingConfiguration with custom parameters
    """
    return BelovaMergingConfiguration(
        name="custom_merging",
        description="Custom FRC merging configuration",
        flux_conserver_radius=flux_conserver_radius,
        domain_half_length=domain_half_length,
        nr=nr,
        nz=nz,
        s_star=s_star,
        elongation=elongation,
        xs=xs,
        beta_s=beta_s,
        separation=separation,
        initial_velocity=initial_velocity,
        compression=compression,
        model_type=model_type,
        dt=dt,
        timeout=timeout,
    )


if __name__ == "__main__":
    # Run case 2 as a quick test
    print("Creating Belova Case 2 configuration...")
    config = belova_case2()

    geometry = config.build_geometry()
    print(f"Geometry: {geometry.nr}x{geometry.nz}")
    print(f"Phases: {config.available_phases()}")
    print(f"dt: {config.dt}")

    print("\nRunning configuration...")
    result = config.run()

    print(f"\nResult: {result.success}")
    for pr in result.phase_results:
        print(f"  {pr.name}: {pr.termination} at t={pr.end_time:.2f}")
