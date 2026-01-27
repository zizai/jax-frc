# tests/test_frc_merging_configuration.py
"""Tests for FRC merging configuration classes."""

import pytest
import jax.numpy as jnp

from jax_frc.configurations import CONFIGURATION_REGISTRY
from jax_frc.configurations.frc_merging import (
    BelovaMergingConfiguration,
    BelovaCase1Configuration,
    BelovaCase2Configuration,
    BelovaCase4Configuration,
)


class TestBelovaMergingConfiguration:
    """Tests for the base BelovaMergingConfiguration class."""

    def test_builds_valid_geometry(self):
        """Geometry should be cylindrical with r_min > 0."""
        config = BelovaMergingConfiguration()
        geometry = config.build_geometry()

        assert geometry.x_max > geometry.x_min
        assert geometry.z_min < 0 < geometry.z_max
        assert geometry.nx == config.nr
        assert geometry.nz == config.nz

    def test_builds_single_frc_state(self):
        """Initial state should be a single FRC symmetric about z=0."""
        config = BelovaMergingConfiguration()
        geometry = config.build_geometry()
        state = config.build_initial_state(geometry)

        # State should have proper shapes
        assert state.B.shape == (geometry.nx, geometry.ny, geometry.nz, 3)
        assert state.p.shape == (geometry.nx, geometry.ny, geometry.nz)
        assert state.n.shape == (geometry.nx, geometry.ny, geometry.nz)

        # Peak Bz should be positive (FRC has trapped flux proxy)
        assert jnp.max(state.B[:, :, :, 2]) > 0.5

        # Profile should be roughly symmetric about z=0
        mid_z = geometry.nz // 2
        left_bz = state.B[:, 0, mid_z - 10:mid_z, 2]
        right_bz = state.B[:, 0, mid_z:mid_z + 10, 2]
        right_bz_flipped = jnp.flip(right_bz, axis=1)
        symmetry_error = jnp.mean(jnp.abs(left_bz - right_bz_flipped))
        assert symmetry_error < 0.1, f"Symmetry error {symmetry_error} too large"

    def test_builds_resistive_mhd_model(self):
        """Model should be created based on model_type parameter."""
        config = BelovaMergingConfiguration(model_type="resistive_mhd")
        model = config.build_model()

        # Model should have required interface methods
        assert hasattr(model, 'compute_rhs')
        assert hasattr(model, 'compute_stable_dt')
        assert hasattr(model, 'apply_constraints')

    def test_builds_boundary_conditions(self):
        """Boundary conditions should be a list (can be empty)."""
        config = BelovaMergingConfiguration()
        bcs = config.build_boundary_conditions()

        assert isinstance(bcs, list)

    def test_available_phases_includes_merging(self):
        """Configuration should list 'merging' as an available phase."""
        config = BelovaMergingConfiguration()
        phases = config.available_phases()

        assert "merging" in phases

    def test_merging_phase_config_provides_parameters(self):
        """merging_phase_config() should return dict with required keys."""
        config = BelovaMergingConfiguration(
            separation=2.0,
            initial_velocity=0.15,
            compression={"mirror_ratio": 1.5},
        )
        phase_config = config.merging_phase_config()

        assert "separation" in phase_config
        assert "initial_velocity" in phase_config
        assert "compression" in phase_config

        assert phase_config["separation"] == 2.0
        assert phase_config["initial_velocity"] == 0.15
        assert phase_config["compression"]["mirror_ratio"] == 1.5

    def test_default_runtime(self):
        """default_runtime() should return dict with t_end and dt."""
        config = BelovaMergingConfiguration()
        runtime = config.default_runtime()

        assert "t_end" in runtime
        assert "dt" in runtime
        assert runtime["dt"] > 0

    def test_build_phase_specs_returns_merging_phase(self):
        """build_phase_specs() should return list with MergingPhase spec."""
        config = BelovaMergingConfiguration()
        specs = config.build_phase_specs()

        assert len(specs) >= 1
        assert specs[0].phase_class == "MergingPhase"
        assert "separation" in specs[0].config


class TestBelovaCase1Configuration:
    """Tests for BelovaCase1Configuration (large FRC, no compression)."""

    def test_has_correct_frc_parameters(self):
        """Case 1 should have S*=25.6, E=2.9, xs=0.69."""
        config = BelovaCase1Configuration()

        assert config.s_star == 25.6
        assert config.elongation == 2.9
        assert config.xs == 0.69
        assert config.beta_s == 0.2

    def test_has_correct_merging_parameters(self):
        """Case 1 should have separation=3.0, velocity=0.2."""
        config = BelovaCase1Configuration()

        assert config.separation == 3.0
        assert config.initial_velocity == 0.2
        assert config.compression is None

    def test_has_correct_geometry(self):
        """Case 1 should use 64x512 grid with zc=5.0."""
        config = BelovaCase1Configuration()

        assert config.nr == 64
        assert config.nz == 512
        assert config.domain_half_length == 5.0


class TestBelovaCase2Configuration:
    """Tests for BelovaCase2Configuration (small FRC, no compression)."""

    def test_has_correct_frc_parameters(self):
        """Case 2 should have S*=20, E=1.5, xs=0.53."""
        config = BelovaCase2Configuration()

        assert config.s_star == 20.0
        assert config.elongation == 1.5
        assert config.xs == 0.53
        assert config.beta_s == 0.2

    def test_has_correct_merging_parameters(self):
        """Case 2 should have separation=1.5, velocity=0.1."""
        config = BelovaCase2Configuration()

        assert config.separation == 1.5
        assert config.initial_velocity == 0.1
        assert config.compression is None

    def test_has_correct_geometry(self):
        """Case 2 should use 64x256 grid with zc=3.0."""
        config = BelovaCase2Configuration()

        assert config.nr == 64
        assert config.nz == 256
        assert config.domain_half_length == 3.0


class TestBelovaCase4Configuration:
    """Tests for BelovaCase4Configuration (large FRC with compression)."""

    def test_has_correct_frc_parameters(self):
        """Case 4 should have same FRC params as Case 1."""
        config = BelovaCase4Configuration()

        assert config.s_star == 25.6
        assert config.elongation == 2.9
        assert config.xs == 0.69
        assert config.beta_s == 0.2

    def test_has_compression_config(self):
        """Case 4 should have compression configuration."""
        config = BelovaCase4Configuration()

        assert config.compression is not None
        assert config.compression["mirror_ratio"] == 1.5
        assert config.compression["ramp_time"] == 19.0
        assert config.compression["profile"] == "cosine"

    def test_compression_drives_merging(self):
        """Case 4 should have zero initial velocity (compression drives merging)."""
        config = BelovaCase4Configuration()

        assert config.initial_velocity == 0.0


class TestConfigurationRegistry:
    """Tests for configuration registry integration."""

    def test_base_class_in_registry(self):
        """BelovaMergingConfiguration should be in registry."""
        assert "BelovaMergingConfiguration" in CONFIGURATION_REGISTRY

    def test_case1_in_registry(self):
        """BelovaCase1Configuration should be in registry."""
        assert "BelovaCase1Configuration" in CONFIGURATION_REGISTRY

    def test_case2_in_registry(self):
        """BelovaCase2Configuration should be in registry."""
        assert "BelovaCase2Configuration" in CONFIGURATION_REGISTRY

    def test_case4_in_registry(self):
        """BelovaCase4Configuration should be in registry."""
        assert "BelovaCase4Configuration" in CONFIGURATION_REGISTRY

    def test_registry_returns_correct_class(self):
        """Registry lookup should return the correct class."""
        cls = CONFIGURATION_REGISTRY["BelovaCase2Configuration"]
        config = cls()

        assert isinstance(config, BelovaCase2Configuration)
        assert config.name == "belova_case2_small_frc"


