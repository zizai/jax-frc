"""Tests for validation runner."""
import pytest
from pathlib import Path
import yaml


def test_runner_loads_yaml_config(tmp_path):
    """ValidationRunner loads YAML case config."""
    from jax_frc.validation.runner import ValidationRunner

    # Create minimal test case YAML
    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test case',
        'configuration': {
            'class': 'MagneticDiffusionConfiguration',
        },
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")

    assert runner.config['name'] == 'test_case'
    assert runner.config['configuration']['class'] == 'MagneticDiffusionConfiguration'


def test_runner_instantiates_configuration(tmp_path):
    """ValidationRunner creates Configuration from registry."""
    from jax_frc.validation.runner import ValidationRunner
    from jax_frc.configurations import MagneticDiffusionConfiguration

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test',
        'configuration': {'class': 'MagneticDiffusionConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")
    config = runner._build_configuration()

    assert isinstance(config, MagneticDiffusionConfiguration)


def test_runner_instantiates_configuration_with_overrides(tmp_path):
    """ValidationRunner passes overrides to Configuration."""
    from jax_frc.validation.runner import ValidationRunner

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test',
        'configuration': {
            'class': 'MagneticDiffusionConfiguration',
            'overrides': {
                'nr': 16,
                'nz': 128,
                'B_peak': 2.0
            }
        },
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")
    config = runner._build_configuration()

    assert config.nr == 16
    assert config.nz == 128
    assert config.B_peak == 2.0


def test_runner_raises_for_unknown_configuration(tmp_path):
    """ValidationRunner raises ValueError for unknown configuration class."""
    from jax_frc.validation.runner import ValidationRunner

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test',
        'configuration': {'class': 'NonExistentConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")

    with pytest.raises(ValueError, match="Unknown configuration"):
        runner._build_configuration()


def test_runner_creates_output_directory(tmp_path):
    """ValidationRunner creates timestamped output directory."""
    from jax_frc.validation.runner import ValidationRunner

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test',
        'configuration': {'class': 'MagneticDiffusionConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")
    run_dir = runner._setup_output_dir()

    assert run_dir.exists()
    assert (run_dir / "plots").exists()
    assert "test_case" in run_dir.name


def test_runner_dry_run(tmp_path):
    """ValidationRunner dry_run returns result without running simulation."""
    from jax_frc.validation.runner import ValidationRunner

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'dry_test',
        'description': 'Test',
        'configuration': {'class': 'MagneticDiffusionConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")
    result = runner.run(dry_run=True)

    assert result.case_name == 'dry_test'
    assert result.runtime_seconds == 0.0
    assert result.metrics == {}


def test_runner_computes_metrics(tmp_path):
    """ValidationRunner computes acceptance metrics."""
    from jax_frc.validation.runner import ValidationRunner

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'metric_test',
        'description': 'Test',
        'configuration': {'class': 'MagneticDiffusionConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {
            'quantitative': [
                {
                    'metric': 'test_metric',
                    'value': 0.05,
                    'expected': 0.0,
                    'tolerance': 0.1
                }
            ]
        }
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")

    # Build minimal state for metric computation
    from jax_frc.configurations import MagneticDiffusionConfiguration
    config = MagneticDiffusionConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    metrics = runner._compute_metrics(state, geometry, config)

    assert 'test_metric' in metrics
    assert metrics['test_metric'].passed is True


def test_runner_metric_failure(tmp_path):
    """ValidationRunner correctly reports metric failures."""
    from jax_frc.validation.runner import ValidationRunner

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'fail_test',
        'description': 'Test',
        'configuration': {'class': 'MagneticDiffusionConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {
            'quantitative': [
                {
                    'metric': 'failing_metric',
                    'value': 1.0,
                    'expected': 0.0,
                    'tolerance': 0.1
                }
            ]
        }
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")
    config = runner._build_configuration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    metrics = runner._compute_metrics(state, geometry, config)

    assert 'failing_metric' in metrics
    assert metrics['failing_metric'].passed is False
