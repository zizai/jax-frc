"""Integration test for full validation pipeline."""
import json
import pytest
from pathlib import Path
import yaml


@pytest.fixture
def validation_case(tmp_path):
    """Create minimal validation case for testing."""
    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'integration_test',
        'description': 'Integration test case',
        'configuration': {
            'class': 'MagneticDiffusionConfiguration',
            'overrides': {
                'nr': 4,
                'nz': 16,  # Small grid for fast testing
            }
        },
        'runtime': {
            't_end': 1e-6,  # Very short run
            'dt': 1e-7
        },
        'acceptance': {
            'quantitative': [
                {
                    'metric': 'l2_error',
                    'field': 'B_z',
                    'threshold': 1.0,  # Very permissive for integration test
                    'description': 'Integration test metric'
                }
            ]
        }
    }))
    return case_yaml


def test_full_validation_pipeline(validation_case, tmp_path):
    """Full pipeline: load config, run sim, generate results."""
    from jax_frc.validation import ValidationRunner

    output_dir = tmp_path / "output"
    runner = ValidationRunner(validation_case, output_dir)
    result = runner.run()

    assert result.case_name == 'integration_test'
    assert result.runtime_seconds > 0

    # Check output files created
    output_dirs = list(output_dir.iterdir())
    assert len(output_dirs) == 1

    run_dir = output_dirs[0]
    assert (run_dir / "config.yaml").exists()

    # Verify metrics.json content
    metrics_file = run_dir / "metrics.json"
    assert metrics_file.exists()

    with open(metrics_file) as f:
        metrics_data = json.load(f)

    assert 'case' in metrics_data
    assert metrics_data['case'] == 'integration_test'
    assert 'runtime_seconds' in metrics_data
    assert metrics_data['runtime_seconds'] > 0
    assert 'overall_pass' in metrics_data
