"""Tests for reference data management."""
import pytest
import jax.numpy as jnp


def test_reference_manager_loads_analytic():
    """ReferenceManager evaluates analytic formulas."""
    from jax_frc.validation.references import ReferenceManager

    mgr = ReferenceManager()
    ref_config = {
        'type': 'analytic',
        'formula': 'A * jnp.exp(-x**2)',
        'variables': ['A', 'x']
    }
    params = {'A': 2.0}
    x = jnp.linspace(-1, 1, 10)

    result = mgr.evaluate_analytic(ref_config['formula'], {'A': params['A'], 'x': x})

    expected = 2.0 * jnp.exp(-x**2)
    assert jnp.allclose(result, expected)


def test_reference_manager_loads_file(tmp_path):
    """ReferenceManager loads CSV reference files."""
    from jax_frc.validation.references import ReferenceManager
    import csv

    # Create test CSV
    csv_path = tmp_path / "test_ref.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerow([0.0, 1.0])
        writer.writerow([1.0, 2.0])

    mgr = ReferenceManager(base_dir=tmp_path)
    ref_config = {
        'type': 'file',
        'path': 'test_ref.csv',
        'columns': {'x': 'x', 'y': 'y'}
    }

    data = mgr.load_file(ref_config)

    assert 'x' in data
    assert 'y' in data
    assert len(data['x']) == 2
