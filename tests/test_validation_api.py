import pytest


def test_validation_runner_removed():
    with pytest.raises(ImportError):
        from jax_frc.validation import ValidationRunner
