def test_frozen_flux_validation_setup_uses_cartesian_defaults():
    from validation.cases.analytic.frozen_flux import setup_configuration

    cfg = setup_configuration()
    assert cfg["nx"] == 64
    assert cfg["ny"] == 64  # Pseudo-2D in x-y plane
    assert cfg["nz"] == 1   # Thin z


def test_frozen_flux_validation_yaml_uses_cartesian_labels():
    from pathlib import Path

    yaml_path = Path("validation/cases/analytic/frozen_flux.yaml")
    contents = yaml_path.read_text(encoding="utf-8")

    assert "plane: xy" in contents
    assert "field: B_magnitude" in contents


def test_magnetic_diffusion_open_bc_defaults():
    from jax_frc.configurations.magnetic_diffusion import MagneticDiffusionConfiguration

    config = MagneticDiffusionConfiguration()
    assert config.bc_x == "neumann"
    assert config.bc_y == "neumann"
    assert config.bc_z == "neumann"


def test_magnetic_diffusion_dirichlet_case_exists():
    from validation.cases.analytic.magnetic_diffusion_dirichlet import run_validation

    assert callable(run_validation)
