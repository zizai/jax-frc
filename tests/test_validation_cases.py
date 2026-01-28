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
