def test_frozen_flux_validation_setup_uses_cartesian_defaults():
    from validation.cases.analytic.frozen_flux import setup_configuration

    cfg = setup_configuration()
    assert cfg["nx"] == 64
    assert cfg["ny"] == 1
    assert cfg["nz"] == 64


def test_frozen_flux_validation_yaml_uses_cartesian_labels():
    from pathlib import Path

    yaml_path = Path("validation/cases/analytic/frozen_flux.yaml")
    contents = yaml_path.read_text(encoding="utf-8")

    assert "axis: x" in contents
    assert "field: B_y" in contents
