import pytest

from validation.cases.regression.gem_reconnection import main


@pytest.mark.slow
def test_gem_reconnection_smoke():
    assert main() in (True, False)
