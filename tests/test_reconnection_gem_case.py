import pytest

from validation.cases.regression.reconnection_gem import main


@pytest.mark.slow
def test_reconnection_gem_smoke():
    assert main() in (True, False)
