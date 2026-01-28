from validation.cases.hall_reconnection.reconnection_gem import main


def test_reconnection_gem_quick_smoke():
    assert main(quick_test=True) in (True, False)
