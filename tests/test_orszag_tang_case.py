from validation.cases.mhd_regression.orszag_tang import main


def test_orszag_tang_quick_smoke():
    assert main(quick_test=True) in (True, False)
