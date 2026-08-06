"""Tests for EnergyCorrectionAnalyzer.get_activity_from_conc."""
import pytest

from free_energy_analysis.free_energy_tool import EnergyCorrectionAnalyzer


def make_analyzer(T, activity_fit=None):
    analyzer = EnergyCorrectionAnalyzer.__new__(EnergyCorrectionAnalyzer)
    analyzer.T = T
    analyzer.activity_fit = activity_fit or EnergyCorrectionAnalyzer._DEFAULT_ACTIVITY_FIT_BY_TEMPERATURE
    return analyzer


@pytest.mark.parametrize("T, conc, expected", [
    (298, 0.5, -0.0444 * 0.5 + 1.0014),
    (298, 10, -0.0444 * 10 + 1.0014),
    (283, 5, -0.0507 * 5 + 1.0),
    (313, 1, -0.0422 * 1 + 1.0),
])
def test_matches_linear_fit_below_solubility(T, conc, expected):
    analyzer = make_analyzer(T)
    assert analyzer.get_activity_from_conc(conc) == pytest.approx(expected)


@pytest.mark.parametrize("T, solubility", [(298, 20), (283, 17.5), (313, 21)])
def test_clamps_to_solubility_limit_above_it(T, solubility):
    analyzer = make_analyzer(T)
    at_limit = analyzer.get_activity_from_conc(solubility)
    above_limit = analyzer.get_activity_from_conc(solubility + 10)
    assert above_limit == pytest.approx(at_limit)


def test_raises_clear_error_for_unconfigured_temperature():
    analyzer = make_analyzer(350)
    with pytest.raises(ValueError, match="350"):
        analyzer.get_activity_from_conc(1.0)


def test_custom_activity_fit_overrides_default_without_touching_it():
    # A second "salt" that happens to share LiCl's 298K, with deliberately
    # different numbers - this must not collide with or mutate the built-in
    # LiCl default, since two systems can share a temperature.
    other_salt_fit = {298: {"slope": 1.0, "intercept": 0.0, "solubility": 5}}
    analyzer = make_analyzer(298, activity_fit=other_salt_fit)

    assert analyzer.get_activity_from_conc(2) == pytest.approx(2.0)
    assert EnergyCorrectionAnalyzer._DEFAULT_ACTIVITY_FIT_BY_TEMPERATURE[298]["slope"] == -0.0444

    licl_analyzer = make_analyzer(298)
    assert licl_analyzer.get_activity_from_conc(2) == pytest.approx(-0.0444 * 2 + 1.0014)


def test_custom_activity_fit_still_raises_for_temperature_it_lacks():
    other_salt_fit = {350: {"slope": 1.0, "intercept": 0.0, "solubility": 5}}
    analyzer = make_analyzer(298, activity_fit=other_salt_fit)
    with pytest.raises(ValueError, match="298"):
        analyzer.get_activity_from_conc(1.0)
