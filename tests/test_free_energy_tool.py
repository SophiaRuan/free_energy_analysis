"""Tests for EnergyCorrectionAnalyzer.get_activity_from_conc."""
import pytest

from free_energy_analysis.free_energy_tool import EnergyCorrectionAnalyzer


def make_analyzer(T):
    analyzer = EnergyCorrectionAnalyzer.__new__(EnergyCorrectionAnalyzer)
    analyzer.T = T
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
