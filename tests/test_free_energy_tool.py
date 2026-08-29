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


def test_init_defaults_to_water_atom_counting():
    analyzer = EnergyCorrectionAnalyzer(
        base_path="x", nstrides=1, data_file="x", traj_list=[], T=298,
    )
    assert analyzer.solvent_atoms_per_molecule == 3
    assert analyzer.solvent_anchor_symbol == "O"


def test_init_accepts_a_different_solvent_atom_count():
    analyzer = EnergyCorrectionAnalyzer(
        base_path="x", nstrides=1, data_file="x", traj_list=[], T=298,
        solvent_atoms_per_molecule=6, solvent_anchor_symbol="N",
    )
    assert analyzer.solvent_atoms_per_molecule == 6
    assert analyzer.solvent_anchor_symbol == "N"


# Flat Hill-notation formulas (no brackets, count omitted when 1) - the
# actual format this codebase's cluster formulas use, e.g. "H8LiO4" for
# Li[H2O]4 (see clu_analysis_sorted.csv from a real run).
@pytest.mark.parametrize("formula, symbol, expected", [
    ("H2Cl2LiO", "O", 1),      # bare "O", no digit suffix -> count 1
    ("H8LiO4", "O", 4),        # Li[H2O]4 in Hill notation
    ("H6LiO3", "O", 3),
    ("H12ClLi2O6", "O", 6),
    ("LiCl3", "O", 0),         # no solvent atom in this cluster at all
    ("N4C8H12Li", "N", 4),     # hypothetical non-water solvent anchor
])
def test_count_solvent_anchor_atoms(formula, symbol, expected):
    assert EnergyCorrectionAnalyzer.count_solvent_anchor_atoms(formula, symbol) == expected


def test_count_solvent_anchor_atoms_defaults_to_oxygen():
    # backward-compat: calling with just a formula (no symbol) still counts O
    assert EnergyCorrectionAnalyzer.count_solvent_anchor_atoms("H8LiO4") == 4
