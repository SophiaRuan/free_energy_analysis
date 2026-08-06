"""Tests for free_energy_analysis.config.load_system_config."""
from pathlib import Path

import pytest

from free_energy_analysis.config import load_system_config

REPO_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "ele_machine.yaml")

VALID_CONFIG = """
system_tag: "LiClOH"
solute_ref_atom: "Li"
coord_env: ["O", "H", "Cl", "Li"]
water_o_symbol: "O"
anion_symbol: "Cl"
lammps_atom_types:
  water_o: 1
  water_h: 2
  cation: 3
  anion: 4
"""


def test_loads_the_real_repo_config():
    cfg = load_system_config(REPO_CONFIG)
    assert cfg["system_tag"] == "LiClOH"
    assert cfg["solute_ref_atom"] == "Li"
    assert cfg["coord_env"] == ["O", "H", "Cl", "Li"]
    assert cfg["lammps_atom_types"] == {"water_o": 1, "water_h": 2, "cation": 3, "anion": 4}


def test_loads_a_well_formed_config(tmp_path):
    config_file = tmp_path / "system.yaml"
    config_file.write_text(VALID_CONFIG)
    cfg = load_system_config(str(config_file))
    assert cfg["anion_symbol"] == "Cl"


@pytest.mark.parametrize("missing_key", [
    "system_tag", "solute_ref_atom", "coord_env",
    "water_o_symbol", "anion_symbol", "lammps_atom_types",
])
def test_raises_on_missing_top_level_key(tmp_path, missing_key):
    lines = [line for line in VALID_CONFIG.strip().splitlines() if not line.startswith(f"{missing_key}:")]
    if missing_key == "lammps_atom_types":
        # also drop the nested lammps_atom_types block
        lines = [l for l in lines if not l.startswith("  ")]
    config_file = tmp_path / "system.yaml"
    config_file.write_text("\n".join(lines))

    with pytest.raises(ValueError, match=missing_key):
        load_system_config(str(config_file))


@pytest.mark.parametrize("missing_type", ["water_o", "water_h", "cation", "anion"])
def test_raises_on_missing_lammps_atom_type(tmp_path, missing_type):
    lines = [line for line in VALID_CONFIG.strip().splitlines() if not line.strip().startswith(f"{missing_type}:")]
    config_file = tmp_path / "system.yaml"
    config_file.write_text("\n".join(lines))

    with pytest.raises(ValueError, match=missing_type):
        load_system_config(str(config_file))
