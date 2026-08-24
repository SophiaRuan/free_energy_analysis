"""Loader for the system-chemistry config (see configs/ele_machine.yaml)."""
import yaml

REQUIRED_KEYS = (
    "system_tag",
    "solute_ref_atom",
    "coord_env",
    "water_o_symbol",
    "anion_symbol",
    "lammps_atom_types",
)


def load_system_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"System config {path} is missing required keys: {missing}")

    missing_types = [k for k in ("water_o", "water_h", "cation", "anion") if k not in cfg["lammps_atom_types"]]
    if missing_types:
        raise ValueError(f"System config {path}: lammps_atom_types is missing: {missing_types}")

    return cfg
