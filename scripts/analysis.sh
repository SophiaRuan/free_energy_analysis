#!/bin/bash
# Runs the free energy analysis pipeline (Linux/macOS bash; on Windows use WSL2).
# See README.rst "Usage" for per-OS setup notes.
#
# Every variable below can be overridden from the environment, so a different
# solvation system can be analyzed without editing this file, e.g.:
#   CONC=2 TEMP=298 ./analysis.sh
# See sweep_analysis.sh to run a batch of concentrations/temperatures.
# variables
CONC="${CONC:-0.5}"
TEMP="${TEMP:-298}"
NUMBER_OF_CV="${NUMBER_OF_CV:-3}"
BASE_PATH="${BASE_PATH:-../data/LiCl_${CONC}M_${TEMP}K}"
NSTRIDES="${NSTRIDES:-10}"
SKIP_FRAMES="${SKIP_FRAMES:-9}"
O_RADII="${O_RADII:-2.65}"
H_RADII="${H_RADII:-2.95}"
CL_RADII="${CL_RADII:-3.05}"

# LI_INDEX (the (id-1) LAMMPS index of the tagged/biased solute ion) has no
# sane universal default - it depends on box composition, which changes per
# system. If not explicitly overridden, derive it from the biased-atom ID
# already recorded in this system's own colvars input: colvar.lmp's first
# CV block has `group1 { atomNumbers N }`, where N is the same tagged-ion
# atom ID for every replica/CV block within one system.
if [ -z "${LI_INDEX}" ]; then
    COLVAR_FILE="${BASE_PATH}/01_IDNR/colvar.lmp"
    LI_ID=$(grep -m1 -oE "atomNumbers[[:space:]]+[0-9]+" "$COLVAR_FILE" 2>/dev/null | grep -oE "[0-9]+")
    if [ -z "$LI_ID" ]; then
        echo "ERROR: LI_INDEX not set and could not auto-derive it from ${COLVAR_FILE}. Set LI_INDEX explicitly." >&2
        exit 1
    fi
    LI_INDEX=$((LI_ID - 1))
    echo "Auto-derived LI_INDEX=${LI_INDEX} (atom id ${LI_ID}) from ${COLVAR_FILE}"
fi

# Locate and activate conda. `conda activate` requires the shell function
# installed by sourcing conda.sh (or `conda init`) - having the `conda`
# binary merely on PATH (e.g. inherited from a parent shell) isn't enough
# and fails with "Run 'conda init' before 'conda activate'". Checks common
# install locations in order; if yours lives elsewhere, source its conda.sh
# yourself before calling this script, or activate the ele_machine env
# before running.
if [ "$(type -t conda)" != "function" ]; then
    for conda_sh in ~/miniforge3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh ~/mambaforge/etc/profile.d/conda.sh; do
        if [ -f "$conda_sh" ]; then
            source "$conda_sh"
            break
        fi
    done
fi
conda activate ele_machine # your environment name
python colvars_analyzer_script.py --base_dir $BASE_PATH --number_of_cv $NUMBER_OF_CV &> colvars_analysis.log
echo "Colvars analysis completed!"
python free_energy_analysis_script.py --base_path $BASE_PATH --skip_frames $SKIP_FRAMES --nstrides $NSTRIDES --O_radii $O_RADII --Cl_radii $CL_RADII --Li_index $LI_INDEX --T $TEMP &> free_energy_analysis.log
echo "Free energy analysis completed!"
python free_energy_correction_script.py --base_path $BASE_PATH --nstrides $NSTRIDES --O_radii $O_RADII --H_radii $H_RADII --Cl_radii $CL_RADII --Li_index $LI_INDEX --T $TEMP --conc $CONC &> energy_correction_analysis.log
echo "Free energy correction completed!"
RESULTS_DIR="${BASE_PATH}/results"
mkdir -p ${RESULTS_DIR}
mv ./*.png ${RESULTS_DIR}/
mv ./*.csv ${RESULTS_DIR}/
mv ./*.pkl ${RESULTS_DIR}/
mv ./*.log ${RESULTS_DIR}/
echo "Moved analysis results to results directory at ${RESULTS_DIR}"
