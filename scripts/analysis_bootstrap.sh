#!/bin/bash
# variables
LI_INDEX=2946 # (id-1)!!!
NUMBER_OF_CV=3  # Customize as needed
CONC=0.5
BASE_PATH="../LiCl_0.5M_298K/"

NSTRIDES=10
O_RADII=2.65
H_RADII=2.95
CL_RADII=3.05
TEMP=298

# Run Python script
conda activate ele_machine
python colvars_analyzer_script.py --base_dir $BASE_PATH --number_of_cv $NUMBER_OF_CV &> colvars_analysis.log
echo "Colvars analysis completed!"
RESULTS_DIR="../LiCl_0.5M_298K/results"
mkdir -p $RESULTS_DIR
mv $BASE_PATH/*.csv $RESULTS_DIR/
mv $BASE_PATH/*.pkl $RESULTS_DIR/
mv $BASE_PATH/*.png $RESULTS_DIR/
mv $BASE_PATH/*.log $RESULTS_DIR/
echo "Results for Colvars analysis moved to $RESULTS_DIR"

# Loop over different SKIP_FRAMES values
for n_skip in {0..9}; do
    SKIP_FRAMES=$n_skip

    # Free Energy Analysis
    python free_energy_analysis.py --base_path $BASE_PATH --skip_frames $SKIP_FRAMES --nstrides $NSTRIDES --O_radii $O_RADII --Cl_radii $CL_RADII --Li_index $LI_INDEX --T $TEMP &> "free_energy_analysis_.log"
    echo "Free energy analysis with SKIP_FRAMES=$SKIP_FRAMES completed!"

    # Free Energy Correction
    python free_energy_correction.py --base_path $BASE_PATH --nstrides $NSTRIDES --O_radii $O_RADII --H_radii $H_RADII --Cl_radii $CL_RADII --Li_index $LI_INDEX --T $TEMP --conc $CONC &> "energy_correction_analysis_.log"
    echo "Free energy correction with SKIP_FRAMES=$SKIP_FRAMES completed!"

    # Move results to results directory
    RESULTS_DIR="../LiCl_0.5M_298K/results_bootstrap/SKIP_${SKIP_FRAMES}"
    mkdir -p $RESULTS_DIR
    mv $BASE_PATH/*.csv $RESULTS_DIR/
    mv $BASE_PATH/*.pkl $RESULTS_DIR/
    mv $BASE_PATH/*.png $RESULTS_DIR/
    mv $BASE_PATH/*.log $RESULTS_DIR/
    echo "Results for SKIP_FRAMES=$SKIP_FRAMES moved to $RESULTS_DIR"
done
