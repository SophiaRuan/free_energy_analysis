#!/bin/bash
# variables
LI_INDEX=2946 # (id-1)!!!
NUMBER_OF_CV=3  # Customize as needed
CONC=0.5
BASE_PATH="../LiCl_0.5M_298K"
NSTRIDES=10
O_RADII=2.65
H_RADII=2.95
CL_RADII=3.05
TEMP=298

# Run Python script
conda activate ele_machine
python colvars_analyzer_script.py --base_dir $BASE_PATH --number_of_cv $NUMBER_OF_CV &> colvars_analysis.log
echo "Colvars analysis completed!"
python free_energy_analysis.py --base_path $BASE_PATH --nstrides $NSTRIDES --O_radii $O_RADII --Cl_radii $CL_RADII --Li_index $LI_INDEX --T $TEMP &> free_energy_analysis.log
echo "Free energy analysis completed!"
python free_energy_correction.py --base_path $BASE_PATH --nstrides $NSTRIDES --O_radii $O_RADII --H_radii $H_RADII --Cl_radii $CL_RADII --Li_index $LI_INDEX --T $TEMP --conc $CONC &> energy_correction_analysis.log
echo "Free energy correction completed!"
RESULTS_DIR="../LiCl_0.5M_298K/results"
mkdir -p ${RESULTS_DIR}
mv ./*.png ${RESULTS_DIR}/
mv ./*.csv ${RESULTS_DIR}/
mv ./*.pkl ${RESULTS_DIR}/
mv ./*.log ${RESULTS_DIR}/
echo "Moved analysis results to results directory at "
