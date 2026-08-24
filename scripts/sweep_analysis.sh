#!/bin/bash
# Runs analysis.sh across a batch of concentrations/temperatures for the
# LiCl solvation system, skipping any system whose data directory isn't
# present under ../data. Data folders are expected to follow the
# LiCl_${CONC}M_${TEMP}K naming convention (see analysis.sh).
#
# LI_INDEX is intentionally not listed here: analysis.sh auto-derives it per
# system from that system's own colvar.lmp (the tagged ion's atom ID is
# already recorded there), so there is no hand-maintained mapping to keep in
# sync when concentrations are added or removed.
#
# For a different salt/ion pair entirely, this script's CONC bookkeeping
# doesn't apply as-is — see the README for what else to change
# (configs/ele_machine.yaml solute/solvent atoms, radii cutoffs in analysis.sh).

CONCENTRATIONS=(0.5 2 5 7 10 12 15 18 20 22 25)
TEMPS=(283 298 313)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for TEMP in "${TEMPS[@]}"; do
    for CONC in "${CONCENTRATIONS[@]}"; do
        BASE_PATH="../data/LiCl_${CONC}M_${TEMP}K"

        if [ ! -d "${SCRIPT_DIR}/${BASE_PATH}" ]; then
            echo "Skipping ${CONC}M @ ${TEMP}K: no data at ${BASE_PATH}"
            continue
        fi

        echo "=== Running analysis for ${CONC}M LiCl @ ${TEMP}K ==="
        CONC="$CONC" TEMP="$TEMP" BASE_PATH="$BASE_PATH" \
            "${SCRIPT_DIR}/analysis.sh"
    done
done
