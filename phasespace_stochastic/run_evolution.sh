#!/bin/bash

SIGMA=1
MACHINE="FCC"    # FCC, ALS
INIT_COND="gaussian"    # gaussian, circle, grid
PARTICLES=10000
MODE="phasespace"    # phasespace, evolution
DATA_FILE="./integrator/${MODE}_qp_${PARTICLES}_${MACHINE}_relaxed_1.00.npz"

MODULATION="yes"    
THERMAL_BATH="yes"

if [ "$MACHINE" == "FCC" ]; then
    DATA_SUFFIX="fcc"
    PARAMS_MODULE="params_fcc"
    GRID_LIM=3

elif [ "$MACHINE" == "ALS" ]; then
    DATA_SUFFIX="als"
    PARAMS_MODULE="params"
    GRID_LIM=8
fi

export MACHINE
export PARAMS_MODULE
export MODULATION
export THERMAL_BATH

echo "Evolving the system..."

python generate_init_conditions.py ${INIT_COND} ${GRID_LIM} ${SIGMA} ${PARTICLES}
python integrator.py ${INIT_COND} ${MODE} ${PARTICLES}
python action_angle.py ${MODE} ${PARTICLES}
#python tune.py ${MODE} ${PARTICLES}
python plotter.py ${MODE} ${PARTICLES}

echo "Completed."