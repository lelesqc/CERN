#!/bin/bash

SIGMA=1
MACHINE="FCC"    # FCC, ALS
INIT_COND="gaussian"    # gaussian, circle, grid, ring
PARTICLES=10000
MODE="phasespace"    # phasespace, evolution
DATA_FILE="./integrator/${MODE}_qp_${PARTICLES}_${MACHINE}_relaxed_1.00.npz"

MODULATION="yes"    
THERMAL_BATH="yes"

if [ "$MACHINE" == "FCC" ]; then
    DATA_SUFFIX="fcc"
    PARAMS_MODULE="params_fcc"
    
    if [ "$INIT_COND" == "grid" ] || [ "$INIT_COND" == "circle" ]; then
        GRID_LIM=3.2
    elif [ "$INIT_COND" == "gaussian" ]; then
        GRID_LIM=1.9
    elif [ "$INIT_COND" == "ring" ]; then
        GRID_LIM=3.4
    fi

elif [ "$MACHINE" == "ALS" ]; then
    DATA_SUFFIX="als"
    PARAMS_MODULE="params"

    if [ "$INIT_COND" == "grid" ] || [ "$INIT_COND" == "circle" ]; then
        GRID_LIM=8.0
    elif [ "$INIT_COND" == "gaussian" ]; then
        GRID_LIM=10.0
    elif [ "$INIT_COND" == "ring" ]; then
        GRID_LIM=13
    fi
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