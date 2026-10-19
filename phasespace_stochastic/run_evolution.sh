#!/bin/bash

MACHINE="FCC"    # FCC, ALS
INIT_COND="gaussian"   # gaussian, circle, grid, grid_big
PARTICLES=10000
CONFIG=ww    # ww, zh, tt
MODE="evolution"    # phasespace, evolution

MODULATION="yes"    
THERMAL_BATH="yes"

if [ "$MACHINE" == "FCC" ]; then
    DATA_SUFFIX="fcc"
    PARAMS_MODULE="params_fcc_${CONFIG}"
    #GRID_LIM=0.2
    GRID_LIM=5.0

elif [ "$MACHINE" == "ALS" ]; then
    DATA_SUFFIX="als"
    PARAMS_MODULE="params_als"
    CONFIG="als"
    GRID_LIM=6
fi

export MACHINE
export PARAMS_MODULE
export MODULATION
export THERMAL_BATH
export CONFIG

echo "Evolving the system..."

python generate_init_conditions.py ${INIT_COND} ${GRID_LIM} ${SIGMA} ${PARTICLES}
python integrator.py ${INIT_COND} ${MODE} ${PARTICLES}
python action_angle.py ${MODE} ${PARTICLES} ${CONFIG}
#python tune.py ${MODE} ${PARTICLES}
python plotter.py ${MODE} ${PARTICLES} ${CONFIG}

echo "Completed."