#!/bin/bash

MACHINE="FCC"    # FCC, ALS
INIT_COND="gaussian"    # gaussian, circle, grid
PARTICLES=10000
MODE="phasespace"    # phasespace, evolution
INPUT_DATA="${1:-./integrator/evolved_qp_evolution.npz}"

MODULATION="no"    
THERMAL_BATH="yes"

if [ "$MACHINE" == "FCC" ]; then
    DATA_SUFFIX="fcc"
    PARAMS_MODULE="params_fcc"
    
    if [ "$INIT_COND" == "grid" ] || [ "$INIT_COND" == "circle" ]; then
        GRID_LIM=1.9
    elif [ "$INIT_COND" == "gaussian" ]; then
        GRID_LIM=0.5
    fi

elif [ "$MACHINE" == "ALS" ]; then
    DATA_SUFFIX="als"
    PARAMS_MODULE="params"

    if [ "$INIT_COND" == "grid" ] || [ "$INIT_COND" == "circle" ]; then
        GRID_LIM=10.0
    elif [ "$INIT_COND" == "gaussian" ]; then
        GRID_LIM=3.3
    fi
fi

export MACHINE
export PARAMS_MODULE
export MODULATION
export THERMAL_BATH

echo "Evolving the system..."

python generate_init_conditions.py ${INIT_COND} ${GRID_LIM} ${PARTICLES}
python integrator.py ${INIT_COND} ${MODE} ${PARTICLES}
python action_angle.py ${MODE} ${PARTICLES}
python plotter.py ${MODE} ${PARTICLES}

echo "Completed."


# AREA ISOLA FCC: 15.8
# AREA CENTRO FCC: 13.404

# PUNTO FISSO ISOLA FCC: (2.66, -0.06)
# PUNTO FISSO ISOLA ALS: (10.3, -0.2)