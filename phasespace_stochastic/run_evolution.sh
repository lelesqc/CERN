#!/bin/bash

#GRID_LIM=1.9    # FCC
#GRID_LIM=10.0    # ALS
GRID_LIM=0.6    # FCC gaussian
#GRID_LIM=3.33    # ALS gaussian
PARTICLES=10000
MODE="evolution"  # Options: "evolution", "phasespace"

# accept DATA_FILE as first argument, fallback to default
DATA_FILE="${1:-./integrator/evolved_qp_evolution.npz}"

echo "Using DATA_FILE: $DATA_FILE"

echo "Evolving the system..."

python generate_init_conditions.py ${GRID_LIM} ${PARTICLES} ${DATA_FILE}
python integrator.py ${MODE} ${PARTICLES}
python action_angle.py ${MODE}
python plotter.py ${MODE}

echo "Completed."


# PUNTO FISSO ISOLA FCC: (2.66, -0.06)
# PUNTO FISSO ISOLA ALS: (10.3, -0.2)