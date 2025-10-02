#!/bin/bash

GRID_LIM=2.0    # FCC
#GRID_LIM=10.0    # ALS
PARTICLES=10000

MODE="phasespace"  # Options: "evolution", "phasespace"

DATA_FILE="../code/integrator/evolved_qp_last.npz"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${GRID_LIM} ${PARTICLES}
#python integrator.py ${MODE} ${PARTICLES}
#python action_angle.py ${MODE}

#if [ "$MODE" = "tune" ]; then
#    python tune.py
#fi

#python plotter.py ${MODE}

echo "Completed."