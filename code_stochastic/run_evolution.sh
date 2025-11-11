#!/bin/bash

#RADIUS=10
RADIUS=2.5
#RADIUS=0.6    
SIGMA=0.0827    # emittanza d'equilibrio per FCC
#SIGMA=0.1

PARTICLES=10000
SEC_TO_PLOT=5
DATA_FILE="./init_conditions/relaxed_qp_fcc.npz"
#DATA_FILE="./integrator/evolved_qp_last_relaxed_fcc.npz"

# Poincaré mode options: "last", "all", "none"
POINCARE_MODE="last"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${RADIUS} ${SIGMA} ${PARTICLES}
python integrator.py ${POINCARE_MODE}
python action_angle.py ${POINCARE_MODE}
python plotter.py ${POINCARE_MODE} ${PARTICLES} ${SEC_TO_PLOT}

echo "Completed."