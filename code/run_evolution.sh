#!/bin/bash

export BASE_DIR="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

RADIUS=10.0
PARTICLES=10000
SEC_TO_PLOT=5

# Poincaré mode options: "first, "last", "all", "none"
POINCARE_MODE="last"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${RADIUS} ${PARTICLES}
python integrator.py ${POINCARE_MODE} ${PARTICLES}
python action_angle.py ${POINCARE_MODE} ${PARTICLES}
python plotter.py ${POINCARE_MODE} ${PARTICLES} ${SEC_TO_PLOT}

echo "Completed."