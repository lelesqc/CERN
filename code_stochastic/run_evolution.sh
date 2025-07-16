#!/bin/bash

RADIUS=10
PARTICLES=1000
SEC_TO_PLOT=5

# Poincaré mode options: "first, "last", "all", "none"
POINCARE_MODE="last"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${RADIUS} ${PARTICLES}
python integrator.py ${POINCARE_MODE}
python action_angle.py ${POINCARE_MODE}
python plotter.py ${POINCARE_MODE} ${PARTICLES} ${SEC_TO_PLOT}

echo "Completed."