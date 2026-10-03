#!/bin/bash

RADIUS=2.0
#RADIUS=2.5
#RADIUS=0.6    
SIGMA=0.001    # emittanza d'equilibrio per FCC

PARTICLES=10000
SEC_TO_PLOT=5
#DATA_FILE="./init_conditions/relaxed_qp_als.npz"
#DATA_FILE="./integrator/evolved_qp_last_relaxed_fcc.npz"
DATA_FILE="../phasespace_stochastic/integrator/evolution_qp_10000_fcc_use_for_relax.npz"

# Poincaré mode options: "last", "all", "none"
POINCARE_MODE="last"

# -------------

#echo "Evolving the system..."

IDX_START=${1:-0}
IDX_END=${2:-${PARTICLES}}

python generate_init_conditions.py ${RADIUS} ${SIGMA} ${PARTICLES} ${DATA_FILE} $IDX_START $IDX_END
python integrator.py ${POINCARE_MODE} $IDX_START $IDX_END
python action_angle.py ${POINCARE_MODE} $IDX_START $IDX_END
python plotter.py ${POINCARE_MODE} ${PARTICLES} ${SEC_TO_PLOT} $IDX_START $IDX_END
#python init_final_actions.py

#echo "Completed."