#!/bin/bash

GRID_LIM=12.9
PARTICLES=80
FFT_STEPS=4096
# 4096, 8192, 16384, 32768

MODE="tune"  # Options: "tune", "phasespace"
TUNE_MODE="phaseadvance"  # Options: "fft", "phaseadvance"

DATA_FILE="../code/integrator/evolved_qp_last_10000.npz"

# -------------

echo "Evolving the system..."

python generate_init_conditions.py ${GRID_LIM} ${PARTICLES} ${DATA_FILE}
python integrator.py ${MODE} ${FFT_STEPS}
#python action_angle.py ${MODE}

if [ "$MODE" = "tune" ]; then
    python tune.py ${FFT_STEPS} ${TUNE_MODE}
fi

#python plotter.py ${MODE} ${FFT_STEPS} ${TUNE_MODE}

echo "Completed."