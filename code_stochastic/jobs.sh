#!/bin/bash
# filepath: /mnt/c/Users/emanu/CERN/code_stochastic/launch_jobs.sh

SWEEP_PARAMS=$1
N_PARTICLES=10000
N_JOBS=20      
CHUNK=$((N_PARTICLES / N_JOBS))  # 500

for ((i=0; i<N_JOBS; i++)); do
    IDX_START=$((i * CHUNK))
    IDX_END=$((IDX_START + CHUNK))
    if (( IDX_END > N_PARTICLES )); then IDX_END=$N_PARTICLES; fi
    ./run_evolution.sh $IDX_START $IDX_END $SWEEP_PARAMS &
done
wait