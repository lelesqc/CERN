#!/bin/bash

# 25 valori (lineare) per nu_m_i: 0.96125 -> 0.9615

NU_I_ARRAY=($(python3 -c "import numpy as np; print(' '.join([f'{v:.7f}' for v in np.linspace(0.9585, 0.962, 10)]))"))

for idx in "${!NU_I_ARRAY[@]}"; do
    NU_I=${NU_I_ARRAY[$idx]}
    echo "Run $((idx+1))/10  ->  nu_m_i=$NU_I"

    # Aggiorna nu_m_i (prima occorrenza non commentata)
    sed -i "0,/^nu_m_i:/s/^nu_m_i:.*/nu_m_i: $NU_I/" params.yaml

    # Lancia la simulazione
    #cp params.yaml "params_sweep_${idx}.yaml"

    # Lancia la simulazione, passando la copia ai job
    #SWEEP_PARAMS="params_sweep_${idx}.yaml"
    bash run_evolution.sh
done

#rm params_sweep_*.yaml

echo "Sweep completato."