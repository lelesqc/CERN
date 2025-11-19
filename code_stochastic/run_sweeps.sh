#!/bin/bash

# 25 valori (lineare) per epsilon_f: valoriiiiiiiiiii:  0.96 -> 0.964 fcc    0.959 -> 0.9625 als

EPS_I_ARRAY=($(python3 -c "import numpy as np; print(' '.join([f'{v:.7f}' for v in np.linspace(0.0240125, 0.0240375, 5)]))"))
NU_I_ARRAY=($(python3 -c "import numpy as np; print(' '.join([f'{v:.7f}' for v in np.linspace(0.9605, 0.9615, 5)]))"))

#NU_I_ARRAY=(0.96168421 0.96294737)
#EPS_I_ARRAY=(0.02404210525 0.02407368425)

# Controllo lunghezze
if [ ${#EPS_I_ARRAY[@]} -ne ${#NU_I_ARRAY[@]} ]; then
  echo "Array length mismatch"; exit 1
fi

for idx in "${!EPS_I_ARRAY[@]}"; do
    EPS_I=${EPS_I_ARRAY[$idx]}
    NU_I=${NU_I_ARRAY[$idx]}
    res=$(echo "scale=3; $EPS_I / $NU_I" | bc -l) 
    echo "Run $((idx+1))/20  ->  a_i="$res"  nu_m_i=$NU_I"

    # Aggiorna epsilon_i e nu_m_i (prima occorrenza non commentata)
    sed -i "0,/^epsilon_i:/s/^epsilon_i:.*/epsilon_i: $EPS_I/" params.yaml
    sed -i "0,/^nu_m_i:/s/^nu_m_i:.*/nu_m_i: $NU_I/" params.yaml

    # Lancia la simulazione
    cp params.yaml "params_sweep_${idx}.yaml"

    # Lancia la simulazione, passando la copia ai job
    SWEEP_PARAMS="params_sweep_${idx}.yaml"
    bash jobs.sh "$SWEEP_PARAMS"
done

rm params_sweep_*.yaml

echo "Sweep completato."