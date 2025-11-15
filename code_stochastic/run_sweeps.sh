#!/bin/bash

# 25 valori (lineare) per epsilon_f: 0.025 -> 0.0225
EPS_I_ARRAY=($(python3 -c "import numpy as np; print(' '.join([f'{v:.7f}' for v in np.linspace(0.024, 0.0240375, 25)]))"))
# 25 valori per nu_m_f: 1.00 -> 0.90
NU_I_ARRAY=($(python3 -c "import numpy as np; print(' '.join([f'{v:.7f}' for v in np.linspace(0.96, 0.9615, 25)]))"))

# Controllo lunghezze
if [ ${#EPS_I_ARRAY[@]} -ne ${#NU_I_ARRAY[@]} ]; then
  echo "Array length mismatch"; exit 1
fi

for idx in "${!EPS_I_ARRAY[@]}"; do
    EPS_I=${EPS_I_ARRAY[$idx]}
    NU_I=${NU_I_ARRAY[$idx]}
    res=$(echo "scale=3; $EPS_I / $NU_I" | bc -l) 
    echo "Run $((idx+1))/25  ->  a_i="$res"  nu_m_i=$NU_I"

    # Aggiorna epsilon_i e nu_m_i (prima occorrenza non commentata)
    sed -i "0,/^epsilon_i:/s/^epsilon_i:.*/epsilon_i: $EPS_I/" params.yaml
    sed -i "0,/^nu_m_i:/s/^nu_m_i:.*/nu_m_i: $NU_I/" params.yaml

    # Lancia la simulazione
    bash jobs.sh

    # (Opzionale) salva risultati separati
    # outdir="results/eps_${EPS_I}_nu_${NU_I}"
    # mkdir -p "$outdir"
    # cp integrator/evolved_qp_* "$outdir"/
done

echo "Sweep completato."