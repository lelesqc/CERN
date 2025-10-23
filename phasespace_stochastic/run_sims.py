import yaml
import subprocess
from pathlib import Path
import numpy as np

var_to_scan = "sigma"    # sigma, epsilon

if var_to_scan == "sigma":
    var_list = np.linspace(0, 1.0, 25)

elif var_to_scan == "epsilon":
    var_list = 1

out_dir = Path("./integrator")
out_dir.mkdir(parents=True, exist_ok=True)

for var in var_list:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    params[var_to_scan] = float(var)
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)

    #data_file = out_dir / f"evolved_qp_evolution_{var:.3f}.npz"

    #print(f"Running sim for {var_to_scan}={var} -> DATA_FILE={data_file}")
    #subprocess.run(["./run_evolution.sh", str(data_file)], check=True)