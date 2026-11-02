import yaml
import subprocess
import numpy as np
import os
import importlib

os.environ["MACHINE"] = "FCC"
os.environ["MODULATION"] = "yes"
os.environ["THERMAL_BATH"] = "yes"
os.environ["PARAMS_MODULE"] = "params_fcc_z"

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

nu_ms = np.linspace(.84, .94, 16)
#epsilons = np.linspace(.065, .080, 16)
epsilons = np.linspace(.025, .080, 16)

var_to_scan = "epsilon"

if var_to_scan == "epsilon":
    var_list = np.copy(epsilons)

elif var_to_scan == "nu_m":
    var_list = np.copy(nu_ms)

for var in var_list:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    params[var_to_scan] = float(var)

    with open("params.yaml", "w") as f:
        yaml.dump(params, f)

    subprocess.run(["./run_evolution.sh", f"{var:.4f}"], check=True)