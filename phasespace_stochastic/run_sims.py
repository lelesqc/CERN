import yaml
import subprocess
import numpy as np
import os
import importlib

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

nu_ms = np.linspace(.84, .94, 11)
epsilon = .0282

var_to_scan = "nu_m"

if var_to_scan == "epsilon":
    var_list = 1

elif var_to_scan == "nu_m":
    var_list = np.copy(nu_ms)

for var in var_list:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    params[var_to_scan] = float(var)

    with open("params.yaml", "w") as f:
        yaml.dump(params, f)

    subprocess.run(["./run_evolution.sh", f"{var:.4f}"], check=True)