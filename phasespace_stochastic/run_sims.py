import yaml
import subprocess
import numpy as np
import os
import importlib

os.environ["MACHINE"] = "FCC"
os.environ["THERMAL_BATH"] = "yes"
os.environ["MODULATION"] = "yes"
os.environ["PARAMS_MODULE"] = "params_fcc_ww"

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

nu_ms = np.linspace(.85, .90, 11)
epsilons = np.linspace(.060, .065, 6)

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