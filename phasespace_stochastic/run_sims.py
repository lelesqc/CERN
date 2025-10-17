import yaml
import subprocess
from pathlib import Path

sigma_list = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
out_dir = Path("./integrator")
out_dir.mkdir(parents=True, exist_ok=True)

for sigma in sigma_list:
    # aggiorna params.yaml se serve
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    params['sigma'] = float(sigma)
    with open("params.yaml", "w") as f:
        yaml.dump(params, f)

    data_file = out_dir / f"evolved_qp_evolution_{sigma}.npz"

    print(f"Running sim for sigma={sigma} -> DATA_FILE={data_file}")
    subprocess.run(["./run_evolution.sh", str(data_file)], check=True)