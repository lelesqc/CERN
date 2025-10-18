import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def plot_test():
    area_island = 15.7
    area_center = 13.5

    files = glob.glob("./infos/trapping_results_*.npz")

    # Estrai il numero dal nome e ordina dal più grande al più piccolo
    def extract_num(fp):
        match = re.search(r"trapping_results_([0-9.]+)\.npz", fp)
        return float(match.group(1)) if match else -np.inf

    files_sorted = sorted(files, key=extract_num, reverse=True)

    sigma_list = [extract_num(fp) for fp in files_sorted]

    trap_list = []
    sigma_list = []
    for fp in files_sorted:
        num = extract_num(fp)
        sigma_list.append(num)
        data = np.load(fp)
        trap = data["trap"]
        trap_list.append(trap / 10000 * 100)

    # Esempio: somma dei trap per ogni file
    print("sigma_list ordinata:", sigma_list)

    pairs = [
        [3889, 6111],
        [4596, 5404],
        [5318, 4682],
        [5291, 4709],
        [5000, 5000],
        [4543, 5457],
        [4287, 5713],
        [4849, 5151],
        [4956, 5044],
        [5318, 4682],
        [9417, 583],
        [7099, 2901],
    ]

    pairs = np.array(pairs)

    #sigma_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
    #               0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
    #sigma_list = np.linspace(0, 1.0, 50)

    pr_area_law = area_island / (area_island + area_center) * 100
    probs_emp = []

    for val in pairs[:, 0]:
        pr_empirical = val / 10000 * 100
        probs_emp.append(pr_empirical)

    print(len(sigma_list), len(trap_list))
    plt.scatter(sigma_list, trap_list, s=8, label="Empirical prob.")
    plt.axhline(y=pr_area_law, color='r', linestyle='--', label='Areas law')
    plt.ylabel("Trapping probability")
    plt.xlabel("White noise variance")
    plt.legend()
    plt.savefig("../results/trapping_comparisons/areas_law_whitenoise_variation")
    plt.show()
    

if __name__ == "__main__":
    plot_test()

