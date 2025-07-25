import numpy as np
import matplotlib.pyplot as plt

data = np.load("actions_stuff/actions_first_part.npz")
actions = data['init_actions']

print(actions[0])