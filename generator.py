
import math
import torch

import numpy as np
import matplotlib.pyplot as plt

# Normal distribution from 0 - 100m
n_rand = np.random.normal(50.0, 10.0, 100000)

fig, axs = plt.subplots(1, 1)
axs.hist(n_rand)

plt.show()
