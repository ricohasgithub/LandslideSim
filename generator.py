
import math
import torch

import numpy as np
import matplotlib.pyplot as plt

# Generate random hw: normal distribution from 0 - 100m
n_hw = np.random.normal(50.0, 10.0, 100000)
# Generate random v0
n_v0 = np.random.normal(0.5, 0.1, 100000)
# Generate random alpha
n_alpha = np.random.normal(5.0, math.sqrt(2.0), 100000)

# Convert to torch tensors for processing
hw = torch.tensor(n_hw)
v0 = torch.tensor(n_v0)
n_alpha = torch.tensor(n_alpha)

# Visualize data distribution
fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].hist(n_hw)
axs[1].hist(n_v0)
axs[1].hist(n_alpha)

plt.show()
