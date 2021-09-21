
import math
import torch

import numpy as np
import matplotlib.pyplot as plt

def plot_distributions(n_hw, n_v0, n_alpha):

    # Visualize data distribution
    fig, axs = plt.subplots(1, 3)
    axs[0].hist(n_hw)
    axs[1].hist(n_v0)
    axs[2].hist(n_alpha)
    
    plt.show()

def gen_dataset(in_dim):

    # Generate random hw: normal distribution from 0 - 100m
    n_hw = np.random.normal(50.0, 10.0, in_dim)
    # Generate random v0
    n_v0 = np.random.normal(0.5, 0.1, in_dim)
    # Generate random alpha
    n_alpha = np.random.normal(5.0, math.sqrt(2.0), in_dim)

    # Calculate output values
    l_left_hand_side = []
    for i in range(in_dim):
        l_left_hand_side.append(n_v0[i] + n_alpha[i] * n_hw[i])

    n_left_hand_side = np.array(l_left_hand_side)

    # Convert to torch tensors for processing
    hw = torch.tensor(n_hw)
    v0 = torch.tensor(n_v0)
    alpha = torch.tensor(n_alpha)

    # Create input data tensor
    right_hand_side = torch.stack([v0, alpha, hw])
    left_hand_side = torch.tensor(n_left_hand_side)

    return right_hand_side, left_hand_side