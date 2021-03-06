
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

# Returns n x n matrix (n = in_dim)
def gen_lin_dataset(in_dim, constant=True):

    # Generate random hw: normal distribution from 0 - 100m
    n_hw = np.random.normal(50.0, 10.0, in_dim)

    # Generate random v0
    if not constant:
        n_v0 = np.random.normal(0.5, 0.1, in_dim)
    else:
        n_v0 = np.full(in_dim, 0.1)
        
    # Generate random alpha
    if not constant:
        n_alpha = np.random.normal(5.0, math.sqrt(2.0), in_dim)
    else:
        n_alpha = np.full(in_dim, 2.5)

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

    return (hw, right_hand_side, left_hand_side)

# Returns n x n matrix (n = in_dim)
def gen_exp_dataset(in_dim, constant=True):

    # Generate random hw: normal distribution from 0 - 100m acts as time t
    # n_hw = np.random.normal(50.0, 10.0, in_dim)
    n_hw = np.linspace(0, 10, in_dim)
    n_exp_hw = np.square(n_hw)

    # Generate random v0
    if not constant:
        n_v0 = np.random.normal(0.5, 0.1, in_dim)
    else:
        n_v0 = np.full(in_dim, 0.1)
        
    # Generate random alpha
    if not constant:
        n_alpha = np.random.normal(5.0, math.sqrt(2.0), in_dim)
    else:
        n_alpha = np.full(in_dim, 2.5)

    hw = torch.tensor(n_hw).float()
    v0 = torch.tensor(n_v0).float()
    alpha = torch.tensor(n_alpha).float()
    exp_hw = torch.tensor(n_exp_hw).float()

    train_window = 10
    data = [((exp_hw[i:i+train_window].reshape(-1, 1, 1), hw[i:i+train_window].reshape(-1, 1, 1)), (exp_hw[i+train_window:i+train_window+1].reshape(1, -1))) for i in range(in_dim - train_window)]

    # plt.plot(right_hand_side, left_hand_side)
    # plt.show()

    return data