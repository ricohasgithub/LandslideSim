
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as plt

from generator import *
from network import *

if __name__ == "__main__":

    # Linear: get dataset for training
    # data_gen = gen_lin_dataset(1000)
    # feedforward_nn = Feedforward_NN()
    # ff_train(feedforward_nn, data_gen, 30)

    # Nonlinear: use ODE-RNN model with autogen
    data_gen = gen_exp_dataset(1000)
    ode_rnn = ODE_RNN(1, 4, 1)
    ode_train(ode_rnn, data_gen, 30)