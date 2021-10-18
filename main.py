
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as plt

from generator import *
from network import *

if __name__ == "__main__":

    # Get dataset for training
    data_gen = gen_lin_dataset(1000)
    feedforward_nn = Feedforward_NN()
    ff_train(feedforward_nn, data_gen, 30)