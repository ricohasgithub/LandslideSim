
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as plt

from generator import *
from network import *

def train(model, data_gen, epochs):

    examples_hw = data_gen[0]
    right_hand_side = data_gen[1]
    left_hand_side = data_gen[2]

    examples = torch.stack(examples_hw, left_hand_side)
    print(examples.size())

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle()

if __name__ == "__main__":

    # Get dataset for training
    data_gen = gen_dataset(1000)
    feedforward_nn = Feedforward_NN()
    train(data_gen, feedforward_nn, 30)