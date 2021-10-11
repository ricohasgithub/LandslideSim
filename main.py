
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

    examples = torch.stack((examples_hw, left_hand_side))
    print(examples.size())

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle(examples)
        epoch_loss = []
        
        for i, (example, label) in enumerate(examples):

            optimizer.zero_grad()
            prediction = model(example)

            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.detach().numpy())

        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    return loss_history

if __name__ == "__main__":

    # Get dataset for training
    data_gen = gen_dataset(1000)
    feedforward_nn = Feedforward_NN()
    train(feedforward_nn, data_gen, 30)