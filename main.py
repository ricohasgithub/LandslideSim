
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib as plt

from generator import *
from network import *

def train(model, data_gen, epochs):

    examples_hw = data_gen[0]
    left_hand_side = data_gen[2]

    examples = torch.stack((examples_hw, left_hand_side))
    examples = torch.chunk(examples, 1000, dim=1)
    examples = torch.stack(examples)

    optimizer = optim.Adam(model.parameters(), lr=0.05)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle(examples)
        epoch_loss = []
        
        for i, (example, label) in enumerate(examples):

            optimizer.zero_grad()
            prediction = model(example.float())

            loss = loss_function(prediction, label.float())
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.detach().numpy())

        loss_history.append(sum(epoch_loss) / len(examples))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

    test_data_gen = gen_dataset(100)
    test_examples_hw = test_data_gen[0]
    test_left_hand_side = test_data_gen[2]

    test_examples = torch.stack((test_examples_hw, test_left_hand_side))
    test_examples = torch.chunk(test_examples, 1000, dim=1)
    test_examples = torch.stack(test_examples)

    for i, (test_example, test_label) in enumerate(test_examples):
        print("Prediction: ", model(test_example.float()))
        print("Label: ", test_label)

    return loss_history

if __name__ == "__main__":

    # Get dataset for training
    data_gen = gen_dataset(1000)
    feedforward_nn = Feedforward_NN()
    train(feedforward_nn, data_gen, 30)