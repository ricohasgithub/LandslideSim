
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt

from torchdiffeq import odeint
from generator import gen_lin_dataset

class Feedforward_NN(nn.Module):

    def __init__(self):
        super(Feedforward_NN, self).__init__()
        self.linear1 = nn.Linear(1, 4)
        self.hidden = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 1)
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        output = self.linear1(x)
        output = self.hidden(output)
        output = self.nonlinear(self.linear2(output))
        return output

def ff_train(model, data_gen, epochs):

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

    test_data_gen = gen_lin_dataset(100)
    test_examples_hw = test_data_gen[0]
    test_left_hand_side = test_data_gen[2]

    test_examples = torch.stack((test_examples_hw, test_left_hand_side))
    test_examples = torch.chunk(test_examples, 1000, dim=1)
    test_examples = torch.stack(test_examples)

    for i, (test_example, test_label) in enumerate(test_examples):
        print("Prediction: ", model(test_example.float()))
        print("Label: ", test_label)

    return loss_history

'''

    Linear Model: Neural ODE with time axis swapped for hw variable

    u_t = -u + C * exp(u)
    u_t -> time derivative
    C -> constant

    ODE-RNN Network
        code licensed from: https://github.com/rtqichen/torchdiffeq

    @article{chen2018neuralode,
        title={Neural Ordinary Differential Equations},
        author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
        journal={Advances in Neural Information Processing Systems},
        year={2018}
    }

'''

class ODE_Func(nn.Module):

    def __init__(self, hidden_dim):
        super(ODE_Func, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        out = self.nonlinear(self.linear(x))
        return out

class ODE_RNN(nn.Module):

    def  __init__(self, input_dim, hidden_dim, output_dim):
        super(ODE_RNN, self).__init__()

        # Initialize initial hidden state
        self.h_0 = torch.zeros(hidden_dim, 1)
        self.hidden_dim = hidden_dim

        # Model layers: input -> hidden -> output
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

        self.ode_func = ODE_Func(hidden_dim)
        self.nonlinear = nn.Tanh()

    def forward(self, t, x):
        
        t = t.reshape(-1).float()
        h = torch.zeros(self.hidden_dim, 1)
        h_i = torch.zeros(1, self.hidden_dim)

        # RNN iteration
        for i, x_i in enumerate(x):
            x_i = torch.reshape(x_i.float(), (1, 1))
            if i > 0:
                h_i = odeint(self.ode_func, h, t[i-1 : i+1], method="euler")[1]
            h = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))

        out = self.decoder(h)
        return out
        
def ode_train(model, data, epochs):

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):

        random.shuffle(data)
        epoch_loss = []
        
        for i, (example, label) in enumerate(data):

            optimizer.zero_grad()
            prediction = model(example[0], example[1])

            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.detach().numpy())

        loss_history.append(sum(epoch_loss) / len(data))
        epoch_loss = []

        print('Epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_history[epoch]))

        for i, (example, label) in enumerate(data):
            print("Label: ", label)
            print("Prediction: ", model(example[0], example[1]))
            break
        
    return loss_history

'''

    Fourier Neural Operator
        code licensed from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
    
    @misc{li2020fourier,
        title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
        author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
        year={2020},
        eprint={2010.08895},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

'''

class Spectral_Conv_1D(nn.Module):

    def __init__(self, input_dim, output_dim, modes):

        super(Spectral_Conv_1D, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes

        self.scale = (1 / (self.input_dim * self.output_dim))
        self.weights = nn.Parameter(self.scale * torch.rand(self.input_dim, self.output_dim, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):

        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.output_dim, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class Fourier_Neural_Operator_1D(nn.Module):

    def __init__(self, modes, width):

        super(Fourier_Neural_Operator_1D, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = Spectral_Conv_1D(self.width, self.width, self.modes1)
        self.conv1 = Spectral_Conv_1D(self.width, self.width, self.modes1)
        self.conv2 = Spectral_Conv_1D(self.width, self.width, self.modes1)
        self.conv3 = Spectral_Conv_1D(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)