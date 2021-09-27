
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torchdiffeq import odeint
from generator import gen_dataset

'''
    Linear Model: Neural ODE with time axis swapped for hw variable
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
        h_i = torch.zeros(self.hidden_dim, 1)

        # RNN iteration
        for i, x_i in enumerate(x):
            if i > 0:
                h_i = odeint(self.ode_func, h, t[i-1 : i+1])[1]
            h = self.nonlinear(self.linear_in(x_i) + self.linear_hidden(h_i))

        out = self.decoder(h)
        return out
        
