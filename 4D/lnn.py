# Graph aesthetics settings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from cycler import cycler
import seaborn as sns
large = 20; medium = 16; small = 12
colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']
text_color = "#0afa9e"

params = {'axes.titlesize': medium,
          'legend.fontsize': small,
          'figure.figsize': (8,5),
          'axes.labelsize': small,
          'axes.linewidth': 2,
          'xtick.labelsize': small,
          'xtick.color': text_color,
          'ytick.color': text_color,
          'ytick.labelsize': small,
          'axes.edgecolor': text_color,
          'figure.titlesize': small,
          'axes.prop_cycle': cycler(color=colors),
          'axes.titlecolor': text_color,
          'axes.labelcolor': text_color,
         }

plt.rcParams.update(params)
plt.style.use('dark_background')

# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch.nn as nn
from neurodiffeq import diff

class CustomNN(nn.Module):
    def __init__(self, n_input_units=2, hidden_units=[16,16], actv=nn.Tanh, n_output_units=1, t_min=0, t_max=1,
                  sigma_learnable=True, initial_sigma=0.06, mu_learnable=True, first=True):
        super(CustomNN, self).__init__()

        self.f = first
        self.mus = nn.ParameterList() if mu_learnable else []
        self.n = n_input_units
        self.sigmas = nn.ParameterList() if sigma_learnable else []
        for i in range(len(hidden_units)):
            if mu_learnable:
                self.mus.append(torch.stack([nn.Parameter(torch.linspace(t_min, t_max, hidden_units[i]))]*n_input_units))
            else:
                self.mus.append(torch.stack([torch.linspace(t_min, t_max, hidden_units[i])]*n_input_units))
            if sigma_learnable:
                self.sigmas.append(torch.stack([nn.Parameter(torch.ones(hidden_units[i])*(initial_sigma)*hidden_units[0]/hidden_units[i])]*n_input_units))
            else:
                self.sigmas.append(torch.stack([torch.ones(hidden_units[i])*(initial_sigma)*hidden_units[0]/hidden_units[i]]*n_input_units))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
        self.output = nn.Linear(hidden_units[-1], n_output_units)
        self.act = actv()

    def forward(self, x):
        for i in range(len(self.layers)-1):
            gaussian = 1
            if i == 0:
                out = self.layers[i](x)
                out = self.act(out)
                for j in range(self.n):
                    gaussian = gaussian * torch.exp(- (x[:,j].view(-1, 1) - self.mus[i][j]) ** 2 / (2 * self.sigmas[i][j] ** 2)) 
                out = out*gaussian
            else:
                out = self.layers[i](out)
                if not self.f:
                    out = self.act(out)
                    for j in range(self.n):
                        gaussian = gaussian * torch.exp(- (x[:,j].view(-1, 1) - self.mus[i][j]) ** 2 / (2 * self.sigmas[i][j] ** 2)) 
                    out = out*gaussian
        out = self.layers[i+1](out)
        out = self.act(out)
        out = self.output(out)
        return out
