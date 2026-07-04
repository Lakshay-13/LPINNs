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
from neurodiffeq.conditions import IVP, DirichletBVP, DirichletBVP2D
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.networks import FCNN, SinActv
from neurodiffeq.monitors import Monitor1D
from neurodiffeq.generators import Generator1D, Generator2D

def render(solver, name=None):
    # Extract data from solver
    history = solver.metrics_history
    
    x, y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))
    u_pred = solver.get_solution(best=True)(x, y, to_numpy = True)
    u_true = he(x, y)
    res = solver.get_residuals(x, y, to_numpy=True)


    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Plot the loss subplot
    ax1.plot(np.log10(history['train_loss']), label='Train', color=colors[0])
    ax1.set_title('Loss plot')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot the results subplot
    ax2.plot_surface(x, y, u_pred,cmap=plt.cm.Reds, edgecolors='k', linewidths=0.5, alpha= 1.0)
    ax2.plot_surface(x, y, u_true,cmap=plt.cm.Greens, edgecolors='k', linewidths=0.5, alpha = 0.5)
    # ax2.plot_surface(x, y, u_true-u_pred,cmap=plt.cm.Greens, edgecolors='k', linewidths=0.5, alpha= 0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('T')
    ax2.set_zlabel('Residuals^2')
    #ax.set_title('Squared residuals of the heat equation halfway through training')

    ax2.set_xlabel('x')
    ax2.set_ylabel('T')
    ax2.set_zlabel('u')

    ax3.plot_surface(x, y, res,cmap=plt.cm.Purples, edgecolors='k', linewidths=0.5, alpha= 0.7)
    ax3.set_xlabel('X')
    ax3.set_ylabel('T')
    ax3.set_zlabel('Residuals^2')


    # Adjust layout and display the plot
    plt.tight_layout()
    if name:
        plt.savefig(fr'graphs\{name}', transparent=True, dpi = 500)
    plt.show()

class SineActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)
    
class CustomNN(nn.Module):
    def __init__(self, n_input_units=2, hidden_units=[16,16], actv=SineActivation, n_output_units=1, t_min=0, t_max=1, mu_learnable=False,
                  sigma_learnable=True, initial_sigma=0.06):
        super(CustomNN, self).__init__()

        
        self.layers = nn.ModuleList()
        self.heads = nn.ModuleList()
        head_neurons = int(hidden_units[0]/n_input_units)
        if mu_learnable:
            self.mus = nn.Parameter(torch.stack([torch.linspace(t_min, t_max, head_neurons)]*n_input_units))
        else:
            self.mus = torch.stack([torch.linspace(t_min, t_max, head_neurons)]*n_input_units)
        if sigma_learnable:
            self.sigmas = nn.Parameter(torch.stack([nn.Parameter(torch.ones(head_neurons)*(initial_sigma))]*n_input_units))
        else:
            self.sigmas = torch.stack([torch.ones(head_neurons)*(initial_sigma)]*n_input_units)
        for i in range(n_input_units):
            self.heads.append(nn.Linear(1, head_neurons))
        self.layers.append(nn.Linear(int(head_neurons*n_input_units), hidden_units[0]))
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
        self.output = nn.Linear(hidden_units[-1], n_output_units)
        self.act = actv()

    def forward(self, x):
        head_out = []
        for i in range(len(self.heads)):
            temp = self.heads[i](x[:,i].view(-1, 1))
            temp = temp * torch.exp(- (temp - self.mus[i]) ** 2 / (2 * self.sigmas[i] ** 2))
            temp = self.act(temp)
            head_out.append(temp)
        head_out = torch.cat(head_out, dim=1)
        x = self.layers[0](head_out)
        x = self.act(x)
        del temp, head_out
        for i in range(1, len(self.layers)-1):
            x = self.layers[i](x)
            x = self.act(x)
        x = self.layers[-1](x)
        x = self.act(x)
        x = self.output(x)
        return x

k = 2

def he(x, y):
    return np.sin(np.pi*x)*np.exp(-np.pi**2*k*y)
def heat_equation(u, x, t):
    return [diff(u, t) - k*diff(u, x, order=2)]

class HE():
    def __init__(self, input_neurons=[32,32], lr = 1e-2, activation = nn.Tanh, mu=False, sigma = True, initial_sigma=None, localization=True):
        
        init_val = [
            DirichletBVP2D(
                x_min=0, x_min_val=lambda y: 0,
                x_max=1, x_max_val=lambda y: 0,                   
                y_min=0, y_min_val=lambda x: torch.sin(np.pi*x),                   
                y_max=1, y_max_val=lambda x: 0,                   
            )
        ]
        if localization:
            nets = [CustomNN(n_input_units=2, n_output_units=1, hidden_units=input_neurons, actv=activation, mu_learnable=mu, sigma_learnable=sigma, initial_sigma=initial_sigma)]
        else:
            nets = [FCNN(n_input_units=2, n_output_units=1, hidden_units=input_neurons, actv=activation)]
        xy_min = (0, 0)
        xy_max = (1, 1)
        train_g = Generator2D((32, 32), xy_min, xy_max , method='equally-spaced-noisy') 
        valid_g = Generator2D((8, 8), xy_min, xy_max , method='equally-spaced')
        optimizer = torch.optim.Adam([p for net in nets for p in net.parameters()],lr = lr)
        # Set up the solver
        self.solver = Solver2D(pde_system = heat_equation, 
                        conditions = init_val, 
                        xy_min=xy_min, 
                        xy_max=xy_max, 
                        nets=nets,
                        train_generator=train_g, 
                        valid_generator=valid_g,
                        n_batches_valid = 0,
                        optimizer = optimizer)

    def train(self, epochs=100, name=None):
        self.solver.fit(max_epochs=epochs)
        render(self.solver, name=name)