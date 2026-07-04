# Graph aesthetics settings
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
large = 20; medium = 16; small = 12
colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']
text_color = "#404040"

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

import torch
import numpy as np
import torch.nn as nn
from neurodiffeq import diff      
from neurodiffeq.ode import solve 
from neurodiffeq.conditions import IVP 
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import Solver1D
from neurodiffeq.generators import Generator1D

def render(solver):
    # Extract data from solver
    history = solver.metrics_history
    nets = solver.best_nets[0]
    mus = nets.mu.cpu().detach().numpy()
    sigmas = nets.sigma.cpu().detach().numpy()
    
    t_vals = np.linspace(solver.t_min, solver.t_max, 1000)
    u_pred = solver.get_solution()(t_vals, to_numpy=True)
    residuals = solver.get_residuals(t_vals, to_numpy=True)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the loss subplot
    axs[0][0].plot(np.log10(history['train_loss']), label='Train', color=colors[0])
    axs[0][0].set_title('Loss plot')
    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Loss')
    axs[0][0].grid(True)
    axs[0][0].legend()

    # Plot the results subplot
    axs[0][1].plot(mus, sigmas, color=colors[1])
    axs[0][1].plot(t_vals, u_pred, color=colors[0])
    axs[0][1].set_title(f'Neural Network approximation at {len(history["train_loss"])} epochs.')
    axs[0][1].set_xlabel('u')
    axs[0][1].set_ylabel('t')
    axs[0][1].grid(True)

    axs[1][1].plot(t_vals, residuals, color=colors[2])
    axs[1][1].set_title(f'Residuals')
    axs[1][1].set_xlabel('t')
    axs[1][1].set_ylabel('Residuals')
    axs[1][1].grid(True)

    t_vals = torch.linspace(solver.t_min, solver.t_max, 256, requires_grad = True)
    residuals = solver.get_residuals(t_vals)
    derivatives = []
    for res in residuals:
        derivatives.append(diff(res**2, solver.best_nets[0].layers[0].weight,shape_check=False).mean())
    derivatives = torch.stack(derivatives).cpu().detach().numpy()
    derivatives = np.abs(derivatives.reshape(-1,1)*derivatives.reshape(1,-1))
    m1, m2 = np.min(derivatives), np.max(derivatives)
    derivatives = (derivatives-m1)/(m2-m1)
    sns.heatmap(derivatives, ax = axs[1][0], xticklabels=False, yticklabels=False)
    axs[1][0].set_title(f'Gradients Heatmap')
    axs[1][0].set_xlabel('Mu')
    axs[1][0].set_ylabel('Sigma')
    axs[1][0].grid(True)
    del derivatives
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

class CustomNN(nn.Module):
    def __init__(self, n_input_units=1, hidden_units=[16,16], actv=nn.Tanh, n_output_units=1, t_min=0, t_max=1, mu_learnable=False, sigma_learnable=True):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))

        # Learnable parameters mu and sigma for the firs layer
        if mu_learnable:
            self.mu = nn.Parameter(torch.linspace(t_min,t_max, hidden_units[0]))
        else:
            self.mu =  torch.linspace(t_min,t_max, hidden_units[0])
        if sigma_learnable:
            self.sigma = nn.Parameter(torch.ones(hidden_units[0])*((t_max-t_min)/hidden_units[0]))
        else:
            self.sigma = torch.ones(hidden_units[0])*((t_max-t_min)/hidden_units[0])

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)

    def forward(self, x):

        #inputx = x[:,0].reshape(-1,1)
        #print(inputx.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(x.shape)
            # Apply the custom operation after the first layer
            if i == 0:
                x = x * torch.exp(- (x - self.mu) ** 2 / 2 * self.sigma ** 2)

        # Output layer transformation
        x = self.fc_out(x)
        return x


def harmonic_oscillator(u, t):
    return [diff(u, t, order=2) + torch.sin(u)]
class HO():
    def __init__(self, t_0=0.0, u_0=0.0, u_0_prime = 1.0, t_min=0.0, t_max=4*np.pi, 
                 train_batch_size=256, valid_batch_size=64, mu=False, sigma=False,
                 input_neurons=[32,32], lr = 1e-2):
        init_val_ho = [IVP(t_0=t_0, u_0=u_0, u_0_prime=u_0_prime)]
        t_min = t_min
        t_max = t_max
        train_batch_size = train_batch_size
        valid_batch_size = valid_batch_size
        mu = mu
        sigma = sigma
        nets = [CustomNN(n_input_units=1, n_output_units=1, hidden_units=input_neurons, t_min=t_min, t_max=t_max, mu_learnable=mu, sigma_learnable=sigma)]
        train_g = Generator1D(train_batch_size, t_min, t_max , method='equally-spaced-noisy')
        valid_g = Generator1D(valid_batch_size, t_min, t_max , method='uniform')
        optimizer = torch.optim.Adam([p for net in nets for p in net.parameters()],lr = lr)
        self.solver = Solver1D(ode_system=harmonic_oscillator, 
                    conditions=init_val_ho, 
                    t_min=t_min, 
                    t_max=t_max,
                    n_batches_valid=0,
                    train_generator=train_g,
                    valid_generator=valid_g,
                    nets = nets,
                    optimizer=optimizer)
        
    def train(self, epochs=100):
        self.solver.fit(max_epochs=epochs)
        render(self.solver)