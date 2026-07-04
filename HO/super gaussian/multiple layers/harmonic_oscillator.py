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

import math
import torch
import numpy as np
import torch.nn as nn
from neurodiffeq import diff      
from neurodiffeq.ode import solve 
from neurodiffeq.conditions import IVP 
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import Solver1D
from neurodiffeq.generators import Generator1D

def render(solver, name=None):
    # Extract data from solver
    history = solver.metrics_history
    nets = solver.best_nets[0]
    flag = False
    try:
        mus = nets.mus
        sigmas = nets.sigmas
        flag=True
    except Exception as e:
        pass
    
    t_vals = np.linspace(solver.t_min, solver.t_max, 1000)
    u_pred = solver.get_solution()(t_vals, to_numpy=True)
    residuals = solver.get_residuals(t_vals, to_numpy=True)

    if flag:
        layers = len(solver.best_nets[0].layers) - 1
        dim = 2 + math.ceil(layers/2)
        fig, axs = plt.subplots(dim, 2, figsize=(12, dim*4))

    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the loss subplot
    axs[0][0].plot(np.log10(history['train_loss']), label='Train', color=colors[0])
    axs[0][0].set_title('Loss plot')
    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Loss')
    axs[0][0].grid(True)
    axs[0][0].legend()

    # Plot the results subplot
    axs[0][1].plot(t_vals, u_pred, color=colors[0])
    axs[0][1].set_title(f'Neural Network approximation at {len(history["train_loss"])} epochs.')
    axs[0][1].set_ylabel('u')
    axs[0][1].set_xlabel('t')
    axs[0][1].grid(True)

    if flag:
        for i in range(len(solver.best_nets[0].mus)-1):
            axs[1][0].plot(solver.best_nets[0].mus[i].cpu().detach().numpy(), solver.best_nets[0].sigmas[i].cpu().detach().numpy(), marker='o', label = f'Layer {i+1}')
        axs[1][0].set_title(f'Mus and Sigmas')
        axs[1][0].set_xlabel('Mus')
        axs[1][0].set_ylabel('Sigmas')
        axs[1][0].legend()
        axs[1][0].grid(True)

    axs[1][1].plot(t_vals, residuals, color=colors[2])
    axs[1][1].set_title(f'Residuals')
    axs[1][1].set_xlabel('t')
    axs[1][1].set_ylabel('Residuals')
    axs[1][1].grid(True)

    t_vals = torch.linspace(solver.t_min, solver.t_max, 256, requires_grad = True)
    residuals = solver.get_residuals(t_vals)
    derivatives = []

    if not flag:
        for res in residuals:
            derivatives.append(diff(res**2, solver.best_nets[0].NN[0].weight,shape_check=False))
        derivatives = torch.stack(derivatives).cpu().detach().numpy()
        heatmap = np.ones((256, 256))/1000
        for i in range(derivatives.shape[1]):
            heatmap += np.abs(derivatives[:,i,0].reshape(-1,1)*derivatives[:,i,0].reshape(1,-1))
        heatmap = np.log(heatmap/np.max(heatmap))
        # cmap = ListedColormap(['black', 'lavender','green'])
        # bounds = [-50, -6, -0.5, 0]
        # norm = BoundaryNorm(bounds, cmap.N)
        # h1 = axs[1][0].imshow(heatmap, cmap=cmap, norm=norm)
        # fig.colorbar(h1, ax=axs[1][0])
        sns.heatmap(heatmap, ax = axs[1][0], xticklabels=False, yticklabels=False)
        axs[1][0].set_title(f'Gradients Heatmap')
        axs[1][0].set_xlabel('t')
        axs[1][0].set_ylabel("$t^'$")
        axs[1][0].set_xticks([])
        axs[1][0].set_yticks([])
        del derivatives
    
    else:
        count = 0
        for i in solver.best_nets[0].layers:
            derivatives = []
            if count==layers:
                break
            for res in residuals:
                derivatives.append(diff(res**2, i.weight,shape_check=False))
            count += 1
            derivatives = torch.stack(derivatives).cpu().detach().numpy()
            heatmap = np.ones((256, 256))/1000
            for j in range(derivatives.shape[1]):
                heatmap += np.abs(derivatives[:,j,0].reshape(-1,1)*derivatives[:,j,0].reshape(1,-1))
            heatmap = np.log(heatmap/np.max(heatmap))
            # cmap = ListedColormap(['black', 'lavender','green'])
            # bounds = [-50, -6, -0.5, 0]
            # norm = BoundaryNorm(bounds, cmap.N)
            # h1 = axs[1][0].imshow(heatmap, cmap=cmap, norm=norm)
            # fig.colorbar(h1, ax=axs[1][0])
            x = math.ceil(count/2) + 1
            y = 1 if (count%2==0) else 0
            sns.heatmap(heatmap, ax = axs[x][y], xticklabels=False, yticklabels=False)
            axs[x][y].set_title(f'Gradients Heatmap layer {count}')
            axs[x][y].set_xlabel('t')
            axs[x][y].set_ylabel("$t^'$")
            axs[x][y].set_xticks([])
            axs[x][y].set_yticks([])
            del derivatives

    # Adjust layout and display the plot
    plt.tight_layout()
    if name:
        plt.savefig(fr'graphs\{name}', transparent=True, dpi = 500)
    plt.show()

class SineActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)
    
class CustomNN(nn.Module):
    def __init__(self, n_input_units=1, hidden_units=[16,16], actv=SineActivation, n_output_units=1, t_min=0, t_max=1,
                  sigma_learnable=True, initial_sigma=0.06):
        super(CustomNN, self).__init__()

        self.mus = []
        self.sigmas = nn.ParameterList() if sigma_learnable else []
        for i in range(len(hidden_units)):
            self.mus.append(torch.linspace(t_min, t_max, hidden_units[i]))
            if sigma_learnable:
                self.sigmas.append(nn.Parameter(torch.ones(hidden_units[i])*(0.06)*hidden_units[0]/hidden_units[i]))
            else:
                self.sigmas.append(torch.ones(hidden_units[i])*(0.06)*hidden_units[0]/hidden_units[i])

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
        self.output = nn.Linear(hidden_units[-1], n_output_units)
        self.act = actv()

    def forward(self, x):
        for i in range(len(self.layers)-1):
            if i == 0:
                out = self.layers[i](x)
            else:
                out = self.layers[i](out)
            out = self.act(out)
            out = out * torch.exp(- (x - self.mus[i]) ** 20 / (2 * self.sigmas[i] ** 20))
        out = self.layers[i+1](out)
        out = self.act(out)
        out = self.output(out)
        return out

def harmonic_oscillator(u, t):
    return [diff(u, t, order=2) + torch.sin(u)]
def scaled_harmonic_oscillator(u, t):
    return [diff(u, t, order=2) + 16*torch.pi**2*torch.sin(u)]
class HO():
    def __init__(self, t_0=0.0, u_0=0.0, u_0_prime = 1.0, t_min=0.0, t_max=4*np.pi, 
                 train_batch_size=256, valid_batch_size=64, input_neurons=[32,32], lr = 1e-2, activation = SineActivation, 
                 sigma = True, initial_sigma=None, localization=True, scaling = True):
        init_val_ho = [IVP(t_0=t_0, u_0=u_0, u_0_prime=u_0_prime)]
        if scaling:
            t_min, t_max = t_min/t_max, t_max/t_max
            if localization:
                nets = [CustomNN(n_input_units=1, n_output_units=1, hidden_units=input_neurons, actv=activation, t_min=t_min, t_max=t_max, 
                                 sigma_learnable=sigma, initial_sigma=initial_sigma)]
            else:
                nets = [FCNN(n_input_units=1, n_output_units=1, actv = activation, hidden_units=input_neurons)]
            train_g = Generator1D(train_batch_size, t_min, t_max , method='equally-spaced-noisy')
            valid_g = Generator1D(valid_batch_size, t_min, t_max , method='uniform')
            optimizer = torch.optim.Adam([p for net in nets for p in net.parameters()],lr = lr)
            self.solver = Solver1D(ode_system=scaled_harmonic_oscillator, 
                        conditions=init_val_ho, 
                        t_min=t_min, 
                        t_max=t_max,
                        n_batches_valid=0,
                        train_generator=train_g,
                        valid_generator=valid_g,
                        nets = nets,
                        optimizer=optimizer)
        else:
            if localization:
                nets = [CustomNN(n_input_units=1, n_output_units=1, hidden_units=input_neurons, actv=activation, t_min=t_min, t_max=t_max, 
                                 sigma_learnable=sigma, initial_sigma=initial_sigma)]
            else:
                nets = [FCNN(n_input_units=1, n_output_units=1, actv = activation, hidden_units=input_neurons)]
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
        
    def train(self, epochs=100, name=None):
        self.solver.fit(max_epochs=epochs)
        render(self.solver, name=name)