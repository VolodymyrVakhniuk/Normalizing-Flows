import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_results_flow_2D(dataset, transformed_dataset, model, device):
    model.eval()
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    
    # 1) Plotting the data distribution over x
    dataset = dataset.cpu().detach().numpy()
    ax[0].scatter(*(dataset.T), alpha=0.5, color='red', edgecolor='white', s=40)
    ax[0].set_title("True (unknown) data distribution over x")
    x_start, x_end = ax[0].get_xlim()
    y_start, y_end = ax[0].get_ylim()

    # 2) Plotting the transformed data distribution over z
    transformed_dataset = transformed_dataset.cpu().detach().numpy()
    ax[1].scatter(*(transformed_dataset.T), alpha=0.5, color='red', edgecolor='white', s=40)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Indiced empirical distribution over z")

    # 3) Plotting the learned input distribution

    # -> Create a meshgrid for the domain in torch
    x = torch.linspace(x_start, x_end, 100)
    y = torch.linspace(y_start, y_end, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # -> Flatten X and Y to create a batch of (x, y) pairs
    X_flat = X.flatten().unsqueeze(dim=1)
    Y_flat = Y.flatten().unsqueeze(dim=1)
    XY_batch = torch.cat([X_flat, Y_flat], dim=1)  # Creates tensor of size (batch, 2)

    # -> Computing the values of density function
    XY_batch = XY_batch.to(device)
    Z = model.log_likelihood(XY_batch)

    # -> Prepare for plotting
    x = x.numpy()
    y = y.numpy()
    Z_grid = Z.view(X.size()).cpu().detach().numpy()

    contourf_output = ax[2].contourf(x, y, np.exp(Z_grid.T), levels=20, cmap='viridis')
    fig.colorbar(contourf_output, ax=ax[2])
    ax[2].set_title("Learned data distribution over x")

    fig.suptitle("2D Flow to Uniform distribution")
    plt.show()

    model.train()
