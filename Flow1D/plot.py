import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_results_flow_1D(dataset, transformed_dataset, model, device):
    model.eval()
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))
    
    # Plotting the data distribution over x
    dataset = dataset.cpu().detach().numpy()
    ax[0].hist(dataset, bins=70)
    ax[0].set_title("True (unknown) data distribution over x")
    x_start, x_end = ax[0].get_xlim()

    # Plotting the graph of the flow from x to z
    x = torch.linspace(start=x_start, end=x_end, steps=200, device=device).unsqueeze(1)
    y = model(x)
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    ax[1].plot(x, y)
    ax[1].set_title("Flow from x to z")

    # Plotting the transformed data distribution over z
    transformed_dataset = transformed_dataset.cpu().detach().numpy()
    ax[2].hist(transformed_dataset, bins=70)
    ax[2].set_xlim(0, 1)
    ax[2].set_title("Indiced empirical distribution over z")

    # Plotting the learned input distribution
    x = torch.linspace(start=x_start, end=x_end, steps=200, device=device).unsqueeze(1)
    y = model.log_likelihood(x)
    x, y = x.squeeze(1).cpu().detach().numpy(), y.squeeze(1).cpu().detach().numpy()
    ax[3].plot(x, np.exp(y), linewidth=2)
    ax[3].set_title("Learned data distribution over x")

    fig.suptitle("1D Flow to Beta distribution")
    plt.show()

    model.train()
