import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_results_flow_1D(dataset, means, stds, mixing_coefs, transformed_dataset, alpha, beta, model, device):
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    
    # Plotting the data distribution over x
    ax[0].hist(dataset.cpu().detach().numpy(), bins=70)
    ax[0].set_title("True (unknown) data distribution over x")

    # Plotting the graph of the flow from x to z
    x = torch.linspace(start=-5, end=5, steps=500).unsqueeze(1).to(device)
    y = model(x)
    ax[1].plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    ax[1].set_title("Flow from x to z")

    # Plotting the transformed data distribution over z
    ax[2].hist(transformed_dataset.cpu().detach().numpy(), bins=70)
    ax[2].set_title("Indiced empirical distribution over z")

    # # Plotting the graph of beta distribution
    # x = np.linspace(start=0, stop=1, num=100)
    # y = Beta.pdf(x, alpha, beta)
    # ax[2].plot(x, y, label='Beta PDF', linewidth=2)

    fig.suptitle("1D Flow to Beta distribution")
    plt.show()








# from scipy.stats import norm as Norm
# from scipy.stats import beta as Beta
    

# def plot_histogram(samples, num_bins, title):

#     fig, ax = plt.subplots()

#     ax.hist(samples.cpu().detach().numpy(), bins=num_bins)
#     ax.set_title(title)

#     plt.show()



# def plot_mapping(f, start_x, end_x, title, device):

#     x = torch.linspace(start=start_x, end=end_x, steps=400).to(device)
#     y = f(x)

#     fig, ax = plt.subplots()
    
#     ax.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
#     ax.set_title(title)

#     plt.show()




# def beta_pdf():

#     x = np.linspace(0, 1, 100)
#     y = beta.pdf(x, a, b)





