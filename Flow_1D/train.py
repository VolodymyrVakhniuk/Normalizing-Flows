import sys
sys.path.append('/Users/volodymyrvakhniuk/Desktop/Projects/Deep Learning/Flows')

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import utils
from torch.utils.data import DataLoader

from Flow_1D import Flow_1D
from dataset import GaussianMixtureDataset

from util.plot import plot_results_flow_1D

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# Hyperparameters
NUM_EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Data distribution:
means = torch.tensor([-1, 3], dtype=torch.float32)
standard_deviations = torch.tensor([0.5, 1], dtype=torch.float32)
mix_coefs = torch.tensor([0.5, 0.5])

# Generating the dataset
dataset = GaussianMixtureDataset(mix_coefs, means, standard_deviations, num_samples=1024)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
alpha = 2
beta = 2
model = Flow_1D(num_mixture_components=5, alpha=alpha, beta=beta).to(device)

with torch.no_grad():
    model.eval()

    # Data gymnastics
    data = dataset.data.to(device)
    transformed_data = model(dataset.data.to(device))
    
    plot_results_flow_1D(
        data, means, standard_deviations, mix_coefs,
        transformed_data, alpha, beta,
        model, device
    )

    model.train()

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Training
for epoch in range(NUM_EPOCHS):
    avg_loss = []
    for batch_idx, data_points in enumerate(data_loader):
        # Data gymnastics
        data_points = data_points.to(device)

        # Loss
        loss = model.get_loss(data_points)
        avg_loss.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Making an update
        optimizer.step()

    avg_loss = torch.mean(torch.tensor(avg_loss))
    print('Epoch %d\t Loss=%.5f' % (epoch, avg_loss))


with torch.no_grad():
    model.eval()

    # Data gymnastics
    data = dataset.data.to(device)
    transformed_data = model(dataset.data.to(device))
    
    plot_results_flow_1D(
        data, means, standard_deviations, mix_coefs,
        transformed_data, alpha, beta,
        model, device
    )

    model.train()


