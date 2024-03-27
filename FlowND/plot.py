import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

def plot_results_NICE(model):
    model.eval()

    # 1) Sampling the model
    image_samples = model.sample(num_samples=64).cpu()

    # 2) Making a grid out of samples 
    grid = make_grid(image_samples)

    # 3) Displaying the grid
    grid = grid.numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # Convert from Tensor image
    plt.show()

    model.train()
    