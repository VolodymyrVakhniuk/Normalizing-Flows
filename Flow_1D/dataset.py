import torch
from torch.utils.data import Dataset

def sample_gaussian_mixture(mix_coefs, means, stds, num_samples):
    # Sample gaussian clusters from the multinomial
    clusters = torch.multinomial(mix_coefs, num_samples, replacement=True)
    samples = torch.normal(means[clusters], stds[clusters])
    
    return samples


class GaussianMixtureDataset(Dataset):
    def __init__(self, mix_coefs, means, stds, num_samples):
        self.data = sample_gaussian_mixture(mix_coefs, means, stds, num_samples=num_samples).view(-1, 1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    