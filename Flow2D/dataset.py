import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll


def sample_swiss_roll(num_samples, noise=1.0):
    # Sample 3D swiss roll and return only 2D coords
    x, _= make_swiss_roll(num_samples, noise=noise)
    x = torch.tensor(x[:, [0, 2]] / 10.0, dtype=torch.float32)
    return x


class SwissRollDataset(Dataset):
    def __init__(self, num_samples):
        self.data = sample_swiss_roll(num_samples=num_samples)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
