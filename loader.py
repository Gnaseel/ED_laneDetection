import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import numpy as np
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

class MyDataset(Dataset):
    def __init__(self, np_file_paths):
        self.files = np_file_paths
    
    def __getitem__(self, index):
        x = np.load(self.files, allow_pickle=True)
        # x = np.load(self.files[index])
        x = torch.from_numpy(x).float()
        return x
    
    def __len__(self):
        return len(self.files)