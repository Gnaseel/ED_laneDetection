import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, np_file_paths):
        self.files = np_file_paths
    
    def __getitem__(self, index):
        x = np.load(self.files, allow_pickle=True)
        x = torch.from_numpy(x).float()
        return x
    
    def __len__(self):
        return len(self.files)