# data.py
import numpy as np
import torch
from torch.utils.data import Dataset

class PriceDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.cls = (self.y[:, -1, :] - self.x[:, -1, :]) / self.x[:, -1, :] > 0.02
        self.cls = self.cls.float()

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].view(-1), self.cls[idx]
