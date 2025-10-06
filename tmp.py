import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import fpsample
from utils.data_utils import load_ply , knn

class ISNDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pcd1 = torch.from_numpy(data['pcd1'].astype(np.float32))
        self.pcd2 = torch.from_numpy(data['pcd2'].astype(np.float32))
        self.labels = torch.from_numpy(data['labels'].astype(np.float32))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.pcd1[idx], self.pcd2[idx], self.labels[idx]

full_dataset = ISNDataset('/mnt/c/Users/matsu/SICK/pay-10-bucks/ISN/dataset/pcd_32points_noise005_transformed_2064.npz')

test_dataloader = DataLoader(full_dataset, batch_size = 32, shuffle=True)

for pcd1, pcd2, label in test_dataloader:
    print(pcd1.size())
    print(pcd2.size())
    print(label.size())