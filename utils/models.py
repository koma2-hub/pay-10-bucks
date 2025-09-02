
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

class SubNetwork(nn.Module):
    def __init__(self, input_dim=32, embedding_dim=64):
        super(SubNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, x):
        return self.fc(x)

class SiameseNetwork(nn.Module):
    def __init__(self, sub_network):
        super(SiameseNetwork, self).__init__()
        self.sub_network = sub_network
    def forward(self, input1, input2):
        output1 = self.sub_network(input1)
        output2 = self.sub_network(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class kICNDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.hist1 = torch.from_numpy(data['hist1'].astype(np.float32))
        self.hist2 = torch.from_numpy(data['hist2'].astype(np.float32))
        self.labels = torch.from_numpy(data['labels'].astype(np.float32))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.hist1[idx], self.hist2[idx], self.labels[idx]
