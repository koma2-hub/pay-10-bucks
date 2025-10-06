#model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import knn, edge_feature

class DGCNNwithIntensity(nn.Module):
    def __init__(self, k=32, embedding_dims=512, output_channels=40):
        super(DGCNNwithIntensity, self).__init__()
        self.k = k
        self.embedding_dims=embedding_dims
        self.output_channel = output_channels

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(embedding_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(16*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(336, self.embedding_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(self.embedding_dims*2, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, output_channels)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = edge_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = edge_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = edge_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv4(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn5(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn6(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x