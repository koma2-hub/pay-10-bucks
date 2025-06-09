import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from data_utils import load_ply

class PointCloudDataset(Dataset):
    def __init__(self, data):
        #constract data
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx]}
        return sample
    
def list_files(directory):
    return os.listdir(directory)


def get_PCL(filepath):
    files = os.listdir(filepath)
    PCLData = []
    PCLData = [load_ply(filepath + file) for file in files]

    PCLData = torch.tensor(PCLData)
    return PCLData


data = get_PCL('./data/')
print(data.size())

pcl_dataset = PointCloudDataset(data)
#pytorch_geometricのデータ型
data_loader = DataLoader(dataset = pcl_dataset, batch_size = 2, shuffle = True)

for batch in data_loader:
    x = batch.get('data')
    x = x[:, :, :3]
    print(x.size())
    x = x.transpose(2,1)
    print(x.size())
    
    k = 5
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    print(idx)
'''
for batch in data_loader:
    x = batch[:, :, :3]
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim = 1, keepdim = True)
    pairwise_distance = -xx - inner -xx.transpose(2, 1)
    idx = pairwise_distance.topk(k = 10, dim = -1)[1]
    print(idx)
'''

