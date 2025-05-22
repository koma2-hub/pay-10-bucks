import torch
from torch_geometric.nn import max_pool_x
from torch_geometric.nn import knn
import torch.nn as nn
from utils.data_utils import load_ply

class EdgeConv(nn.Module):
    def __init__(self):
        super(EdgeConv, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(2, 64), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope = 0.2)
            )
    
    def forward(self, pcd):
        assign_index = knn(x = pcd[:, :3], y = pcd[:, :3], k = 20)
        p = assign_index[1, :]  #着目点
        q = assign_index[0, :]  #近傍点
        x = torch.cat([pcd[p][-1],pcd[q][-1] - pcd[p][-1]], dim=1)
        x = self.shared_mlp(x)

        edge_batch = p
        x, _ = max_pool_x(cluster= assign_index[1, :], x = x, batch=edge_batch)
        return x 
    
src = load_ply("./data/src1.ply")
src = torch.from_numpy(src)
print(src.shape[0])