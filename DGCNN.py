import torch
from torch_geometric.nn import max_pool_x
from torch_geometric import knn
import torch.nn as nn

class EdgeConv(nn.Module):
    def __init__(self):
        super(EdgeConv, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope = 0.2)
            )
    
    def forward(self, batch):
        assign_index = knn(x = batch.pos, y = batch.pos, k = 20, batch_x = batch.batch,
                           batch_y = batch.batch)
        p = batch.pos[assign_index[1, :], :]
        q = batch.pos[assign_index[0, :], :]
        x = torch.cat([p, q-p], dim=1)
        x = self.shared_mlp(x)

        edge_batch = batch.batch[assign_index[1, :]]
        x, _ = max_pool_x(cluster= assign_index[1, :], x = x, batch=edge_batch)
        return x 
    
