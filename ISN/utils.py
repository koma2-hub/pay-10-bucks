#utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 補助関数 ---
def knn(x, k):
    """
    Input:
        x: (B, C, N)
        k: int
    Return:
        idx: (B, N, k)  # 各点のk近傍点のインデックス
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx

def edge_feature(x, k=32, idx=None):
    """
    Input:
        x: (B, C, N)  # B:バッチサイズ, C:特徴量次元, N:点数
        k: int
        idx: (B, N, k)  # Precomputed k-NN indices (optional)
    Return:
        feature: (B, 2*C, N, k) # Cは入力xの次元
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    num_dims = x.size(1) # C
    
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx = idx.to(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x.transpose(2, 1).contiguous().view(-1, num_dims) 
    neighbor = x_flat[idx, :] 
    neighbor = neighbor.view(batch_size, num_points, k, num_dims) 
    x = x.transpose(2, 1).contiguous().view(batch_size, num_points, 1, num_dims)

    feature = torch.cat([x.expand_as(neighbor), neighbor - x], dim=3)
    feature = feature.permute(0, 3, 1, 2).contiguous() 

    return feature