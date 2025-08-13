# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 補助関数 ---
def knn_coords(x, k):
    """
    Input:
        x: (B, 3, N)  # 3次元座標のみを期待
        k: int
    Return:
        idx: (B, N, k)  # 各点のk近傍点のインデックス
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx

def knn_features(x, k):
    """
    Input:
        x: (B, C_features, N)  # 任意のC_features次元特徴量
        k: int
    Return:
        idx: (B, N, k)  # 各点のk近傍点のインデックス
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Input:
        x: (B, C, N)  # B:バッチサイズ, C:特徴量次元 (ここでは4), N:点数
        k: int
        idx: (B, N, k)  # Precomputed k-NN indices (optional)
    Return:
        feature: (B, 2*C', N, k) # C'はエッジ特徴量結合後の次元 (ここでは入力Cと同じ4)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    num_features = x.size(1) # C = 4 (x, y, z, intensity)

    # 1. k-NNの計算（座標のみを使用）
    x_coords = x[:, :3, :] # 最初の3次元 (x, y, z) のみを取得

    if idx is None:
        idx = knn_coords(x_coords, k=k) # (batch_size, num_points, k)
        
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x.transpose(2, 1).contiguous().view(-1, num_features) 
    
    neighbor = x_flat[idx, :] 
    neighbor = neighbor.view(batch_size, num_points, k, num_features) 
    x = x.transpose(2, 1).contiguous().view(batch_size, num_points, 1, num_features)

    feature = torch.cat([x.expand_as(neighbor), neighbor - x], dim=3)
    feature = feature.permute(0, 3, 1, 2).contiguous() 

    return feature

def get_graph_feature_generic(x, k=20, idx=None):
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
        idx = knn_features(x, k=k)
        
    device = x.device
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

# --- DGCNNLocalFeatureExtractor モデル ---
class DGCNNLocalFeatureExtractor(nn.Module):
    def __init__(self, k=20, emb_dims=512, projection_dim=128):
        super(DGCNNLocalFeatureExtractor, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.projection_dim = projection_dim

        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(self.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        # プロジェクションヘッド
        self.projection_head = nn.Sequential(
            nn.Linear(self.emb_dims, self.emb_dims // 2, bias=False),
            nn.BatchNorm1d(self.emb_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dims // 2, self.projection_dim, bias=False)
        )

    def forward(self, x):
        # x: (B, N, 4) -> (B, 4, N)
        x = x.permute(0, 2, 1) 

        batch_size = x.size(0)

        x_initial_features = x
        
        x = get_graph_feature(x_initial_features, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        local_features = torch.cat((x1, x2, x3), dim=1)
        local_features = self.conv4(local_features) # (B, emb_dims, N)

        # プロジェクションヘッドの適用
        B, C_feats, N = local_features.shape
        projected_features = self.projection_head(local_features.view(B * N, C_feats))
        projected_features = projected_features.view(B, N, self.projection_dim)

        # 訓練時と推論時で返す値を切り替える
        if self.training:
            return projected_features # (B, N, projection_dim) - コントラスティブ学習用
        else:
            return local_features.permute(0, 2, 1) # (B, N, emb_dims) - 特徴量抽出用 (N, C) 形式に戻す