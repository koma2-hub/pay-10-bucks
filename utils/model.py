import torch
import torch.nn as nn
import torch.nn.functional as F

# --- get_graph_feature 関数 (修正版) ---
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
    # x_coords は (B, 3, N)
    x_coords = x[:, :3, :] # 最初の3次元 (x, y, z) のみを取得

    if idx is None:
        # k-NNは座標のみで行う
        # DGCNNのutil.pyにあるknn関数を想定
        # ここでは、後述するknn_coords関数を使用する
        idx = knn_coords(x_coords, k=k) # (batch_size, num_points, k)
        
    device = x.device

    # グラフの構築のためのインデックス操作
    # k-NNで得られたインデックスを使って、対応する特徴量を取得
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # x_flat は (B*N, C)
    x_flat = x.transpose(2, 1).contiguous().view(-1, num_features) 
    
    # neighbor は (B*N*k, C) - 近傍点の全ての特徴量
    neighbor = x_flat[idx, :] 
    
    # neighbor を (B, N, k, C) にreshape
    neighbor = neighbor.view(batch_size, num_points, k, num_features) 
    
    # 各点 x を (B, N, 1, C) にreshapeして拡張
    x = x.transpose(2, 1).contiguous().view(batch_size, num_points, 1, num_features)

    # 2. エッジ特徴量の構築
    # DGCNNのエッジ特徴量: (x_i, x_j - x_i)
    # ここでは、元の特徴量 (x_i) と差分 (x_j - x_i) を結合する
    # x.expand_as(neighbor) は (B, N, k, C)
    # neighbor - x は (B, N, k, C)
    # feature は (B, N, k, 2*C) となる (C=4 なので、2*4=8)
    feature = torch.cat([x.expand_as(neighbor), neighbor - x], dim=3)
    
    # feature を (B, N, k, 2*C) から (B, 2*C, N, k) に変更
    feature = feature.permute(0, 3, 1, 2).contiguous() 

    return feature

# --- k-NN関数 (座標専用版) ---
def knn_coords(x, k):
    """
    Input:
        x: (B, 3, N)  # 3次元座標のみを期待
        k: int
    Return:
        idx: (B, N, k)  # 各点のk近傍点のインデックス
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    # 距離のトップkを取得（最も小さい距離、つまり最も近いk個の点）
    # topkはデフォルトで降順なので、-pairwise_distanceを使うか、largest=Falseを指定
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]   # (batch_size, num_points, k)
    return idx


# --- 汎用k-NN関数 (特徴量空間用) ---
# DGCNNの2回目以降のEdgeConvで、特徴量空間でk-NNを行う場合に使う
def knn_features(x, k):
    """
    Input:
        x: (B, C_features, N)  # 任意のC_features次元特徴量
        k: int
    Return:
        idx: (B, N, k)  # 各点のk近傍点のインデックス
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]   # (batch_size, num_points, k)
    return idx


# --- DGCNNモデルクラス (修正版) ---
class DGCNNFeatureExtractor(nn.Module): # クラス名を変更して意図を明確にする
    def __init__(self, k=20, emb_dims=1024): # num_classes を削除
        super(DGCNNFeatureExtractor, self).__init__()
        self.k = k
        self.emb_dims = emb_dims

        # 最初のEdgeConv層 (変更なし)
        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        # 2番目のEdgeConv層 (変更なし)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        # 3番目のEdgeConv層 (変更なし)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))

        # 4番目の畳み込み層 (変更なし)
        self.conv4 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(self.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # 最終の全結合層（分類ヘッド）は削除する
        # self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=0.5)
        # self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 4) - to_dense_batch() からくる形式
        x = x.permute(0, 2, 1) # (B, 4, N)
        
        batch_size = x.size(0)
        
        # 1. 最初のEdgeConv層 (k-NNは座標のみ)
        x = get_graph_feature(x, k=self.k) 
        x = self.conv1(x) # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)
        
        # 2. 2番目のEdgeConv層 (k-NNは特徴量空間で)
        x = get_graph_feature_generic(x1, k=self.k) 
        x = self.conv2(x) # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # 3. 3番目のEdgeConv層 (k-NNは特徴量空間で)
        x = get_graph_feature_generic(x2, k=self.k) 
        x = self.conv3(x) # (B, 128, N, k)
        x3 = x.max(dim=-1, keepdim=False)[0] # (B, 128, N)

        # 4. Global Feature (Concat of Max/Avg Pooling)
        x = torch.cat((x1, x2, x3), dim=1) # (B, 256, N)
        
        x = self.conv4(x) # (B, emb_dims, N)
        
        # Global Max PoolingとGlobal Avg Pooling
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (B, emb_dims)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # (B, emb_dims)
        
        # 最終特徴量を結合
        global_feature = torch.cat((x_max, x_avg), dim=1) # (B, emb_dims * 2)

        # クラス分類の全結合層は削除し、global_feature を直接返す
        return global_feature

# --- 汎用版 get_graph_feature_generic の定義 ---
# (通常のDGCNNのget_graph_featureと同じだが、k-NNの入力をスライスしない)
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
        # k-NNは入力特徴量全体 (C次元) で行う
        idx = knn_features(x, k=k) # (batch_size, num_points, k)
        
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