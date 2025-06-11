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

class DGCNNLocalFeatureExtractor(nn.Module):
    def __init__(self, k=20, emb_dims=1024, projection_dim=128): # projection_dim を追加
        super(DGCNNLocalFeatureExtractor, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.projection_dim = projection_dim # プロジェクション次元を保存

        # 既存のDGCNNの層 (変更なし)
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

        # --- ここからプロジェクションヘッドの追記 ---
        # ローカル特徴量 (emb_dims) を受け取り、projection_dim に射影するMLP
        # 典型的には、バッチ正規化とReLUを挟んだ2層のMLPが使われます。
        self.projection_head = nn.Sequential(
            nn.Linear(self.emb_dims, self.emb_dims // 2, bias=False), # 1層目: 半分に次元削減
            nn.BatchNorm1d(self.emb_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dims // 2, self.projection_dim, bias=False) # 2層目: 最終プロジェクション次元へ
            # 論文によっては、最後のLinear層の後に活性化関数やBNを含めないこともあります。
            # 例: SimCLRでは最後のLinear層の後に活性化関数は適用しない。
        )
        # --- プロジェクションヘッドの追記終了 ---

    def forward(self, x):
        # x: (B, N, 4) - 入力点群 (x, y, z, intensity)
        x = x.permute(0, 2, 1) # (B, 4, N)

        batch_size = x.size(0)

        # 1. 最初のEdgeConv層 (k-NNは座標のみ)
        x_initial_features = x # (B, 4, N) 全特徴量を使用
        
        x = get_graph_feature(x_initial_features, k=self.k) # (B, 8, N, k)
        x = self.conv1(x) # (B, 64, N, k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # 2. 2番目のEdgeConv層 (k-NNは特徴量空間で)
        x = get_graph_feature_generic(x1, k=self.k) # (B, 128, N, k)
        x = self.conv2(x) # (B, 64, N, k)
        x2 = x.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # 3. 3番目のEdgeConv層 (k-NNは特徴量空間で)
        x = get_graph_feature_generic(x2, k=self.k) # (B, 128, N, k)
        x = self.conv3(x) # (B, 128, N, k)
        x3 = x.max(dim=-1, keepim=False)[0] # (B, 128, N)

        # 4. Concatenate (点ごとの特徴量を結合)
        local_features = torch.cat((x1, x2, x3), dim=1) # (B, 256, N)

        local_features = self.conv4(local_features) # (B, emb_dims, N)

        # --- ここからプロジェクションヘッドの適用 ---
        # local_features の形状は (B, emb_dims, N)
        # プロジェクションヘッドは nn.Linear を含むため、通常は (Batch_size, Features) 形式の入力を期待します。
        # ここでは、各点の特徴量に対してプロジェクションを適用したいので、形状を (B*N, emb_dims) に変更します。
        B, C_feats, N = local_features.shape # B: バッチサイズ, C_feats: emb_dims, N: 点数

        # 形状を (B*N, emb_dims) に変更してプロジェクションヘッドに渡す
        # ここで Batch Normalization が Batch*N の次元で機能することに注意
        projected_features = self.projection_head(local_features.view(B * N, C_feats))

        # 元の形状 (B, N, projection_dim) に戻す
        projected_features = projected_features.view(B, N, self.projection_dim)
        # --- プロジェクションヘッドの適用終了 ---

        # 訓練時には、プロジェクションヘッドの出力をコントラスティブ損失の計算に用いる
        # 推論（特徴量抽出）時には、プロジェクションヘッドを通さずにローカル特徴量（emb_dims）を返すか、
        # プロジェクション後の特徴量（projection_dim）を返すか、タスクによる。
        # 通常はエンコーダの出力（local_features）をダウンストリームタスクに使うため、
        # 学習モード（model.train()）ではprojected_featuresを、
        # 評価モード（model.eval()）ではlocal_featuresを返すようにすることもできます。
        # あるいは、常にprojected_featuresを返し、必要に応じて後でemb_dimsの特徴を使う。

        # コントラスティブ学習の損失計算のために、projected_featuresを返す
        # 位置合わせのためのマッチングには、local_features (emb_dims次元) を使うことが多いです。
        # どちらの出力を使うかは、学習タスクとダウンストリームタスクの目的に依存します。
        
        # 例: 学習時はprojected_featuresを返し、実際の使用時はlocal_featuresを返す場合
        if self.training: # `self.training` はモデルが訓練モードか（model.train()が呼ばれているか）を示す
            return projected_features # (B, N, projection_dim)
        else:
            return local_features # (B, N, emb_dims)


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