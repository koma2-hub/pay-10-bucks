import os, sys
sys.path.append(os.pardir)
import numpy as np
import torch 
from utils.data_utils import load_ply


#点群xから近傍点のindexをk個返す関数
def knn(x: torch.Tensor, k: int):
    """
    一つの点群データ `x` の各点について、k近傍点のインデックスを計算する。

    Args:
        x: 点群データ。形状は (C, N)。
           C: 特徴量次元（通常はXYZ座標なので3）
           N: 点の数
        k: 探す近傍点の数。

    Returns:
        torch.Tensor: 各点のk近傍点のインデックス。形状は (N, k)。
    """
    # x の形状を (N, C) に転置
    x_transposed = x.transpose(1, 0) # (C, N) -> (N, C)

    # 全ての点間の距離の2乗を効率的に計算
    
    # 1. 各点のノルム（長さ）の2乗 ||p||^2 を計算
    # (N, C) のテンソルで各点の特徴量（C次元）のノルムの2乗を計算
    x_norm_sq = torch.sum(x_transposed**2, dim=1, keepdim=True) # 形状: (N, 1)

    # 2. 全ての点 p と q の内積 p・q を計算
    # x_transposed と x を行列積
    dot_product = torch.matmul(x_transposed, x) # 形状: (N, N)
    
    # 3. ユークリッド距離の2乗を計算
    # 距離行列: dist = x_norm_sq + x_norm_sq.T - 2 * dot_product
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T # 形状: (N, N)

    # 4. 各点について、最も距離が小さい k 個の点のインデックスを取得
    # topk() はデフォルトで最大値を取得するため、距離行列を負にして最小値を求める
    _, indices = torch.topk(-dist_matrix, k=k, dim=1)
    # indices の形状: (N, k)

    return indices

#輝度値は0~1を想定している
def intensity_histogram(pcd, bin, k):
    idx = knn(pcd, k)
    hist = torch.zeros(pcd.shape[0], bin)

    #すべての点群について近傍点のヒストグラムを作成
    for point in range(pcd.shape[0]):
        #近傍点の輝度値を取得し分割に従って個数を数える
        for indice in range(idx.shape[1]):
            for i in range(bin):
                if(i/bin < pcd[point, indice]):
                    hist[point, bin] += 1
    
    return intensity_histogram


x = torch.arange(0,30).reshape(10,3)
y = torch.randn(10, 3)
"""
print('x:', x)
print('x transpose:', x.transpose(1, 0))
print('inner:', -2*torch.matmul(x.transpose(1, 0), x))
print('x**2', torch.sum(x**2, dim = 1, keepdim=True))
print('x** transpose', torch.sum(x**2, dim = 1, keepdim=True).transpose(1,0))
"""
data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/robot_record_icn/raw"
file_names = os.listdir(data_path)
for file in file_names:
    file_path = os.path.join(data_path, file)
    pcd = load_ply(file_path)
    torch_pcd = torch.from_numpy(pcd)
    intensity_false = torch_pcd[:, :3]
    indices = knn(intensity_false, 5)
    print(indices)

