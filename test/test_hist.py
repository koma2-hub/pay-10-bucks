import os,sys
sys.path.append(os.pardir)

import torch
from utils.data_utils import load_ply


# 修正後の knn 関数
def knn(x: torch.Tensor, k: int):
    # 点群の総点数を取得
    num_points = x.shape[1]

    # kが点群の総点数を超えないように調整
    k = min(k, num_points)
    if k <= 1: # 少なくとも自分自身は含むようにする
        k = 1

    # x の形状を (N, C) に転置
    x_transposed = x.transpose(1, 0) # (C, N) -> (N, C)
    
    # 全ての点間の距離の2乗を効率的に計算
    x_norm_sq = torch.sum(x_transposed**2, dim=1, keepdim=True)
    dot_product = torch.matmul(x_transposed, x)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T

    # k近傍のインデックスを取得    
    _, indices = torch.topk(-dist_matrix, k=k, dim=1)

    return indices

# 修正後の intensity_histogram 関数
def intensity_histogram(pcd: torch.Tensor, bin_count: int, k: int):
    """
    各点とそのk近傍点の輝度値ヒストグラムを計算する。
    
    Args:
        pcd: 点群データ。形状は (N, 4) (XYZI)
        bin_count: ヒストグラムのビンの数
        k: k近傍点の数

    Returns:
        torch.Tensor: 各点のローカルヒストグラム。形状は (N, bin_count)
    """
    # 輝度値チャンネル（4番目の次元）が存在するか確認
    if pcd.shape[1] < 4:
        raise ValueError("Input point cloud must have an intensity channel (shape: N, 4)")
    
    # knn 関数は (C, N) の形状を期待するため、輝度値の次元だけ抜き出して転置する
    # しかし、ここでは XYZ座標のみで近傍を探し、輝度値は後で使うという前提で実装する
    pcd_coords = pcd[:, :3].transpose(1, 0) # (N, 3) -> (3, N)
    pcd_intensity = pcd[:, 3]               # (N,)

    # k近傍のインデックスを取得
    # knn関数は(C, N)を期待するため、x_transposedの形で渡す
    indices = knn(pcd_coords, k) # 形状: (N, k)

    # 近傍点の輝度値を取得
    # indices を使って、pcd_intensity から近傍点の輝度値を抽出
    neighbor_intensities = pcd_intensity[indices] # 形状: (N, k)

    # torch.histc を使ってヒストグラムを計算
    # histc は1次元テンソルにしか使えないため、各行（各点）ごとにループが必要
    # ただし、元のコードの3重ループよりはるかに高速
    histograms = torch.zeros(pcd.shape[0], bin_count, device=pcd.device)

    for i in range(pcd.shape[0]):
        # 各点の近傍輝度値のヒストグラムを計算
        histograms[i, :] = torch.histc(neighbor_intensities[i], bins=bin_count, min=0.0, max=1.0)
    
    return histograms

# 修正後のメイン実行ブロック

# ... (関数の定義後) ...

if __name__ == '__main__':
    # ... (ファイルの読み込みは既存のまま) ...
    data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/robot_record_icn/raw"
    file_names = os.listdir(data_path)
    
    bin_count = 10 # ヒストグラムのビンの数
    k_neighbors = 5 # k近傍点の数

    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        
        # 点群の点数が少なすぎる場合はスキップ
        if pcd.shape[0] < k_neighbors:
            print(f"Skipping {file} due to insufficient points ({pcd.shape[0]} < {k_neighbors}).")
            continue
            
        print(f"Processing {file} with {pcd.shape[0]} points.")
        torch_pcd = torch.from_numpy(pcd).float()

        # knn関数は座標のみで近傍を探す
        pcd_coords = torch_pcd[:, :3].transpose(1,0) # (N, 3) -> (3, N)
        print(knn(pcd_coords, k=k_neighbors))