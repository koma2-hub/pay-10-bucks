#utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import fpsample

def load_ply(filename):
    #.plyファイルを読み込み　点群(x, y, z, intensity)のnumpy配列を返す
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            header_index = None
            for i, line in enumerate(lines):
                if 'end_header' in line:
                    header_index = i
                    break
            if header_index is None:
                raise ValueError("PLYファイルのヘッダが正しく読み込めませんでした。")
            
            # ヘッダ以降の行を読み込み
            points = np.array([list(map(float, l.split())) for l in lines[header_index+1:]])
            if points.shape[1] < 4:
                # xyz + intensity(もしくは他の属性)がなければエラー
                raise ValueError(f"期待される列数に満たないデータが検出されました: {points.shape[1]}列")
            
            # [x, y, z, intensity] の形に整形
            # intensity が最後の列にあると仮定 (points[:, -1])
            points = np.concatenate([points[:, :3], points[:, -1].reshape(-1, 1)], axis=1)
            
            return points
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        # 必要に応じて sys.exit(1) などで終了するか、Noneを返す
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました(load_ply): {e}")
        return None
    

def l2_norm(a, b):
    return ((a - b) ** 2).sum(axis =1)

def farthest_point_sampling(pcd, k, metrics = l2_norm):
    indices = np.zeros(k, dtype=np.int32)
    distances = np.zeros((k, pcd.shape[0]), dtype=np.float32)
    indices[0] = np.random.randint(len(pcd))
    farthest_point = pcd[indices[0]]
    min_distances = metrics(farthest_point, pcd)
    distances[0, :] = min_distances
    for i in range(1,k):
        indices[i] = np.argmax(min_distances)
        farthest_point = pcd[indices[i]]
        distances[i, :] = metrics(farthest_point, pcd)
        min_distances = np.minimum(min_distances, distances[i, :])
    return indices

def knn(x: np.ndarray, k: int):
    """
    一つの点群データ `x` の各点について、k近傍点のインデックスをNumPyで計算する。

    Args:
        x: 点群データ。形状は (N, C)。
           N: 点の数
           C: 特徴量次元
        k: 探す近傍点の数。

    Returns:
        np.ndarray: 各点のk近傍点のインデックス。形状は (N, k)。
    """
    num_points = x.shape[0]

    # kが点群の総点数を超えないように調整
    k = min(k, num_points)
    if k <= 1:
        k = 1
    x_norm_sq = np.sum(x**2, axis=1, keepdims=True)  # 形状: (N, 1)
    dot_product = np.matmul(x, x.T)  # 形状: (N, N)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T  # 形状: (N, N)
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

    return indices

def create_intensity_histogram(point_cloud_intensity_data, bins=4, density=True):
    """
    輝度値の配列から正規化されたヒストグラムを作成する関数
    """
    # ヒストグラムを作成
    hist, bin_edges = np.histogram(
        point_cloud_intensity_data,
        bins=bins,
        range=(0.0, 1.0), # 輝度値が0.0~1.0に正規化されていると仮定
        density=density  # Trueにするとヒストグラムの面積が1になるように正規化される
    )
    
    # # (別オプション) L2正規化
    # hist = hist.astype(np.float32)
    # l2_norm = np.linalg.norm(hist)
    # if l2_norm > 0:
    #     hist /= l2_norm
        
    return hist

def random_transform(points_np, rotation_range=(-180, 180), translation_range=(-10, 10)):
    """
    点群にランダムなアフィン変換、ノイズ、ドロップアウトを適用する関数。
    points_np: (N, C) NumPy配列 (Nは点数、Cは特徴量次元, C>=3)
    """
    transformed_points = points_np.copy()
    
    # 1. 回転 (Z軸回転が一般的ですが、XYZ軸回転も可)
    angle_z = np.random.uniform(np.deg2rad(rotation_range[0]), np.deg2rad(rotation_range[1]))
    cos_z = np.cos(angle_z)
    sin_z = np.sin(angle_z)
    rotation_matrix = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    # その他の軸回転も追加可能
    transformed_points[:, :3] = transformed_points[:, :3] @ rotation_matrix.T
    print(rotation_matrix)
        # 2. 並進
    translation = np.random.uniform(translation_range[0], translation_range[1], size=3).astype(np.float32)
    transformed_points[:, :3] += translation
    
    return transformed_points, rotation_matrix

def save_ply(filename, pcd):
    """
    点群データを PLY ファイルとして保存する関数。
    出力フォーマットは以下の通り:
      property float32 x
      property float32 y
      property float32 z
      property uint8 r
      property uint8 g
      property uint8 b
      property float32 i
      
    数値の桁指定:
      - x, y, z: 小数点以下4桁まで
      - r, g, b: 整数 (常に 0)
      - i: 小数点以下2桁まで

    例:
      -1334.0197 -1060.7484 1785.6458 0 0 0 0.16

    入力:
      pcd: (N,3) または (N,4) の numpy 配列または torch.Tensor
           4列目が存在する場合は intensity として使用、なければ 0 とする。
    """
    # Noneチェック
    if pcd is None:
        print(f"保存対象が None のためスキップします: {filename}")
        return

    # torch.Tensor の場合は numpy 配列に変換
    if isinstance(pcd, torch.Tensor):
        pcd_np = pcd.cpu().numpy()
    else:
        pcd_np = np.asarray(pcd)

    # 入力の形状チェック (N,3) または (N,4)
    if pcd_np.ndim != 2 or pcd_np.shape[1] not in (3, 4):
        raise ValueError("pcd は (N,3) または (N,4) の形状である必要があります。")

    # 座標は float32 として取得
    xyz = pcd_np[:, :3].astype(np.float32)
    
    # intensity の取得: 4列目があればその値、なければ 0
    if pcd_np.shape[1] == 4:
        intensity = pcd_np[:, 3].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((pcd_np.shape[0], 1), dtype=np.float32)
    
    # r, g, b は uint8 の 0 として生成
    rgb = np.zeros((pcd_np.shape[0], 3), dtype=np.uint8)
    
    # x, y, z, r, g, b, i の順にデータを結合 (shape: (N,7))
    data = np.hstack((xyz, rgb, intensity))

    # PLY ヘッダーの作成
    header = f"""ply
                format ascii 1.0
                element vertex {data.shape[0]}
                property float32 x
                property float32 y
                property float32 z
                property uint8 r
                property uint8 g
                property uint8 b
                property float32 i
                end_header
                """
    # ファイルにヘッダーと各点のデータを書き出す
    with open(filename, "w") as f:
        f.write(header)
        for row in data:
            # 書式: x,y,z は小数点以下4桁、i は小数点以下2桁で出力
            f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])} {int(row[4])} {int(row[5])} {row[6]:.2f}\n")


def downsample_pcd(pointcloud, downsample_point) -> np.ndarray:
    if pointcloud.shape[0] < downsample_point:
        downsample_point = pointcloud.shape[0]
    fps_indices = fpsample.fps_sampling(pointcloud, downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :4]
    return downsampled_pc


# --- 補助関数 ---
def knn(x: np.ndarray, k: int):
    """
    一つの点群データ `x` の各点について、k近傍点のインデックスをNumPyで計算する。

    Args:
        x: 点群データ。形状は (N, C)。
           N: 点の数
           C: 特徴量次元
        k: 探す近傍点の数。

    Returns:
        np.ndarray: 各点のk近傍点のインデックス。形状は (N, k)。
    """
    num_points = x.shape[0]

    # kが点群の総点数を超えないように調整
    k = min(k, num_points)
    if k <= 1:
        k = 1
    x_norm_sq = np.sum(x**2, axis=1, keepdims=True)  # 形状: (N, 1)
    dot_product = np.matmul(x, x.T)  # 形状: (N, N)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T  # 形状: (N, N)
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

    return indices

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