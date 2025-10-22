import os,sys
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import load_ply
import time
import fpsample 
import open3d as o3d



# from plyfile import PlyData # .plyファイルを扱う場合の一例
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

def random_transform(points_np, rotation_range=(-180, 180), translation_range=(-0.1, 0.1), noise_std=0.01):
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

    # 2. 並進
    translation = np.random.uniform(translation_range[0], translation_range[1], size=3).astype(np.float32)
    transformed_points[:, :3] += translation

    # 3. ノイズ
    noise = np.random.normal(0, noise_std, size=transformed_points[:, :3].shape).astype(np.float32)
    transformed_points[:, :3] += noise


    return transformed_points

class PcdPairDataset(Dataset):
    """
    輝度値ヒストグラムのペアとラベルを返すデータセットクラス
    """
    def __init__(self, pairs, transform=None):
        """
        Args:
            pairs (list): (ヒストグラム1, ヒストグラム2, ラベル) のタプルを含むリスト
            transform (callable, optional): データに適用される変換
        """
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        # データセットの総ペア数を返す
        return len(self.pairs)

    def __getitem__(self, idx):
        # 指定されたインデックスのデータペアを取得
        pcd1, pcd2, label = self.pairs[idx]

        # NumPy配列をPyTorchテンソルに変換
        pcd1 = torch.from_numpy(pcd1.astype(np.float32))
        pcd2 = torch.from_numpy(pcd2.astype(np.float32))
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            # 何らかの変換を適用（今回は省略）
            pass
            
        return pcd, pcd2, label
    
"""ノイズ生成用の関数"""
def generate_noise(min, max, step, num):
    num_steps = int((max - min) / step) + 1
    random_integer = np.random.randint(0, num_steps, num)
    result = min + random_integer*step
    return result
   
"""ここで点群データを読み込み、近傍点探索を行う。"""
data_path1 = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/robot_record_icn/raw"
data_path2 = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs_icn/raw"
file_names1 = os.listdir(data_path1)
file_names2 = os.listdir(data_path2)

dummy_pairs = []
k = 32

for i, file in enumerate(file_names1):
    #print("file name:", file)
    file_path = os.path.join(data_path1, file)
    pcd = load_ply(file_path)
    indices = knn(pcd, k=k)
    rng = np.random.default_rng()

    """
    正例のペアの作成
    """
    for j in range(8):
        sample_point = rng.integers(pcd.shape[0])
        #近傍点の取得
        neighbor_indices = indices[sample_point]

        #近傍点の取得
        pcd_positive = pcd[neighbor_indices]
        #正例のペアとしてノイズと変換を加えた点群を取得
        pcd_positive_transformed = np.copy(pcd_positive)
        pcd_positive_transformed = random_transform(pcd_positive_transformed)
        pcd_positive_transformed[:, -1] += generate_noise(-0.05, 0.05, 0.01, k)
        #輝度値が負になっている可能性があるので0か最大値を返すようにする
        pcd_positive_transformed = np.maximum(0, pcd_positive_transformed)
        #print(pcd_positive_noise)
        #データに加える
        dummy_pairs.append((pcd_positive, pcd_positive_transformed, 1))
    
    """
    負例のペアの作成
    """
    
    for j in range(8):
        #負のペア用に異なるデータからファイルをロード
        negative_file_index = rng.integers(len(file_names2))
        negative_file_path = os.path.join(data_path2, file_names2[negative_file_index])
        negative_pcd = load_ply(negative_file_path)
        negative_pcd_indices = knn(negative_pcd, k = k)


        #サンプリングする点をランダムに選択
        sample_point = rng.integers(pcd.shape[0])
        #サンプリングする点の近傍点のindexを取得
        neighbor_indices = indices[sample_point]
        #サンプリングする点とその近傍点を取得
        pcd_negative = pcd[neighbor_indices]

        #負のペアのサンプリング点をランダムに選択
        sample_point_negative = rng.integers(negative_pcd.shape[0])
        #サンプリングする点の近傍点のインデックスを取得
        neighbor_indices_negative = negative_pcd_indices[sample_point_negative]
        #サンプリングする点とその近傍点を取得
        pcd_negative_negative = negative_pcd[neighbor_indices_negative]
        
        #ヒストグラムを負のペアとして保存
        dummy_pairs.append((pcd_negative, pcd_negative_negative, 0))


"""
同じデータが含まれる可能性があるので後で変更する
"""
#別データからもデータセットを作成する
for i, file in enumerate(file_names2):
    #print("file name:", file)
    file_path = os.path.join(data_path2, file)
    pcd = load_ply(file_path)
    indices = knn(pcd, k=k)
    rng = np.random.default_rng()

    """
    正例のペアの作成
    """
    for j in range(8):
        sample_point = rng.integers(pcd.shape[0])
        #近傍点の取得
        neighbor_indices = indices[sample_point]
        #近傍点のの取得。ユークリッド距離をもとにソートされているはず
        pcd_positive = pcd[neighbor_indices]
        #正例のペアとしてノイズと変換を加えた点群を取得
        pcd_positive_transformed = np.copy(pcd_positive)
        pcd_positive_transformed = random_transform(pcd_positive_transformed)
        pcd_positive_transformed[:, -1] += generate_noise(-0.05, 0.05, 0.01, k)
        #輝度値が負になっている場合は0とする
        pcd_positive_transformed = np.maximum(0, pcd_positive_transformed)
        #データに加える
        dummy_pairs.append((pcd_positive, pcd_positive_transformed, 1))
    
    """
    負例のペアの作成
    """
    
    for j in range(8):
        #負のペア用に異なるデータからファイルをロード
        negative_file_index = rng.integers(len(file_names1))
        negative_file_path = os.path.join(data_path1, file_names1[negative_file_index])
        negative_pcd = load_ply(negative_file_path)
        negative_pcd_indices = knn(negative_pcd, k = k)


        #サンプリングする点をランダムに選択
        sample_point = rng.integers(pcd.shape[0])
        #サンプリングする点の近傍点のindexを取得
        neighbor_indices = indices[sample_point]
        #サンプリングする点とその近傍点を取得
        pcd_negative = pcd[neighbor_indices]

        #負のペアのサンプリング点をランダムに選択
        sample_point_negative = rng.integers(negative_pcd.shape[0])
        #サンプリングする点の近傍点のインデックスを取得
        neighbor_indices_negative = negative_pcd_indices[sample_point_negative]
        #サンプリングする点とその近傍点の輝度値を取得
        pcd_negative_negative = negative_pcd[neighbor_indices_negative]

        #ヒストグラムを負のペアとして保存
        dummy_pairs.append((pcd_negative, pcd_negative_negative, 0))




# データセットとデータローダーの作成
dataset = PcdPairDataset(pairs=dummy_pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# データローダーからバッチを取得してみる
pcd1_batch, pcd2_batch, labels_batch = next(iter(dataloader))

print("Batch of pcd 1 Shape:", pcd1_batch.shape)
print("Batch of pcd 2 Shape:", pcd2_batch.shape)
print("Batch of Labels Shape:", labels_batch.shape)


# データをNumPy配列に変換
# 扱いやすいように、hist1, hist2, labelsで別々の配列にまとめます
pcd1_list = np.array([pair[0] for pair in dummy_pairs])
pcd2_list = np.array([pair[1] for pair in dummy_pairs])
labels_list = np.array([pair[2] for pair in dummy_pairs])

# .npz 形式で圧縮して保存
save_file = 'pcd_'  + str(k) + 'points_' +'noise005_transformed_'+ str(pcd1_list.shape[0])  + '.npz'
save_path = os.path.join('/mnt/c/Users/matsu/SICK/pay-10-bucks/ISN/dataset', save_file)
np.savez_compressed(
    save_path, 
    pcd1=pcd1_list,
    pcd2=pcd2_list,
    labels=labels_list
)

print("データセットを", save_path, "として保存しました。")

# 保存したデータを読み込む
data = np.load(save_path)

pcd1_loaded = data['pcd1']
pcd2_loaded = data['pcd2']
labels_loaded = data['labels']

print("読み込んだデータのShape:", pcd1_loaded.shape)

# PyTorchのDatasetクラスは少し変更が必要です
class NpDataset(Dataset):
    def __init__(self, pcd1, pcd2, labels):
        self.pcd1 = torch.from_numpy(pcd1.astype(np.float32))
        self.pcd2 = torch.from_numpy(pcd2.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pcd1[idx], self.pcd2[idx], self.labels[idx]

# 読み込んだデータでデータセットを作成
dataset = NpDataset(pcd1_loaded, pcd2_loaded, labels_loaded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 動作確認
pcd1_batch, pcd2_batch, labels_batch = next(iter(dataloader))
print("\nデータローダーから取得したバッチのShape:")
print("pcd1:", pcd1_batch.shape)
print("Labels:", labels_batch.shape)


