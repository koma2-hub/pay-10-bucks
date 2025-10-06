import numpy as np
import time
import fpsample 
import open3d as o3d
from utils.data_utils import load_ply , farthest_point_sampling
import os,sys
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import load_ply


class VectorPairDataset(Dataset):
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
        intensity_vec1_data, intensity_vec2_data, label = self.pairs[idx]

        # NumPy配列をPyTorchテンソルに変換
        intensity1 = torch.from_numpy(intensity_vec1_data.astype(np.float32))
        intensity2 = torch.from_numpy(intensity_vec2_data.astype(np.float32))
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            # 何らかの変換を適用（今回は省略）
            pass
            
        return intensity1, intensity2, label

dataset_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/kICN_Dataset/vector_dataset_32points_noise005_2064_labs.npz"


# 保存したデータを読み込む
data = np.load(dataset_path)

intensity1_loaded = data['intensity1']
intensity2_loaded = data['intensity2']
labels_loaded = data['labels']

print("読み込んだデータのShape:", intensity1_loaded.shape)

# PyTorchのDatasetクラスは少し変更が必要です
class NpDataset(Dataset):
    def __init__(self, intensity1, intensity2, labels):
        self.intensity1 = torch.from_numpy(intensity1.astype(np.float32))
        self.intensity2 = torch.from_numpy(intensity2.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.intensity1[idx], self.intensity2[idx], self.labels[idx]

# 読み込んだデータでデータセットを作成
dataset = NpDataset(intensity1_loaded, intensity2_loaded, labels_loaded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 動作確認
intensity1_batch, intensity2_batch, labels_batch = next(iter(dataloader))
print("\nデータローダーから取得したバッチのShape:")
print("Intensity1:", intensity1_batch.shape)
print("Labels:", labels_batch.shape)

print(intensity1_batch)


