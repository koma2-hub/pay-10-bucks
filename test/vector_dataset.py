import os,sys
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import load_ply


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

class HistogramPairDataset(Dataset):
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
        hist1_data, hist2_data, label = self.pairs[idx]

        # NumPy配列をPyTorchテンソルに変換
        hist1 = torch.from_numpy(hist1_data.astype(np.float32))
        hist2 = torch.from_numpy(hist2_data.astype(np.float32))
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            # 何らかの変換を適用（今回は省略）
            pass
            
        return hist1, hist2, label

# --- 使用例 ---


"""ここで点群データを読み込み、近傍点探索を行う。"""
data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/robot_record_icn/raw/"
file_names = os.listdir(data_path)
dummy_pairs = []
hist_datas = []
k = 64
bins = 16

for file in file_names:
    file_path = os.path.join(data_path, file)
    pcd = load_ply(file_path)
    indices = knn(pcd, k = k)
    rng = np.random.default_rng()

    """
    正例のペアを作成
    一つのplyファイルから4点ランダムに選びそれ周りの点群の輝度値を取得する
    """
    for i in range(16):
        #サンプリングする点をランダムに選択
        sample_point = rng.integers(pcd.shape[0])
        #サンプリングする点の近傍点のindexを取得
        neighbor_indices = indices[sample_point]
        #サンプリングする点とその近傍点の輝度値を取得
        pcd_intensity = pcd[neighbor_indices,-1]
        #ヒストグラムの作成(オリジナル)
        hist_data = create_intensity_histogram(pcd_intensity, bins=bins, density=True)
        #ヒストグラムの作成(ノイズ有)
        pcd_intensity_noise = np.copy(pcd_intensity)
        pcd_intensity_noise = pcd_intensity_noise + np.random.normal(0, 0.01, k)
        hist_data_noise = create_intensity_histogram(pcd_intensity_noise, bins=bins, density=True)
        #ヒストグラムをペアとして保存
        dummy_pairs.append((hist_data, hist_data_noise, 1))
        
    """
    負例のペアを作成
    1つのplyファイルから4点ランダムに選び他のファイルから選んだ4点とペアリングする
    """
    #負のペア用にファイルをランダムにロード
    random_file_index = rng.integers(len(file_names))
    random_file_path = os.path.join(data_path, file_names[random_file_index])
    random_pcd = load_ply(random_file_path)
    random_pcd_indices = knn(random_pcd, k = k)
    for i in range(16):
        #サンプリングする点をランダムに選択
        sample_point = rng.integers(pcd.shape[0])
        #サンプリングする点の近傍点のindexを取得
        neighbor_indices = indices[sample_point]
        #サンプリングする点とその近傍点の輝度値を取得
        pcd_intensity = pcd[neighbor_indices, -1]
        #ヒストグラムの作成
        hist_data = create_intensity_histogram(pcd_intensity, bins=bins, density=True)

        #負のペアのサンプリング点をランダムに選択
        sample_point_negative = rng.integers(random_pcd.shape[0])
        #サンプリングする点の近傍点のインデックスを取得
        neighbor_indices_negative = random_pcd_indices[sample_point_negative]
        #サンプリングする点とその近傍点の輝度値を取得
        pcd_intensity_negative = random_pcd[neighbor_indices_negative, -1]
        #ヒストグラムの作成
        hist_data_negative = create_intensity_histogram(pcd_intensity_negative, bins=bins, density=True)
        
        #ヒストグラムを負のペアとして保存
        dummy_pairs.append((hist_data, hist_data_negative, 0))


# データセットとデータローダーの作成
dataset = HistogramPairDataset(pairs=dummy_pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# データローダーからバッチを取得してみる
hist1_batch, hist2_batch, labels_batch = next(iter(dataloader))

print("Batch of Histogram 1 Shape:", hist1_batch.shape)
print("Batch of Histogram 2 Shape:", hist2_batch.shape)
print("Batch of Labels Shape:", labels_batch.shape)

# dummy_pairs = [(hist1, hist2, label), ...]

# データをNumPy配列に変換
# 扱いやすいように、hist1, hist2, labelsで別々の配列にまとめます
hist1_list = np.array([pair[0] for pair in dummy_pairs])
hist2_list = np.array([pair[1] for pair in dummy_pairs])
labels_list = np.array([pair[2] for pair in dummy_pairs])

# .npz 形式で圧縮して保存
save_file = 'histogram_dataset_' + str(bins) + 'bins_' + str(k) + 'points_' + 'weaknoise' + str(hist1_list.shape[0]) + '.npz'
save_path = os.path.join('/mnt/c/Users/matsu/SICK/pay-10-bucks/kICN_Dataset', save_file)
np.savez_compressed(
    save_path, 
    hist1=hist1_list,
    hist2=hist2_list,
    labels=labels_list
)

print("データセットを", save_path, "として保存しました。")

# 保存したデータを読み込む
data = np.load(save_path)

hist1_loaded = data['hist1']
hist2_loaded = data['hist2']
labels_loaded = data['labels']

print("読み込んだデータのShape:", hist1_loaded.shape)

# PyTorchのDatasetクラスは少し変更が必要です
class NpDataset(Dataset):
    def __init__(self, hist1, hist2, labels):
        self.hist1 = torch.from_numpy(hist1.astype(np.float32))
        self.hist2 = torch.from_numpy(hist2.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hist1[idx], self.hist2[idx], self.labels[idx]

# 読み込んだデータでデータセットを作成
dataset = NpDataset(hist1_loaded, hist2_loaded, labels_loaded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 動作確認
hist1_batch, hist2_batch, labels_batch = next(iter(dataloader))
print("\nデータローダーから取得したバッチのShape:")
print("Hist1:", hist1_batch.shape)
print("Labels:", labels_batch.shape)

