import os
import torch
import numpy as np
import open3d as o3d # 点群の読み込みや変換に便利
from data_utils import load_ply


def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    # 必要に応じて輝度値などもロード
    return points

def sample_patch(points, center_point_index, num_points_per_patch, radius=None, k_neighbors=None):
    """点群から中心点周辺のパッチをサンプリングする関数"""
    center = points[center_point_index]
    if radius is not None:
        # 半径内の点を抽出
        distances = np.linalg.norm(points - center, axis=1)
        indices = np.where(distances <= radius)[0]
    elif k_neighbors is not None:
        # K近傍点を抽出 (効率的な実装にはKD-treeなどを使用)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points)
        _, indices = knn.kneighbors([center])
        indices = indices[0]
    else:
        raise ValueError("Either radius or k_neighbors must be specified.")

    patch_points = points[indices]
    if len(patch_points) > num_points_per_patch:
        # ランダムにサブサンプリング
        random_indices = np.random.choice(len(patch_points), num_points_per_patch, replace=False)
        patch_points = patch_points[random_indices]
    elif len(patch_points) < num_points_per_patch:
        # パディング (例: ゼロパディング)
        padding = np.zeros((num_points_per_patch - len(patch_points), points.shape[1]), dtype=np.float32)
        patch_points = np.concatenate([patch_points, padding], axis=0)

    return patch_points

def random_transform(points, rotation_range=(-180, 180), translation_range=(-0.2, 0.2)):
    """点群にランダムな剛体変換を適用する関数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(device) # (1, N, 3 or 4)

    # ランダムな回転 (度数法)
    rot_x = np.random.uniform(*rotation_range)
    rot_y = np.random.uniform(*rotation_range)
    rot_z = np.random.uniform(*rotation_range)
    angles = torch.tensor([[rot_x, rot_y, rot_z]]).float().to(device)

    # ランダムな並進
    translation = torch.tensor([[np.random.uniform(*translation_range) for _ in range(3)]]).float().to(device)

    # 回転行列の生成 (ここでは簡易的にX軸回転のみ)
    cos_x = torch.cos(torch.deg2rad(angles[:, 0]))
    sin_x = torch.sin(torch.deg2rad(angles[:, 0]))
    rotation_matrix = torch.eye(3).unsqueeze(0).to(device)
    rotation_matrix[:, 1, 1] = cos_x
    rotation_matrix[:, 1, 2] = -sin_x
    rotation_matrix[:, 2, 1] = sin_x
    rotation_matrix[:, 2, 2] = cos_x

    # 点群の座標部分のみを回転
    rotated_coords = torch.matmul(points_tensor[:, :, :3], rotation_matrix.transpose(1, 2))

    # 並進を適用
    transformed_points = torch.cat([rotated_coords + translation.unsqueeze(1), points_tensor[:, :, 3:]], dim=-1)

    return transformed_points.squeeze(0).cpu().numpy()

class ContrastivePointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, num_points_per_patch=1024, patch_radius=0.1):
        self.file_list = file_list
        self.num_points_per_patch = num_points_per_patch
        self.patch_radius = patch_radius

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        original_points = load_ply(file_path)

        # ランダムに中心点を選択
        center_index_A = np.random.randint(len(original_points))
        center_index_B = np.random.randint(len(original_points))

        # パッチをサンプリング
        patch_A = sample_patch(original_points, center_index_A, self.num_points_per_patch, radius=self.patch_radius)
        patch_B = sample_patch(original_points, center_index_B, self.num_points_per_patch, radius=self.patch_radius)

        # 各パッチにランダムな変換を適用 (ポジティブペア)
        transformed_patch_A = random_transform(patch_A)
        transformed_patch_B = random_transform(patch_B)

        # PyTorch Tensorに変換
        transformed_patch_A = torch.from_numpy(transformed_patch_A).float()
        transformed_patch_B = torch.from_numpy(transformed_patch_B).float()

        # 必要に応じて正規化などの前処理を追加

        return transformed_patch_A, transformed_patch_B

# データセットの作成とDataLoader
file_list =  os.listdir('/mnt/c/Users/matsu/SICK/pay-10-bucks/dataset/constractive_dataset')# あなたのPLYファイルのリスト
contrastive_dataset = ContrastivePointCloudDataset(file_list)
contrastive_dataloader = torch.utils.data.DataLoader(contrastive_dataset, batch_size=32, shuffle=True, num_workers=4)

# 学習ループ内で dataloader をイテレートしてパッチペアを取得
# for batch_idx, (patch_A, patch_B) in enumerate(contrastive_dataloader):
#     # patch_A と patch_B を DGCNNLocalFeatureExtractor に入力して特徴量を抽出
#     features_A = model(patch_A.permute(0, 2, 1)) # (B, C, N) に変換
#     features_B = model(patch_B.permute(0, 2, 1))
#     # コントラスティブ損失を計算
#     # ...