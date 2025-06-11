# dataset.py
import os,sys
sys.path.append(os.pardir)
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader # 名前を区別するため
from sklearn.neighbors import NearestNeighbors # sample_patch用

# --- 補助関数 (以前の定義から) ---
def load_ply(filename):
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
            
            points = np.array([list(map(float, l.split())) for l in lines[header_index+1:]])
            if points.shape[1] < 4:
                raise ValueError(f"期待される列数に満たないデータが検出されました: {points.shape[1]}列")
            
            # [x, y, z, intensity] の形に整形
            points = np.concatenate([points[:, :3], points[:, -1].reshape(-1, 1)], axis=1)
            
            return points
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました(load_ply): {e}")
        return None

def l2_norm(a, b):
    return ((a - b) ** 2).sum(axis=1)

def farthest_point_sampling(pcd, k, metrics=l2_norm):
    indices = np.zeros(k, dtype=np.int32)
    distances = np.zeros((k, pcd.shape[0]), dtype=np.float32)
    indices[0] = np.random.randint(len(pcd))
    farthest_point = pcd[indices[0]]
    min_distances = metrics(farthest_point, pcd)
    distances[0, :] = min_distances
    for i in range(1, k):
        indices[i] = np.argmax(min_distances)
        farthest_point = pcd[indices[i]]
        distances[i, :] = metrics(farthest_point, pcd)
        min_distances = np.minimum(min_distances, distances[i, :])
    return indices

# --- パッチサンプリングとデータ拡張の補助関数 ---
# dataset.py (sample_patch_from_point_cloud 関数内)

def sample_patch_from_point_cloud(points_np, num_points_per_patch, patch_radius=None, k_neighbors=None):
    """
    点群から中心点周辺のパッチをサンプリングする関数。
    points_np: (N, C) NumPy配列 (Nは点数、Cは特徴量次元)
    """
    if points_np.shape[0] == 0: # 空の点群の場合の追加ガード
        return np.zeros((num_points_per_patch, points_np.shape[1]), dtype=np.float32)

    center_point_index = np.random.randint(len(points_np))
    center = points_np[center_point_index, :3] # 座標のみで近傍探索

    # --- 修正箇所 ---
    if patch_radius is not None: # ここを radius -> patch_radius
        # 半径内の点を抽出
        distances = np.linalg.norm(points_np[:, :3] - center, axis=1)
        indices = np.where(distances <= patch_radius)[0] # ここも radius -> patch_radius
    elif k_neighbors is not None:
        # K近傍点を抽出 (効率的な実装にはKD-treeなどを使用)
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_np[:, :3])
        _, indices = knn.kneighbors([center])
        indices = indices[0]
    else:
        # デフォルトはK近傍点を使用 (K_NEIGHBORS のような固定値がなければエラーになる可能性)
        # ここでは、もし radius も k_neighbors も指定されなかった場合のデフォルト動作を定義
        # 例として、patch_radius を num_points_per_patch に応じて自動設定することも考えられます
        # または、エラーを発生させるか、事前にどちらかを指定するように強制する
        # ここでは、k_neighbors を num_points_per_patch に設定する簡易的な対応
        k_neighbors = num_points_per_patch # パッチ内の点数と同じ数の近傍を探す
        knn_finder = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_np[:, :3])
        _, indices = knn_finder.kneighbors([center])
        indices = indices[0]

    patch_points = points_np[indices]
    
    # ... (以降のパディング・サブサンプリングロジックは変更なし) ...
    if len(patch_points) > num_points_per_patch:
        random_indices = np.random.choice(len(patch_points), num_points_per_patch, replace=False)
        patch_points = patch_points[random_indices]
    elif len(patch_points) < num_points_per_patch:
        # パディング (ここではゼロパディング)
        padding = np.zeros((num_points_per_patch - len(patch_points), points_np.shape[1]), dtype=np.float32)
        patch_points = np.concatenate([patch_points, padding], axis=0)

    # パッチ内の点の中心を原点に移動
    patch_points[:, :3] -= np.mean(patch_points[:, :3], axis=0)

    return patch_points


def random_transform(points_np, rotation_range=(-180, 180), translation_range=(-0.1, 0.1), noise_std=0.01, dropout_ratio=(0.0, 0.2)):
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

    # 4. ドロップアウト (点の削除)
    if dropout_ratio[1] > 0:
        num_points = transformed_points.shape[0]
        drop_ratio = np.random.uniform(dropout_ratio[0], dropout_ratio[1])
        num_drop = int(num_points * drop_ratio)
        if num_drop > 0:
            drop_indices = np.random.choice(num_points, num_drop, replace=False)
            transformed_points = np.delete(transformed_points, drop_indices, axis=0)
            
            # ドロップアウト後に点数が減る可能性があるので、パディングで元に戻す
            if transformed_points.shape[0] < num_points:
                padding_size = num_points - transformed_points.shape[0]
                padding = np.zeros((padding_size, transformed_points.shape[1]), dtype=np.float32)
                transformed_points = np.concatenate([transformed_points, padding], axis=0)


    return transformed_points


# --- データセットクラス ---
# ContrastivePointCloudDataset は InMemoryDataset を継承せず、直接 Dataset を継承
# 各データファイルから動的にパッチを生成するため
class ContrastivePointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_points_per_patch=1024, patch_radius=0.1, transform=None):
        self.root_dir = root_dir
        self.num_points_per_patch = num_points_per_patch
        self.patch_radius = patch_radius
        self.transform = transform # オプションでデータ拡張を適用

        # rawディレクトリ内のPLYファイルリストを取得
        self.file_list = [os.path.join(root_dir, 'raw', f) 
                          for f in os.listdir(os.path.join(root_dir, 'raw')) 
                          if f.endswith('.ply')]
        
        if not self.file_list:
            raise RuntimeError(f"No .ply files found in {os.path.join(root_dir, 'raw')}")

    def __len__(self):
        return len(self.file_list) * 10 # 各ファイルから複数回パッチを生成する想定 (調整可能)

    def __getitem__(self, idx):
        # 1つのファイルからパッチを生成する
        file_path_idx = idx % len(self.file_list)
        file_path = self.file_list[file_path_idx]
        
        original_points = load_ply(file_path) # (N, 4) - xyz intensity
        
        # NumPy配列をfloat32にキャスト (モデルのRuntimeError対策)
        if original_points is not None:
            original_points = original_points.astype(np.float32)
        else:
            # エラー処理。ここでは、ダミーのデータで返すか、例外を再度投げるか。
            # 学習ループが中断しないように、適切なサイズでゼロテンソルを返すのが一般的
            print(f"Error loading {file_path}. Returning zero tensor.")
            return torch.zeros((self.num_points_per_patch, 4), dtype=torch.float32), \
                   torch.zeros((self.num_points_per_patch, 4), dtype=torch.float32)

        # ランダムに中心点を選択しパッチをサンプリング
        # コントラスティブ学習では、同じ点群から異なるビューのパッチを生成することがポジティブペアの基本
        # ここでは、同じ点群から2つの独立したパッチをサンプリングします
        patch_A_raw = sample_patch_from_point_cloud(
            original_points, 
            self.num_points_per_patch, 
            patch_radius=self.patch_radius
        )
        patch_B_raw = sample_patch_from_point_cloud(
            original_points, 
            self.num_points_per_patch, 
            patch_radius=self.patch_radius 
        )
       

        # それぞれのパッチにランダムな変換を適用 (データ拡張)
        transformed_patch_A = random_transform(patch_A_raw)
        transformed_patch_B = random_transform(patch_B_raw)
        
        # PyTorch Tensorに変換
        transformed_patch_A = torch.from_numpy(transformed_patch_A).float()
        transformed_patch_B = torch.from_numpy(transformed_patch_B).float()

        return transformed_patch_A, transformed_patch_B