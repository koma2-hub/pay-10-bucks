import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

# ==============================================================================
# 1. 準備フェーズ (ユーザー定義関数とモデル定義)
# ==============================================================================
#
# ご提示いただいた自作関数群をここに貼り付けます
#
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
  

def knn(x: np.ndarray, k: int):
    # (ご提示のコードをここに貼り付け)
    num_points = x.shape[0]
    k = min(k, num_points)
    if k <= 1: k = 1
    x_coords = x[:, :3] # 距離計算はXYZ座標のみで行う
    x_norm_sq = np.sum(x_coords**2, axis=1, keepdims=True)
    dot_product = np.matmul(x_coords, x_coords.T)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    return indices

def create_intensity_histogram(point_cloud_intensity_data, bins=16, density=True):
    # (ご提示のコードをここに貼り付け)
    if len(point_cloud_intensity_data) == 0:
        return np.zeros(bins).astype(np.float32)
    hist, _ = np.histogram(
        point_cloud_intensity_data,
        bins=bins,
        range=(0.0, 1.0), # 輝度値が0.0~1.0に正規化されていると仮定
        density=density
    )
    return hist.astype(np.float32)

#
# 訓練時に使用したモデル定義をここに貼り付けます
#
class SubNetwork(nn.Module):
    def __init__(self, input_dim=16, embedding_dim=64): # ★ご自身のモデルに合わせて修正
        super(SubNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, x): return self.fc(x)

class SiameseNetwork(nn.Module):
    def __init__(self, sub_network):
        super(SiameseNetwork, self).__init__()
        self.sub_network = sub_network
    def forward(self, input1, input2):
        return self.sub_network(input1), self.sub_network(input2)

#
# 新しく定義するヘルパー関数
#
def compute_descriptors(pcd, indices, bins):
    """点群の各点について、周辺の輝度値ヒストグラム（特徴記述子）を計算する"""
    num_points = pcd.shape[0]
    intensities = pcd[:, 3] # 4列目を輝度値として利用
    
    descriptors = []
    for i in range(num_points):
        neighbor_indices = indices[i]
        neighbor_intensities = intensities[neighbor_indices]
        hist = create_intensity_histogram(neighbor_intensities, bins=bins)
        descriptors.append(hist)
        
    return np.array(descriptors)

# ==============================================================================
# 2. メイン処理 (マッチングの実行)
# ==============================================================================

# --- パラメータ設定 ---
# ファイルパス
PCD_A_PATH = '/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/lab1_icn_fps.ply'  # ★ソース点群ファイルのパスに修正してください
PCD_B_PATH = '/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/lab2_icn_fps.ply'  # ★ターゲット点群ファイルのパスに修正してください
MODEL_PATH = '/mnt/c/Users/matsu/SICK/pay-10-bucks/kICN_model/siamese_model.pth' # ★学習済みモデルのパス

# モデルのパラメータ
INPUT_DIM = 16      # ★訓練時のbinsと同じ値
EMBEDDING_DIM = 32  # ★訓練時と同じ値

# 特徴記述子のパラメータ
K_NEIGHBORS = 64        # ★データセット作成時のkと同じ値
HIST_BINS = INPUT_DIM   # モデルの入力次元と一致させる

# マッチングのパラメータ
DISTANCE_THRESHOLD = 0.90 # ★評価で見つけた最適な値を基準に調整

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- モデルのロード ---
sub_net = SubNetwork(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)
model = SiameseNetwork(sub_network=sub_net)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# --- 点群データのロード ---
print("Loading point clouds...")
pcd_a = load_ply(PCD_A_PATH)
pcd_b = load_ply(PCD_B_PATH)
if pcd_a is None or pcd_b is None:
    exit("点群ファイルの読み込みに失敗しました。")

# --- k-NNインデックスの事前計算 ---
print("Computing k-NN for point clouds...")
indices_a = knn(pcd_a, k=K_NEIGHBORS)
indices_b = knn(pcd_b, k=K_NEIGHBORS)

# --- 特徴記述子の計算 ---
print("Computing descriptors for point cloud A...")
desc_a = compute_descriptors(pcd_a, indices_a, HIST_BINS)
print("Computing descriptors for point cloud B...")
desc_b = compute_descriptors(pcd_b, indices_b, HIST_BINS)

# --- モデルを使って記述子を比較し、マッチング ---
print("Matching descriptors using the trained model...")
desc_a_tensor = torch.from_numpy(desc_a).to(device)
desc_b_tensor = torch.from_numpy(desc_b).to(device)

correspondences = []
with torch.no_grad():
    emb_a, _ = model(desc_a_tensor, desc_a_tensor)
    emb_b, _ = model(desc_b_tensor, desc_b_tensor)
    
    for i in range(len(emb_a)):
        distances = F.pairwise_distance(emb_a[i].unsqueeze(0), emb_b)
        best_dist, best_idx = torch.min(distances, dim=0)
        
        if best_dist.item() < DISTANCE_THRESHOLD:
            correspondences.append([i, best_idx.item()])
            
print(f"マッチング完了。 {len(correspondences)} 個の対応点が見つかりました。")

# ==============================================================================
# 3. 結果の可視化 (open3dを使用)
# ==============================================================================
if correspondences:
    # --- open3d用のデータ形式に変換 ---
    # XYZ座標のみを抽出
    points_a_xyz = pcd_a[:, :3]
    points_b_xyz = pcd_b[:, :3]

    # open3dのPointCloudオブジェクトを作成
    o3d_pcd_a = o3d.geometry.PointCloud()
    o3d_pcd_a.points = o3d.utility.Vector3dVector(points_a_xyz)
    o3d_pcd_a.paint_uniform_color([0, 0.651, 0.929]) # 青色

    o3d_pcd_b = o3d.geometry.PointCloud()
    o3d_pcd_b.points = o3d.utility.Vector3dVector(points_b_xyz)
    o3d_pcd_b.paint_uniform_color([1, 0.706, 0]) # オレンジ色

    # --- 対応点から線を作成 ---
    corr_set = o3d.utility.Vector2iVector(correspondences)
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        o3d_pcd_a, o3d_pcd_b, corr_set)
    line_set.paint_uniform_color([1, 0, 0]) # 線の色を赤に

    print("可視化ウィンドウを表示します。閉じるには 'q' キーを押してください。")
    # 可視化
    o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_b, line_set],
                                      window_name="Point Cloud Correspondences")
else:
    print("対応点が見つからなかったため、可視化をスキップします。")