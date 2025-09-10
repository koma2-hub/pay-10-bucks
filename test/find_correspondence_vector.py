import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. 準備フェーズ (必要な関数とモデル定義)
# ==============================================================================

# (ご自身の定義した load_ply, knn, create_intensity_histogram, 
#  SubNetwork, SiameseNetwork をここに貼り付けてください)

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
    # ... (あなたのコード)
    num_points = x.shape[0]
    k = min(k, num_points)
    if k <= 1: k = 1
    x_coords = x[:, :3]
    x_norm_sq = np.sum(x_coords**2, axis=1, keepdims=True)
    dot_product = np.matmul(x_coords, x_coords.T)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    return indices

class SubNetwork(nn.Module):
    # ... (あなたのモデル定義)
    def __init__(self, input_dim=16, embedding_dim=32):
        super(SubNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, x): return self.fc(x)

class SiameseNetwork(nn.Module):
    # ... (あなたのモデル定義)
    def __init__(self, sub_network):
        super(SiameseNetwork, self).__init__()
        self.sub_network = sub_network
    def forward(self, input1, input2):
        return self.sub_network(input1), self.sub_network(input2)


# ==============================================================================
# 2. メイン処理 (データ準備とマッチング)
# ==============================================================================

# --- パラメータ設定 ---
# (ご自身のコードから引用)
SRC_PATH = "/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/src_icn_fps.ply"
TGT_PATH = "/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/tgt_icn_fps.ply"
MODEL_PATH = '/mnt/c/Users/matsu/SICK/pay-10-bucks/models/vector_siamese_model.pth'

K_NEIGHBORS = 32
#近傍点の個数と一致
INPUT_DIM = 32
EMBEDDING_DIM = 32
DISTANCE_THRESHOLD = 0.2

# --- データ読み込みと記述子計算 ---
print("Loading point clouds and computing descriptors...")
src_icn_fps = load_ply(SRC_PATH)
tgt_icn_fps = load_ply(TGT_PATH)

src_knn = knn(src_icn_fps, k=K_NEIGHBORS)
tgt_knn = knn(tgt_icn_fps, k=K_NEIGHBORS)

"""
ソース点群における各近傍点の輝度値を取得する。
取得した輝度値はリストに保存する。
"""
src_vector = []
for i in range(src_knn.shape[0]):
    #各点の近傍点を取得
    neighbor_indices = src_knn[i]
    #近傍点の輝度値を取得
    src_intensity = src_icn_fps[neighbor_indices, -1]
    #輝度値を保存
    src_vector.append(src_intensity)

"""
ターゲット点群における各近傍点の輝度値を取得する。
取得した輝度値はリストに保存する。
"""
tgt_vector = []
for i in range(tgt_knn.shape[0]):
    neighbor_indices = tgt_knn[i]
    tgt_intensity = tgt_icn_fps[neighbor_indices, -1]
    tgt_vector.append(tgt_intensity)

#取得した輝度値はnumpy配列に変換する
src_vector = np.array(src_vector, dtype=np.float32)
tgt_vector = np.array(tgt_vector, dtype=np.float32)

print(src_vector)
print(tgt_vector)

# --- モデルのロードと設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sub_net = SubNetwork(input_dim=INPUT_DIM, embedding_dim=EMBEDDING_DIM)
model = SiameseNetwork(sub_network=sub_net)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# --- ★★★【修正箇所】Ratio Testを導入した効率的なマッチング処理 ★★★ ---
print("Matching descriptors using the trained model with Ratio Test...")
# (NumPy配列をTorchテンソルに変換する部分は同じ)
src_vector_tensor = torch.from_numpy(src_vector).to(device)
tgt_vector_tensor = torch.from_numpy(tgt_vector).to(device)
print(src_vector_tensor)
print(tgt_vector_tensor)

correspondences = []
# Ratio Testのための閾値
RATIO_THRESHOLD = 0.8 # この値は0.7～0.8あたりで調整します

with torch.no_grad():
    # 全てのヒストグラムを一度にモデルに通し、特徴ベクトル（Embedding）を計算
    # model.sub_network を直接使うことで、SiameseNetworkのforwardを2回呼ぶ無駄を省く
    emb_src = model.sub_network(src_vector_tensor)
    emb_tgt = model.sub_network(tgt_vector_tensor)

    # ソース点群の各点について、ターゲット点群で最も近い点を探す
    for i in range(len(emb_src)):
        # i番目のソース特徴ベクトルと、全てのターゲット特徴ベクトルとの距離を計算
        distances = F.pairwise_distance(emb_src[i].unsqueeze(0), emb_tgt)
        
        # 最も距離が近い点のインデックスと、その距離を取得
        best_dist, best_idx = torch.min(distances, dim=0)
        
        # 距離が閾値以下なら、対応点としてインデックスのペアを保存
        if best_dist.item() < DISTANCE_THRESHOLD:
            correspondences.append([i, best_idx.item()])

print(f"マッチング完了。 {len(correspondences)} 個の対応点が見つかりました。")


# ==============================================================================
# 3. 結果の可視化 (open3dを使用)
# ==============================================================================
if correspondences:
    # (可視化部分は前回の回答と同じ)
    o3d_pcd_a = o3d.geometry.PointCloud()
    o3d_pcd_a.points = o3d.utility.Vector3dVector(src_icn_fps[:, :3])
    o3d_pcd_a.paint_uniform_color([0, 0.651, 0.929])

    o3d_pcd_b = o3d.geometry.PointCloud()
    o3d_pcd_b.points = o3d.utility.Vector3dVector(tgt_icn_fps[:, :3])
    o3d_pcd_b.paint_uniform_color([1, 0.706, 0])

    corr_set = o3d.utility.Vector2iVector(correspondences)
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        o3d_pcd_a, o3d_pcd_b, corr_set)
    line_set.paint_uniform_color([1, 0, 0])
    
    print("可視化ウィンドウを表示します。閉じるには 'q' キーを押してください。")
    o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_b, line_set],
                                      window_name="Point Cloud Correspondences")
    
# ==============================================================================
# 3. 結果の可視化と位置合わせ (前回のコードの続き)
# ==============================================================================
if len(correspondences) > 3: # RANSACは最低3つの対応点を必要とします
    print("\n--- 対応点を用いた位置合わせを開始 ---")

    # --- open3d用のデータ形式に変換 ---
    # 対応点のインデックスリスト (Vector2iVector形式)
    corr_set = o3d.utility.Vector2iVector(correspondences)
    
    # XYZ座標のみを持つPointCloudオブジェクト (色情報なし)
    pcd_a_xyz = o3d.geometry.PointCloud()
    pcd_a_xyz.points = o3d_pcd_a.points
    pcd_b_xyz = o3d.geometry.PointCloud()
    pcd_b_xyz.points = o3d_pcd_b.points
    
    # --- RANSACによる位置合わせの実行 ---
    # 距離の閾値 (この距離内にある対応点をインライアとみなす)
    ransac_distance_threshold = 0.1  # ★点群のスケールに合わせて調整してください (例: 2cm)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=pcd_a_xyz,
        target=pcd_b_xyz,
        corres=corr_set,
        max_correspondence_distance=ransac_distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    # 計算された変換行列を取得
    transformation_matrix = result_ransac.transformation
    print("計算された4x4変換行列:\n", transformation_matrix)

    # --- 結果の可視化 ---
    # ソース点群に変換行列を適用して、位置合わせされた新しい点群を作成
    pcd_a_transformed = o3d.geometry.PointCloud(pcd_a_xyz)
    pcd_a_transformed.transform(transformation_matrix)

    # 元の点群（色付き）で可視化
    o3d_pcd_a.paint_uniform_color([0, 0.651, 0.929]) # ソース(青)
    o3d_pcd_b.paint_uniform_color([1, 0.706, 0])   # ターゲット(黄)
    o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_b,],
                                      window_name="Point Cloud Before Transformed")
    
    # 変換後のソース点群を緑色で表示
    pcd_a_transformed_color = o3d.geometry.PointCloud(pcd_a_transformed)
    pcd_a_transformed_color.paint_uniform_color([0, 1, 0]) # 変換後(緑)
    
    print("\n位置合わせ後の結果を可視化します。")
    print("緑色の点群（変換後ソース）が黄色の点群（ターゲット）に重なっていれば成功です。")
    o3d.visualization.draw_geometries(
        [pcd_a_transformed_color, o3d_pcd_b],
        window_name="位置合わせ後の結果 (緑が黄に重なるか確認)"
    )

else:
    print("対応点が3個未満のため、位置合わせをスキップします。")
