# main_registration.py
import os
import torch
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import time # 処理時間計測用

# model.py と dataset.py から必要なものをインポート
from model import DGCNNLocalFeatureExtractor # DGCNNLocalFeatureExtractorのみ
from dataset import load_ply, farthest_point_sampling # load_ply関数のみ

# --- 設定 ---
MODEL_PATH = "dgcnn_local_feature_extractor_contrastive.pth" # 学習済みモデルのパス
EMB_DIMS = 1024 # 学習時のemb_dimsに合わせる
PROJECTION_DIM = 128 # 学習時のprojection_dimに合わせる (推論時は使わないがモデル定義に必要)
K_NEIGHBORS = 20 # DGCNNのK (学習時と同じ値)
NUM_POINTS_PER_PATCH = 1024
# 位置合わせ対象の点群ファイルパス (実際のファイルパスに合わせてください)
SOURCE_PLY_PATH = "./test_data/lab1.ply" # 例: ロボットの現在のスキャン
TARGET_PLY_PATH = "./test_data/lab2.ply" # 例: ロボットの既知の地図、または以前のスキャン

# --- 特徴量マッチングとRANSACのパラメータ ---
FEATURE_MATCHING_THRESHOLD = 0.7# 特徴量間の最大距離 (コサイン類似度なら0.8など)
RANSAC_DISTANCE_THRESHOLD = 0.05 # RANSACのインライア閾値 (座標の単位に合わせる)
RANSAC_MAX_ITERATIONS = 100000 # RANSACの最大試行回数
RANSAC_CONFIDENCE = 0.999 # RANSACの信頼度

# --- ICPのパラメータ ---
ICP_THRESHOLD = 0.02 # ICPの対応点探索距離 (RANSACより厳しくすることが多い)
ICP_MAX_ITERATIONS = 200 # ICPの最大繰り返し回数

def random_transform(points_np, rotation_range=(0, 30)):
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
    return transformed_points

def load_model(model_path, emb_dims, projection_dim, k_neighbors, device):
    """学習済みモデルをロードする関数"""
    model = DGCNNLocalFeatureExtractor(k=k_neighbors, emb_dims=emb_dims, projection_dim=projection_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 推論モードに設定
    model.to(device)
    print(f"Model loaded from {model_path} successfully.")
    return model

def extract_features_from_pcd(model, pcd_np, num_points, device):
    """点群からローカル特徴量を抽出する関数"""
    # pcd_np は (N, 4) のNumPy配列を想定
    # モデルの入力は (B, N_fixed, 4) を想定
    
    # 点数が異なる場合はサンプリング/パディングを行う
    if pcd_np.shape[0] < num_points:
        # 足りない場合はランダムに点を複製してパディング
        print(f"Warning: Point cloud has {pcd_np.shape[0]} points, less than {num_points}. Duplicating points.")
        indices = np.random.choice(pcd_np.shape[0], num_points, replace=True)
        processed_pcd_np = pcd_np[indices]
    elif pcd_np.shape[0] > num_points:
        # 多すぎる場合はFPSでサブサンプリング
        print(f"Warning: Point cloud has {pcd_np.shape[0]} points, more than {num_points}. Subsampling using FPS.")
        fps_indices = farthest_point_sampling(pcd_np[:, :3], num_points) # 座標のみでFPS
        processed_pcd_np = pcd_np[fps_indices]
    else:
        processed_pcd_np = pcd_np
    
    # PyTorchテンソルに変換し、バッチ次元を追加 (1, N_fixed, 4)
    pcd_tensor = torch.from_numpy(processed_pcd_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # モデルのforwardは (B, N_fixed, emb_dims) を返す
        features = model(pcd_tensor) # この時model.eval()なのでローカル特徴量が返る
    
    # 形状を (N_fixed, emb_dims) に戻す
    return features.squeeze(0).cpu().numpy(), processed_pcd_np[:, :3] # 特徴量と(処理後の)座標

# main_registration.py (または dataset.py の find_correspondences 関数)

def find_correspondences(features1, points1, features2, points2, threshold):
    """
    特徴量間のNNマッチングを行い、対応点を見つける。
    features1: (N1, D) - sourceのローカル特徴量
    points1: (N1, 3) - sourceの座標
    features2: (N2, D) - targetのローカル特徴量
    points2: (N2, 3) - targetの座標
    threshold: 特徴量間の最大距離閾値
    
    Returns:
        source_corr_points: (M, 3) NumPy配列 - sourceの対応点の座標
        target_corr_points: (M, 3) NumPy配列 - targetの対応点の座標
        correspondence_indices: (M, 2) NumPy配列 - sourceとtargetの対応点のインデックスペア
    """
    # KDTree for efficient nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(features2)
    distances, indices = nbrs.kneighbors(features1) # indices: (N1, 1)

    # 双方向NNマッチング
    nbrs_rev = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(features1)
    distances_rev, indices_rev = nbrs_rev.kneighbors(features2)

    correspondences_list = [] # (src_idx, tgt_idx) のリストを保持
    source_corr_points = []
    target_corr_points = []

    for i in range(features1.shape[0]):
        j = indices[i, 0]
        if distances[i, 0] < threshold:
            if indices_rev[j, 0] == i:
                correspondences_list.append((i, j)) # ここでインデックスをタプルで追加

    source_corr_points = np.array([points1[c[0]] for c in correspondences_list], dtype=np.float64)
    target_corr_points = np.array([points2[c[1]] for c in correspondences_list], dtype=np.float64)
    
    # correspondence_indices は、o3d.utility.Vector2iVector に直接渡すために (M, 2) numpy.int32 にする
    # または、correspondences_list (Iterable) を直接渡す
    
    print(f"Found {len(correspondences_list)} initial correspondences.")
    return source_corr_points, target_corr_points, correspondences_list # リストのまま返す

def visualize_registration_result(source_pcd_o3d, target_pcd_o3d, transformed_source_pcd_o3d, title="Registration Result"):
    """位置合わせ結果を可視化する"""
    source_pcd_o3d.paint_uniform_color([1, 0.706, 0]) # 黄色 (Source)
    target_pcd_o3d.paint_uniform_color([0, 0.651, 0.929]) # 青色 (Target)
    transformed_source_pcd_o3d.paint_uniform_color([1, 0, 0]) # 赤色 (Transformed Source)

    # 可視化
    o3d.visualization.draw_geometries([source_pcd_o3d, target_pcd_o3d, transformed_source_pcd_o3d], window_name=title)
    print("Close the visualization window to continue.")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 学習済みモデルのロード
    model = load_model(MODEL_PATH, EMB_DIMS, PROJECTION_DIM, K_NEIGHBORS, device)

    # 2. 2つの点群データの読み込み
    print(f"Loading source point cloud from {SOURCE_PLY_PATH}...")
    source_pcd_raw = load_ply(SOURCE_PLY_PATH)
    if source_pcd_raw is None:
        print("Failed to load source point cloud. Exiting.")
        exit()
    source_pcd_raw = source_pcd_raw.astype(np.float32)

    print(f"Transforming source point cloud")
    target_pcd_raw = random_transform(source_pcd_raw)
    if target_pcd_raw is None:
        print("Failed to load target point cloud. Exiting.")
        exit()
    target_pcd_raw = target_pcd_raw.astype(np.float32)

    # Open3Dの点群オブジェクトを作成 (可視化用)
    source_o3d_original = o3d.geometry.PointCloud()
    source_o3d_original.points = o3d.utility.Vector3dVector(source_pcd_raw[:, :3])

    target_o3d_original = o3d.geometry.PointCloud()
    target_o3d_original.points = o3d.utility.Vector3dVector(target_pcd_raw[:, :3])

    # 3. ローカル特徴量の抽出
    print("Extracting features from source point cloud...")
    start_time = time.time()
    source_features, source_pcd_processed_coords = extract_features_from_pcd(model, source_pcd_raw, NUM_POINTS_PER_PATCH, device)
    end_time = time.time()
    print(f"Source features extracted in {end_time - start_time:.4f} seconds. Shape: {source_features.shape}")

    print("Extracting features from target point cloud...")
    start_time = time.time()
    target_features, target_pcd_processed_coords = extract_features_from_pcd(model, target_pcd_raw, NUM_POINTS_PER_PATCH, device)
    end_time = time.time()
    print(f"Target features extracted in {end_time - start_time:.4f} seconds. Shape: {target_features.shape}")

    # 4. 特徴量マッチング
    print("Finding correspondences based on features...")
    start_time = time.time()
    # 戻り値に correspondence_list を追加 (numpy配列ではなくPythonのリスト)
    src_corr_pts_np, tgt_corr_pts_np, correspondences_list = find_correspondences(
        source_features, source_pcd_processed_coords,
        target_features, target_pcd_processed_coords,
        threshold=FEATURE_MATCHING_THRESHOLD
    )
    end_time = time.time()
    print(f"Correspondences found in {end_time - start_time:.4f} seconds.")
    
    if len(src_corr_pts_np) < 3: # RANSACには最低3点必要
        print("Not enough correspondences found for RANSAC. Aborting registration.")
        exit()

    # --- 修正箇所: Vector2iVector の作成方法 ---
    # `correspondences_list` はタプルのリストなので、Iterable として直接渡す
    # Open3Dの古いAPIを使うと警告が出る可能性がありますが、TypeErrorは回避できます。
    # この correspondence_set は、o3d.pipelines.registration.registration_ransac_based_on_correspondence
    # のような関数に渡すものであり、o3d.pipelines.registration.registration_ransac_based_on_feature_matching
    # には直接渡しません。
    # したがって、この行はコメントアウトまたは削除しても良いですが、TypeErrorの解消を優先します。
    # 実際には、この行は RANSAC の引数として使われないため、不要な場合はコメントアウト/削除してください。

    # --- 修正箇所終了 ---


    
    # 5. RANSACによる初期変換推定
    print("Estimating initial transform using RANSAC...")
    start_time = time.time()
    
    # Open3DのRegistrationResultオブジェクトを生成
    # RANSACのために、o3d.utility.Vector3dVector に変換
    source_o3d_processed = o3d.geometry.PointCloud()
    source_o3d_processed.points = o3d.utility.Vector3dVector(source_pcd_processed_coords)
    
    target_o3d_processed = o3d.geometry.PointCloud()
    target_o3d_processed.points = o3d.utility.Vector3dVector(target_pcd_processed_coords)

    # RANSACの対応点探索距離は、特徴量空間ではなく、実際の3D空間での距離です。
    # これが、対応点が「インライア」であるとみなされる最大距離です。
    # initial_transform は単位行列から開始
    
    # Open3D 0.16.0 以降の RANSAC は、キーポイントと記述子から直接実行するAPIが多い
    # ここでは、簡略化のため、`registration_ransac_based_on_correspondence` に近いロジックを使用
    # ただし、直接対応点を与えるAPIは非推奨になっているため、特徴量マッチングの結果を
    # 適切に o3d.pipelines.registration.registration_ransac_based_on_feature_matching
    # に渡す必要があります。
    # 簡略化のため、ここでは`src_corr_pts_np`と`tgt_corr_pts_np`を直接ICPの初期値として使う
    # または、Procrustes analysisでRANSACを自作するが、Open3Dを使う方が確実。

    # Open3D 0.16.0以降の推奨されるRANSAC使用例（キーポイントと特徴量から）
    # ただし、現状のDGCNNLocalFeatureExtractorはパッチ単位なので、
    # 全点に対する特徴量が出力されます。キーポイントを選ぶロジックが必要になります。
    # ここでは、マッチングされた点群を直接使う簡易RANSACを想定します。
    
    # 簡易RANSAC (Open3DのAPIを直接使わない) -> procrustes_analysis_numpy を利用
    # これは一般的なRANSACの実装ではなく、Procrustes analysisなので注意
    
    # Open3DのAPIを使用する例 (RANSACの結果からICPへ)
    # RANSACは`source_o3d_processed`と`target_o3d_processed`に対して実行
    # `src_corr_pts_np` と `tgt_corr_pts_np` は対応する点群のサブセット
    # これらのサブセット間の変換をRANSACで推定
    
    # RANSAC for rigid transformation
    # Open3d 0.16.0+
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        max_iteration=RANSAC_MAX_ITERATIONS,
        confidence=RANSAC_CONFIDENCE
    )
    
    # correspondences from find_correspondences are 2D arrays, need to map to o3d.pipelines.registration.CorrespondenceArray
    # this requires the original point cloud indices
    # We will pass the pre-matched points to an ICP-like function that takes initial transform
    # The RANSAC part is the most challenging to do robustly without detailed O3D usage for feature-based RANSAC.
    
    # As an alternative, let's use `estimate_rigid_transform` from scipy's `Rotation.align_vectors` or similar for RANSAC's inner loop
    # Or simply: if your correspondences are reliable enough (due to good DGCNN features), you might directly compute the transform.
    
    # For a robust RANSAC from *matched point pairs*, Open3D has `compute_transformation_based_on_correspondences`
    # which is used within RANSAC pipelines.
    
    # Let's simplify and use Open3D's direct RANSAC for a very basic test.
    # This requires `o3d.pipelines.registration.Feature` objects if matching based on features
    # For point-to-point correspondence, it's simpler.

    # Convert NumPy arrays to Open3D point clouds for RANSAC
    # This is a bit tricky if using `registration_ransac_based_on_feature_matching`
    # which expects original point clouds + features.
    # Let's assume we use simpler RANSAC or trust our feature matching.

    # Simple RANSAC: Directly compute transform from a subset of correspondences and check inliers
    # This loop is for demonstration, actual Open3D RANSAC is more optimized.
    best_inlier_count = 0
    best_initial_transform = np.identity(4)

    # Prepare correspondence set for Open3D (from indices)
    # This is assuming you extracted the original indices when finding correspondences
    # For now, let's just use the source_corr_pts_np and target_corr_pts_np directly with `TransformationEstimationPointToPoint`
    
    # RANSAC based on corresponding point pairs
    # Note: `registration_ransac_based_on_correspondence` is older/simpler
    # For feature-based RANSAC, it needs `o3d.pipelines.registration.registration_ransac_based_on_feature_matching`
    # and feature descriptors for o3d.registration.Feature()
    
    # For simplicity, let's use the transformation estimation on the *matched* points directly
    # and then apply ICP. This assumes DGCNN features are good enough for direct matching.
    # A true RANSAC would sample points from the *original* point clouds, not just the matched ones.
    
    # Let's assume a function that estimates transform from correspondences exists (e.g., from a robust RANSAC)
    # As a fallback, we can use a direct point-to-point estimation as the 'initial_transform'
    # if the feature matching is very strong.
    
    # For a proper RANSAC:
    # 1. Select 3 random points from `source_features` and `target_features` (and their coords).
    # 2. Compute transformation from these 3 pairs.
    # 3. Apply transform to all `source_features` and count inliers based on `FEATURE_MATCHING_THRESHOLD`.
    # 4. Repeat many times and choose the best.
    
    # Open3D's API often expects `o3d.pipelines.registration.Feature` objects.
    # Our DGCNN model gives a (N, C) numpy array.
    
    # Simplified approach for RANSAC's initial transform:
    # We will directly use `o3d.pipelines.registration.registration_ransac_based_on_feature_matching`
    # which implies we need to convert our features to `o3d.registration.Feature`
    
    # Convert features to Open3D Feature objects
    source_features_o3d = o3d.pipelines.registration.Feature()
    source_features_o3d.data = source_features.T # (D, N) expected

    target_features_o3d = o3d.pipelines.registration.Feature()
    target_features_o3d.data = target_features.T # (D, N) expected

    # Perform RANSAC
    # max_correspondence_distance: マッチングされた点間の最大距離 (特徴空間ではなく3D空間)
    # estimation_method: 点群間の変換を推定する方法 (Point-to-Point)
    # ransac_n: RANSACでサンプリングする点の最小数 (剛体変換は通常3点)
    # criteria: RANSACの収束基準 (最大繰り返し回数、信頼度)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_o3d_processed,              # source PointCloud
        target=target_o3d_processed,              # target PointCloud
        source_feature=source_features_o3d,       # source Feature
        target_feature=target_features_o3d,       # target Feature
        mutual_filter=True,                       # <-- この引数を追加（通常True）
        max_correspondence_distance=RANSAC_DISTANCE_THRESHOLD, # <-- 名前を修正
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=RANSAC_MAX_ITERATIONS,
            confidence=RANSAC_CONFIDENCE
        )
    )
    
    initial_transform = result_ransac.transformation
    end_time = time.time()
    print(f"RANSAC estimated initial transform in {end_time - start_time:.4f} seconds.")
    print("Initial Transform from RANSAC:\n", initial_transform)

    # 6. ICPによる精密位置合わせ
    print("Performing ICP for fine registration...")
    start_time = time.time()
    
    # ICPはRANSACで得られた初期変換を初期値として使用
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_o3d_processed, target_o3d_processed, 
        ICP_THRESHOLD, # 対応点探索距離
        initial_transform, # RANSACの結果を初期値として渡す
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS)
    )
    
    final_transform = reg_p2p.transformation
    end_time = time.time()
    print(f"ICP completed in {end_time - start_time:.4f} seconds.")
    print("Final Transform from ICP:\n", final_transform)

    # 7. 結果の可視化
    print("Visualizing registration results...")
    
    # 初期位置の可視化 (RANSAC適用前)
    transformed_source_pcd_ransac = source_o3d_original.transform(initial_transform)
    # visualize_registration_result(source_o3d_original, target_o3d_original, transformed_source_pcd_ransac, title="RANSAC Initial Registration")

    # 最終位置合わせの可視化 (ICP適用後)
    transformed_source_pcd_final = source_o3d_original.transform(final_transform)

    print(f"Source points count: {len(source_o3d_original.points)}")
    print(f"Target points count: {len(target_o3d_original.points)}")
    print(f"Transformed source points count: {len(transformed_source_pcd_final.points)}")
    visualize_registration_result(source_o3d_original, target_o3d_original, transformed_source_pcd_final, title="ICP Final Registration")

    print("Registration process complete.")