import sys
import os
import numpy as np
import random
import torch 
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm
from util import load_ply, downsample_pcd,knn


def calculate_centroid(pcd):
    """(N, D) の点群から重心 (D,) を計算"""
    # 座標 (XYZ) のみで重心を計算
    centroid = np.mean(pcd[:, :3], axis=0)
    return centroid

def get_pcd_around_centroid(pcd, overlap_num):
    # (この関数は現在使用されていないようです)
    centroid = calculate_centroid(pcd)
    distance = (pcd[:,:3]-centroid)**2
    distance = np.sum(distance, axis=1)
    indices = np.argsort(distance)[:overlap_num:]
    return indices

def random_rotation(pcd, rotation_range=(-30, 30)):
    # (この関数は変更ありません)
    #ランダムな回転行列の生成
    angle_x = np.random.uniform(*rotation_range)
    angle_y = np.random.uniform(*rotation_range)
    angle_z = np.random.uniform(*rotation_range)

    sinx = np.sin(np.deg2rad(angle_x))
    cosx = np.cos(np.deg2rad(angle_x))
    siny = np.sin(np.deg2rad(angle_y))
    cosy = np.cos(np.deg2rad(angle_y))
    sinz = np.sin(np.deg2rad(angle_z))
    cosz = np.cos(np.deg2rad(angle_z))

    #各軸の回転行列
    rotation_x = np.array([[1, 0, 0],
                           [0, cosx, -sinx],
                           [0, sinx, cosx]])
    rotation_y = np.array([[cosy, 0, siny],
                           [0, 1, 0],
                           [-siny, 0, cosy]])
    rotation_z = np.array([[cosz, -sinz, 0],
                           [sinz, cosz, 0],
                           [0, 0, 1]])
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    pcd_rotated = pcd.copy()
    #点群が輝度値を持つときは座標のみ変換する
    #点群が輝度値を持たないときはそのまま変換する
    euler = np.asarray([angle_z, angle_y, angle_x])
    if(pcd.shape[1] != 3):
        pcd_rotated[:,:3] = pcd_rotated[:, :3].dot(rotation_matrix.T)
        pcd = np.random.permutation(pcd)
        pcd_rotated = np.random.permutation(pcd_rotated)
        return pcd, pcd_rotated, rotation_matrix, euler
    else:
        pcd = np.random.permutation(pcd)
        pcd_rotated = np.random.permutation(pcd_rotated)
        pcd_rotated = pcd_rotated.dot(rotation_matrix.T)
        return pcd, pcd_rotated, rotation_matrix, euler


def random_transform(pcd, translation_range = (-1, 1)):
    translation_vector = np.array([np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1])])
    
    pcd_translated = np.copy(pcd)
    pcd_translated[:,:3] = pcd_translated[:,:3] + translation_vector
    return pcd, pcd_translated, translation_vector


#点群を可視化する関数
def visualize_pcd(pcd_list):
    # (変更ありません)
    pcd_o3d_list = []
    for pcd in pcd_list:
        pointcloud = pcd[:,:3]
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pointcloud)
        rgb = [random.uniform(0,1) for i in range(3)]
        pcd_obj.paint_uniform_color(rgb)
        pcd_o3d_list.append(pcd_obj)
    o3d.visualization.draw_geometries(pcd_o3d_list, window_name="Point Cloud")


def visualize_pcd_overlap(pcd1,pcd2,overlap_pcd):
    # (変更ありません)
    pointcloud1 = pcd1[:, :3]
    pointcloud2 = pcd2[:, :3]
    pointcloud3 = overlap_pcd[:, :3]
    #open3dのPointCloudオブジェクトを生成
    pcd_o3d1 = o3d.geometry.PointCloud()
    pcd_o3d1.points = o3d.utility.Vector3dVector(pointcloud1)
    pcd_o3d1.paint_uniform_color([1, 0, 0])
    #open3dのPointCloudオブジェクトを生成
    pcd_o3d2 = o3d.geometry.PointCloud()
    pcd_o3d2.points = o3d.utility.Vector3dVector(pointcloud2)
    pcd_o3d2.paint_uniform_color([0, 0, 1])
    #
    pcd_o3d3 = o3d.geometry.PointCloud()
    pcd_o3d3.points = o3d.utility.Vector3dVector(pointcloud3)
    pcd_o3d3.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd_o3d1, pcd_o3d2, pcd_o3d3],
                                        window_name="Point Cloud")


class DCPDataset(Dataset):
    """
    前処理済みの .pt ファイルを読み込むデータセットクラス
    """
    def __init__(self, processed_dir, intensity=True, n_points=1024):
        self.processed_dir = processed_dir
        self.intensity = intensity
        self.n_points = n_points # n_points を受け取る (main.py から渡されるため)
        
        # processed_dir にある .pt ファイルのリストを作成
        self.file_paths = []
        for f in os.listdir(self.processed_dir):
            if f.endswith(".pt"):
                self.file_paths.append(os.path.join(self.processed_dir, f))
        
        self.file_paths.sort() # 順序を保証

    def __len__(self):
        # データセットの総数 (ファイルの数) を返す
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # weights_only=False は pickle 化されたNumpy配列を読み込むために必要
        data = torch.load(file_path, weights_only=False) 

        # 3. 辞書からテンソルを取得
        # (make_dcpDataset で (C, L) 形式で保存されているため、transpose 不要)
        src_pcd = torch.as_tensor(data['src_pcd']).float()
        tgt_pcd = torch.as_tensor(data['tgt_pcd']).float()
        
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_ts = torch.as_tensor(data['t_ts']).float().view(1, 3)
        R_ts = torch.as_tensor(data['R_ts']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(1, 3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        euler_ts = torch.as_tensor(data['euler_ts']).float()

        # 輝度値を除外する場合
        if not self.intensity:
            src_pcd = src_pcd[:3, :] # (C, L) 形式なので、:3 でスライス
            tgt_pcd = tgt_pcd[:3, :]
            
        return src_pcd, tgt_pcd, R_st, t_st, R_ts, t_ts, euler_st, euler_ts


def sample_knn_patches_with_overlap(points_full, 
                                        num_points_k=1024, 
                                        overlap_ratio_range=(0.3, 0.5), 
                                        max_retries=20):
    """
    (この関数は変更ありません)
    """
    
    N, D = points_full.shape
    K = num_points_k
    min_overlap, max_overlap = overlap_ratio_range

    if N < K:
        # 点が足りない場合は、両方とも同じ点群を返す（重複率100%）
        indices = np.random.choice(N, K, replace=True)
        patch = points_full[indices, :]
        return patch, patch

    # 1. 座標データと KDTree を構築 (これは1回だけでよい)
    coords_xyz = points_full[:, :3]
    tree = KDTree(coords_xyz)

    # 最後に試行したインデックスを保持するため
    indices_src = None
    indices_tgt = None

    # 2. リトライループ
    for _ in range(max_retries):
        # 2a. 1つ目のパッチ (tgt) をサンプリング
        center_index_1 = np.random.randint(0, N)
        center_point_1 = coords_xyz[center_index_1]
        _, indices_tgt = tree.query(center_point_1, k=K)

        # 2b. 2つ目のパッチ (src) をサンプリング
        # (1つ目のパッチの近傍から中心を選ぶ)
        center_index_2 = np.random.choice(indices_tgt)
        center_point_2 = coords_xyz[center_index_2]
        _, indices_src = tree.query(center_point_2, k=K)

        # 2c. 重複率を計算
        set_tgt = set(indices_tgt)
        set_src = set(indices_src)
        num_intersection = len(set_tgt.intersection(set_src))
        actual_overlap_ratio = num_intersection / K

        # 2d. 重複率が指定範囲内かチェック
        if min_overlap <= actual_overlap_ratio <= max_overlap:
            # 成功！インデックスからデータを抽出して早期リターン
            points_tgt_patch = points_full[indices_tgt, :]
            points_src_patch = points_full[indices_src, :]
            return points_src_patch, points_tgt_patch

    # 3. フォールバック (max_retries 回試行しても失敗した場合)
    if indices_src is None or indices_tgt is None:
        indices = np.random.choice(N, K, replace=True)
        patch = points_full[indices, :]
        return patch, patch

    points_tgt_patch = points_full[indices_tgt, :]
    points_src_patch = points_full[indices_src, :]
    
    return points_src_patch, points_tgt_patch


def make_dcpDataset(sample_point, k, overlap_ratio, data_path, output_dir, intensity=True):
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_names = os.listdir(data_path)
    print(f"対象ファイル: {file_names}")
    
    pair_counter = 0

    total_pairs_to_generate = len(file_names) * 8
    pbar = tqdm(total=total_pairs_to_generate, desc="Generating data pairs")

    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        
        if pcd is None: 
            pbar.update(4) 
            continue
            
        ds_pcd = downsample_pcd(pcd, sample_point)
        
        for i in range(8): # 1ファイルあたり8ペア生成
            # 1. (N, D) 形式でパッチをサンプリング
            src_pcd, tgt_pcd = sample_knn_patches_with_overlap(
                ds_pcd, num_points_k=k, overlap_ratio_range=overlap_ratio
            )
            
            # ★ 変更点 2: 両方のパッチを正規化 (センタリング) ★
            centroid_src = calculate_centroid(src_pcd)
            centroid_tgt = calculate_centroid(tgt_pcd)
            
            src_pcd[:, :3] = src_pcd[:, :3] - centroid_src
            tgt_pcd[:, :3] = tgt_pcd[:, :3] - centroid_tgt
            
            # 3. 正規化された tgt_pcd にランダム変換を適用
            #    rigit_transform は (N, D) を受け取り (N, D) を返す
            _, transformed_tgt, R_st, translation_st, \
            R_ts, translation_ts, euler_st, euler_ts = rigit_transform(tgt_pcd)
            
            # 4. (N, D) -> (C, L) 形式 (D, N) に転置して保存
            #    (D=4, N=1024)
            src_pcd = src_pcd.T
            transformed_tgt = transformed_tgt.T
            
            data_dict = {
                'src_pcd': src_pcd.astype(np.float32),
                'tgt_pcd': transformed_tgt.astype(np.float32),
                'R_st': R_st.astype(np.float32),
                't_st': translation_st.astype(np.float32),
                'R_ts': R_ts.astype(np.float32),
                't_ts': translation_ts.astype(np.float32),
                'euler_st': euler_st.astype(np.float32),
                'euler_ts': euler_ts.astype(np.float32)
            }
            
            output_filename = os.path.join(output_dir, f"pair_{pair_counter:06d}.pt")
            
            torch.save(data_dict, output_filename)
            
            pair_counter += 1
            
            pbar.update(1)

    pbar.close()
    print(f"完了: 合計 {pair_counter} ペアのデータを {output_dir} に書き出しました。")


def test_function(pcd):
    # (変更ありません)
    pcd, pcd_translated, t = random_transform(pcd)
    pcd, pcd_rotated, rotation_matrix = random_rotation(pcd)
    print('translation', t)
    print('rotation', rotation_matrix)

    visualize_pcd([pcd,pcd_translated])
    visualize_pcd([pcd, pcd_rotated])


def rigit_transform(pcd):
    # (この関数は random_rotation と random_transform を呼ぶだけなので変更不要)
    pcd ,pcd_rotated, rotation_matrix, euler_zyx = random_rotation(pcd)
    _, pcd_rotated_tranfromed , transform_vector = random_transform(pcd_rotated)
    rotation_src_to_tgt = rotation_matrix
    rotation_tgt_to_src = rotation_src_to_tgt.T
    transform_tgt_to_src = -rotation_tgt_to_src.dot(transform_vector)
    euler_src_to_tgt = euler_zyx
    euler_tgt_to_src = -euler_zyx[::-1]
    return pcd, pcd_rotated_tranfromed, rotation_src_to_tgt,transform_vector,\
           rotation_tgt_to_src, transform_tgt_to_src, euler_src_to_tgt, euler_tgt_to_src


# --- メイン実行部 ---
# (実行パスを修正)
path = "/mnt/d/SICK/pay-10-bucks/data/mylabs/raw/"
output_dir = "/mnt/d/SICK/pay-10-bucks/DCPwithIntensity/dataset" 
overlap_range = (0.3, 0.5)

# (1) データセットの再生成
print(f"データセットを {output_dir} に生成します...")
make_dcpDataset(
    sample_point=4096, 
    k=1024, 
    overlap_ratio=overlap_range, 
    data_path=path,
    output_dir=output_dir,
    intensity=True # ★ intensity=True を渡すように修正
)
print("データセット生成完了。")

# (2) 生成されたデータセットの読み込みテスト
print("\n--- データセット読み込みテスト ---")
processed_path = output_dir

try:
    train_dataset = DCPDataset(processed_path, intensity=True)
    print(f"データセット準備完了。合計ペア数: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0 # ★ main.py 以外で num_workers > 0 を使うとエラーになることがあるため 0 に変更
    )
    
    batch = next(iter(train_loader))

    src_pcd_batch, tgt_pcd_batch, R_st_batch, t_st_batch, \
    R_ts_batch, t_ts_batch, euler_st_batch, euler_ts_batch = batch
    
    print(f"\n--- 最初のバッチ ---")
    print(f"ソース点群 (Torch Tensor) の形状: {src_pcd_batch.shape}")
    print(f"ターゲット点群 (Torch Tensor) の形状: {tgt_pcd_batch.shape}")
    print(f"変換行列 (Torch Tensor) の形状: {R_st_batch.shape}")
    print(f"並進 (Torch Tensor) の形状: {t_st_batch.shape}")
    
    # ★ 正規化された並進ベクトルの値を確認
    print(f"並進ベクトルのサンプル値 (t_st):\n {t_st_batch[:2]}")

except Exception as e:
    print(f"データセットのテスト中にエラーが発生しました: {e}")
    print("トレースバック:")
    import traceback
    traceback.print_exc()