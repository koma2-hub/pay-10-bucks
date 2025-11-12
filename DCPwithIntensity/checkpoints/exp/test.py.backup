#make_dataset.py
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
    centroid = np.sum(pcd[:, :3], axis=0)/pcd.shape[0]
    return centroid

def get_pcd_around_centroid(pcd, overlap_num):
    centroid = calculate_centroid(pcd)
    distance = (pcd[:,:3]-centroid)**2
    distance = np.sum(distance, axis=1)
    indices = np.argsort(distance)[:overlap_num:]
    return indices
"""
データのランダムな回転
各軸に対して-45~45°回転させる
"""
def random_rotation(pcd, rotation_range=(-45, 45)):
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


def random_transform(pcd, translation_range = (-10, 10)):
    translation_vector = np.array([np.random.uniform(translation_range[0], translation_range[1]),np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1])])
    translation_vector = translation_vector*10
    pcd_translated = np.copy(pcd)
    pcd_translated[:,:3] = pcd_translated[:,:3] + translation_vector
    return pcd, pcd_translated, translation_vector


#点群を可視化する関数
def visualize_pcd(pcd_list):
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
    def __init__(self, processed_dir, intensity=True):
        self.processed_dir = processed_dir
        self.intensity = intensity
        
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
        # data = torch.load(file_path) # デフォルト(weights_only=True)でOK
        # ★ 解決策1と併用する場合は weights_only=False のままにしてください
        data = torch.load(file_path, weights_only=False) # ← 互換性のため残すのが安全


        # 3. 辞書からテンソルを取得 (すでにテンソルになっている)
        src_pcd = data['src_pcd'].transpose(0, 1)
        tgt_pcd = data['tgt_pcd'].transpose(0, 1)
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_ts = torch.as_tensor(data['t_ts']).float().view(1, 3)
        R_ts = torch.as_tensor(data['R_ts']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(1, 3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        euler_ts = torch.as_tensor(data['euler_ts']).float()

        # 輝度値を除外する場合
        if not self.intensity:
            src_pcd = src_pcd[:, :3]
            tgt_pcd = tgt_pcd[:, :3]
            
        return src_pcd, tgt_pcd, R_st, t_st, R_ts, t_ts, euler_st, euler_ts


def sample_knn_patches_with_overlap(points_full, 
                                      num_points_k=1024, 
                                      overlap_ratio_range=(0.3, 0.5), 
                                      max_retries=20):
    """
    点群から k-NN パッチを2つサンプリングする。
    2つのパッチの重複率が指定範囲内になるまでリトライ処理を行う。
    
    Args:
        points_full (np.ndarray): 元の点群 (N, D)
        num_points_k (int): 各パッチの点数 (K)
        overlap_ratio_range (tuple): 望ましい重複率の範囲 (min, max)
        max_retries (int): 最大試行回数

    Returns:
        points_src (np.ndarray): 1つ目のパッチ (K, D)
        points_tgt (np.ndarray): 2つ目のパッチ (K, D)
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
    #    警告を出し、最後にサンプリングしたペア（重複率が範囲外）をそのまま返す
    
    # print(f"警告: {max_retries}回試行しましたが、重複率 {min_overlap}-{max_overlap} の"
    #       f"ペアを見つけられませんでした。最後の試行（重複率: {actual_overlap_ratio:.2f}）を使用します。")
    
    if indices_src is None or indices_tgt is None:
        # 1回もループが回らなかった場合 (N < K など) の安全装置
        # (このコードでは N < K は上で処理済みのため、通常ここには来ない)
        return sample_knn_patches_with_overlap(points_full, num_points_k, (0.0, 1.0), 1)

    points_tgt_patch = points_full[indices_tgt, :]
    points_src_patch = points_full[indices_src, :]
    
    return points_src_patch, points_tgt_patch


# (1) 関数が 'output_dir' を引数で受け取るように変更
def make_dcpDataset(sample_point, k, overlap_ratio, data_path, output_dir, intensity=True):
    
    # (2) 保存先ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = [] # (これはもう使わないが、他の部分で参照しているなら残す)
    
    file_names = os.listdir(data_path)
    print(file_names)
    
    # (3) ペアのファイル名を一意にするためのカウンターを初期化
    pair_counter = 0

    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        
        if pcd is None: 
            print(f"警告: スキップします: {file_path}")
            continue
            
        ds_pcd = downsample_pcd(pcd, sample_point)
        
        for i in range(4):
            src_pcd, tgt_pcd = sample_knn_patches_with_overlap(
                ds_pcd, num_points_k=k, overlap_ratio_range=overlap_ratio
            )
            
            _, transformed_tgt, R_st, translation_st, \
            R_ts, translation_ts, euler_st, euler_ts = rigit_transform(tgt_pcd)

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
            
            # (4) ★ torch.save の前に output_filename を定義 ★
            output_filename = os.path.join(output_dir, f"pair_{pair_counter:06d}.pt")
            
            torch.save(data_dict, output_filename)
            
            # (5) カウンターを増やす
            pair_counter += 1

    print(f"合計 {pair_counter} ペアのデータを {output_dir} に書き出しました。")
    # (この関数は Dataset を返さないので return は不要、またはパスのリストを返す)# test.py 内

# (1) tqdm はファイルの先頭で import してください
# from tqdm import tqdm 

def make_dcpDataset(sample_point, k, overlap_ratio, data_path, output_dir, intensity=True):
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_names = os.listdir(data_path)
    print(f"対象ファイル: {file_names}")
    
    pair_counter = 0

    # ★ 変更点 1: tqdmプログレスバーを初期化 ★
    # 1ファイルあたり4ペア生成すると仮定
    total_pairs_to_generate = len(file_names) * 4
    pbar = tqdm(total=total_pairs_to_generate, desc="Generating data pairs")

    # (file_names のループ自体は tqdm でラップしない)
    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        
        if pcd is None: 
            print(f"警告: スキップします: {file_path}")
            # このファイル分 (4ペア) を進捗から引くか、
            # update(4) して強制的に進める (ここでは後者)
            pbar.update(4) 
            continue
            
        ds_pcd = downsample_pcd(pcd, sample_point)
        
        for i in range(4): # 1ファイルあたり4ペア生成
            src_pcd, tgt_pcd = sample_knn_patches_with_overlap(
                ds_pcd, num_points_k=k, overlap_ratio_range=overlap_ratio
            )
            
            _, transformed_tgt, R_st, translation_st, \
            R_ts, translation_ts, euler_st, euler_ts = rigit_transform(tgt_pcd)

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
            
            # ★ 変更点 2: ペアを1つ保存するたびに進捗バーを1つ進める ★
            pbar.update(1)

    # ★ 変更点 3: ループ終了後にプログレスバーを閉じる ★
    pbar.close()

    print(f"完了: 合計 {pair_counter} ペアのデータを {output_dir} に書き出しました。")

#make_dcpDataset(sample_point=4096, k=1024, overlap_num=512, data_path="/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/raw/additional")


        

def test_function(pcd):
    pcd, pcd_translated, t = random_transform(pcd)
    pcd, pcd_rotated, rotation_matrix = random_rotation(pcd)
    print('translation', t)
    print('rotation', rotation_matrix)

    visualize_pcd([pcd,pcd_translated])
    visualize_pcd([pcd, pcd_rotated])


def rigit_transform(pcd):
    pcd ,pcd_rotated, rotation_matrix, euler_zyx = random_rotation(pcd)
    _, pcd_rotated_tranfromed , transform_vector = random_transform(pcd_rotated)
    rotation_src_to_tgt = rotation_matrix
    rotation_tgt_to_src = rotation_src_to_tgt.T
    transform_tgt_to_src = -rotation_tgt_to_src.dot(transform_vector)
    euler_src_to_tgt = euler_zyx
    euler_tgt_to_src = -euler_zyx[::-1]
    return pcd, pcd_rotated_tranfromed, rotation_src_to_tgt,transform_vector,\
            rotation_tgt_to_src, transform_tgt_to_src, euler_src_to_tgt, euler_tgt_to_src


"""
path = "/mnt/d/SICK/pay-10-bucks/data/mylabs/raw/"
# (1) 保存先のディレクトリパスを定義
output_dir = "/mnt/d/SICK/pay-10-bucks/data/mylabs/processed/" 
overlap_range = (0.3, 0.5)

# (2) output_dir を引数として渡す
make_dcpDataset(
    sample_point=4096, 
    k=1024, 
    overlap_ratio=overlap_range, 
    data_path=path,
    output_dir=output_dir
)

# --- 1. データセットの準備 ---
processed_path = "/mnt/d/SICK/pay-10-bucks/data/mylabs/processed"

# ★ ディスクから読み込むDatasetを初期化 ★
train_dataset = DCPDataset(processed_path, intensity=True)
 
print(f"データセット準備完了。合計ペア数: {len(train_dataset)}")

# --- 2. DataLoader の作成 ---
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)
# --- 3. データの取り出し (NumPyでの利用) ---
# 最初の1バッチを取り出してみる
# iter(train_loader).next() と同じ
batch = next(iter(train_loader))

# データセットが返す8つの値が、バッチ化されて取り出される
# (batch_size, num_points, channels)
src_pcd_batch, tgt_pcd_batch, R_st_batch, t_st_batch, \
R_ts_batch, t_ts_batch, euler_st_batch, euler_ts_batch = batch
 
print(f"\n--- 最初のバッチ ---")
print(f"ソース点群 (Torch Tensor) の形状: {src_pcd_batch.shape}")
print(f"変換行列 (Torch Tensor) の形状: {R_st_batch.shape}")
print(f"並進 (Torch Tensor) の形状: {t_st_batch.shape}")

# ★ NumPyで使いたい場合 ★
# .numpy() を呼び出すだけです
src_pcd_numpy = src_pcd_batch.numpy()
print(f"\nNumPy配列に変換した形状: {src_pcd_numpy.shape}")

"""