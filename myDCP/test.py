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

def random_rotation(pcd, rotation_range=(-np.pi/6, np.pi/6)):
    # (この関数は変更ありません)
    #ランダムな回転行列の生成
    angle_x = np.random.uniform(*rotation_range)
    angle_y = np.random.uniform(*rotation_range)
    angle_z = np.random.uniform(*rotation_range)

    sinx = np.sin(angle_x)
    cosx = np.cos(angle_x)
    siny = np.sin(angle_y)
    cosy = np.cos(angle_y)
    sinz = np.sin(angle_z)
    cosz = np.cos(angle_z)

    #各軸の回転行列
    rotation_x = np.array([[1, 0,    0],
                           [0, cosx, -sinx],
                           [0, sinx, cosx]])
    rotation_y = np.array([[cosy, 0, siny],
                           [0,    1, 0],
                           [-siny, 0, cosy]])
    rotation_z = np.array([[cosz, -sinz, 0],
                           [sinz, cosz,  0],
                           [0,    0,     1]])
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    pcd_rotated = pcd.copy()
    #点群が輝度値を持つときは座標のみ変換する
    #点群が輝度値を持たないときはそのまま変換する
    euler = np.asarray([angle_z, angle_y, angle_x])
    rotation_st_from_euler = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])
    
    if(pcd.shape[1] != 3):
        pcd_rotated[:,:3] = rotation_st_from_euler.apply(pcd[:,:3])
        return pcd, pcd_rotated,  rotation_matrix, rotation_st_from_euler, euler
    else:
        pcd_rotated = pcd_rotated.apply(pcd)
        return pcd, pcd_rotated, rotation_matrix, euler
    

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


data_path = "/mnt/d/SICK/pay-10-bucks/data/mylabs/raw/mylab020.ply"
pcd = load_ply(data_path)
pcd, pcd_rotated, rotation_matrix,rotation_matrix_from_euler, euler = random_rotation(pcd)
print(rotation_matrix_from_euler.as_matrix())
print(rotation_matrix)
visualize_pcd([pcd, pcd_rotated])
pcd1 = rotation_matrix_from_euler.apply(pcd[:,:3])
visualize_pcd([pcd1, pcd_rotated])
rotation_matrix_inverse = rotation_matrix_from_euler.inv()
print(rotation_matrix_inverse.as_matrix())
pcd_reversed = rotation_matrix_inverse.apply(pcd_rotated[:,:3])
visualize_pcd([pcd, pcd_reversed])


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