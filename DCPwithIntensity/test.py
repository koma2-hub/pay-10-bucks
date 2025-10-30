import sys
import os
import numpy as np
import random
import torch 
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from utils import load_ply, downsample_pcd,knn


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
    if(pcd.shape[1] != 3):
        pcd_rotated[:,:3] = pcd_rotated[:, :3].dot(rotation_matrix)
        return pcd, pcd_rotated, rotation_matrix
    else:
        pcd_rotated = pcd_rotated.dot(rotation_matrix)
        return pcd, pcd_rotated, rotation_matrix

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
    点群のペアとその変換行列を返すデータセットクラス
    """
    def __init__(self, pairs, intensity=True):
        self.pairs = pairs
        self.intensity = intensity

    def __len__(self):
        #データセットの総数を返す
        return len(self.pairs)
    
    def __getitem__(self, idx):
        #指定されたインデックスのデータを取得
        src_pcd, tgt_pcd, translation = self.pairs[idx]

        #numpy配列をPyTorchテンソルに変換
        src_pcd = torch.from_numpy(src_pcd.astype(np.float32))
        tgt_pcd = torch.from_numpy(tgt_pcd.astype(np.float32))
        #下記もnumpyで保存すると思うので後で変更
        translation = torch.tensor(translation, dtype=torch.float32)
        #輝度値を含めて返すかどうか場合分け
        if self.intensity:
            return src_pcd, tgt_pcd, translation
        else:
            return src_pcd[:,:3], tgt_pcd[:,:3], translation

def make_dcpDataset(sample_point, k, overlap_num, data_path):
    datasets = []
    #サンプルを抽出するファイルの受け取り
    file_names = os.listdir(data_path)
    print(file_names)
    """
    生の各点群ファイルを読み込み以下の処理を行う
    1.指定した点数にダウンサンプリング
    2.各点の近傍点のインデックスを取得
    3.点群の重心計算,重心周りの近傍点のインデックスの取得
    4.重心周りの点を含む点群を探索
    5.重心周りの点を含む２つの点群の片方を並進,回転変換をさせる
    6.２つの点群とその剛体変換の行列を返す
    """
    for file in file_names:
        #点群のファイル名の取得とその点群の読み込み
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        #点群のダウンサンプリング
        ds_pcd = downsample_pcd(pcd, sample_point)
        #近傍点探索
        knn_indices = knn(ds_pcd, k)
        #重心計算と重心周りの点群の取得
        center_pcd_indices = get_pcd_around_centroid(ds_pcd, overlap_num=overlap_num)
        center_pcd_indices_broadcast = np.ones((sample_point, overlap_num))*center_pcd_indices
        #すべての点の近傍点と重心周りの点群のインデックスを結合
        #(ユニークな値を取り出したときk点になるような点群同士でマッチングさせる)
        #(→元の点群に重心周りの点群がすべて含まれるということ)
        knn_indices = np.concatenate((knn_indices, center_pcd_indices_broadcast), axis = 1)
        knn_indices = knn_indices.astype(int)
        candidates = []
        for i, indice in enumerate(knn_indices):
            unique_indice = np.unique(indice)
            if(unique_indice.shape[0] == 1024):
                candidates.append(i)

        if(len(candidates) < 2):
            break
        #ユニークな点がk点であるような点群同士をペアにして片方剛体返還
        pcd1_index = np.unique(knn_indices[random.randint(0, len(candidates))])
        pcd2_index = np.unique(knn_indices[random.randint(0, len(candidates))])
        pcd1 = ds_pcd[pcd1_index]
        pcd2 = ds_pcd[pcd2_index]
        pcd2, pcd2_rotated, rotation_matrix = random_rotation(pcd2)
        pcd_overlap = ds_pcd[center_pcd_indices]
        #デバックようの可視化
        visualize_pcd_overlap(pcd1, pcd2_rotated, pcd_overlap)

#make_dcpDataset(sample_point=4096, k=1024, overlap_num=512, data_path="/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/raw/additional")



def test_dataset(sample_point, k, overlap_num, data_path):
    #データセットとして返すリスト
    dataset = []
    file_names = os.listdir(data_path)
    print(file_names)
    for file in file_names:
        #ファイルパスを取得
        file_path = os.path.join(data_path, file)
        #点群データのロード
        pcd = load_ply(file_path)
        #ダウンサンプリング
        pcd_ds = downsample_pcd(pcd, downsample_point=sample_point)
        #近傍点探索
        knn_indices = knn(pcd_ds, k=k)
        #ペアのインデックスを保存
        pair_index = []
        for i,indice_src in enumerate(knn_indices):
            for j, indice_tgt in enumerate(knn_indices):
                if(j < i):
                    break
                overlap_count = np.in1d(indice_src, indice_tgt)
                if(overlap_count.sum() == overlap_num):
                    pair_index.append((i,j))
                    print(i,j)
        

#test_dataset(sample_point=4096, k=1024, overlap_num=256, data_path="/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/raw/additional")
for i in range(10):
    for j in range(10):
        print(i,j)
        if(j < i):
            break
        




"""

sample_point = 4096
k = 1024
overlap_num = 512


pcd = load_ply("/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/raw/mylab001.ply")
ds_pcd = downsample_pcd(pcd, sample_point)
knn_indices = knn(ds_pcd[:, :3], k)


center_pcd_indices = get_pcd_around_centroid(ds_pcd, overlap_num)
center_pcd_indices = np.ones((sample_point, overlap_num))*center_pcd_indices

print(center_pcd_indices.shape)
print(knn_indices.shape)

#なぜかfloat形に変換されるので後で治す
knn_indices = np.concatenate((knn_indices, center_pcd_indices) , axis=1)
knn_indices = knn_indices.astype(int)
candidates = []
for i, indice in enumerate(knn_indices):
    unique_indice = np.unique(indice)
    if(unique_indice.shape[0] == 1024):
        candidates.append(i)

print(len(candidates))

print(candidates[5])
idx = candidates[5]
print(knn_indices[idx])


for idx in candidates:
    pcd_index = knn_indices[idx]
    pcd_index = pcd_index.astype(np.int64)
    pcd = ds_pcd[pcd_index]
    visualize_pcd(pcd, [0,1,0])


center_pcd_indices = center_pcd_indices.astype(int)
overlap_pcd = ds_pcd[get_pcd_around_centroid(ds_pcd, overlap_num)]

idx0 = candidates[0]
idx1 = candidates[1]

idx2 = candidates[2]
idx3 = candidates[3]

pcd_index0 = np.unique(knn_indices[idx0])
pcd_index1 = np.unique(knn_indices[idx1])
pcd_index2 = np.unique(knn_indices[idx2])
pcd_index3 = np.unique(knn_indices[idx3])

pcd0 = ds_pcd[pcd_index0]
pcd1 = ds_pcd[pcd_index1]
pcd2 = ds_pcd[pcd_index2]
pcd3 = ds_pcd[pcd_index3]

print(pcd0.shape)
print(pcd1.shape)
print(pcd2.shape)
print(pcd3.shape)

visualize_pcd(pcd0, pcd1, overlap_pcd)
visualize_pcd(pcd0, pcd2, overlap_pcd)
visualize_pcd(pcd0, pcd3, overlap_pcd)
visualize_pcd(pcd1, pcd2, overlap_pcd)
visualize_pcd(pcd1, pcd3, overlap_pcd)
visualize_pcd(pcd2, pcd3, overlap_pcd)


"""
    