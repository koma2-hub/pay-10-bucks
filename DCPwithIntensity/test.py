import sys
import os
import numpy as np
import torch 
import open3d as o3d
from utils import load_ply, downsample_pcd, knn

def calculate_centroid(pcd):
    centroid = np.sum(pcd[:, :3], axis=0)/pcd.shape[0]
    return centroid

def get_pcd_around_centroid(pcd, overlap_num):
    centroid = calculate_centroid(pcd)
    #print(centroid)
    distance = (pcd[:,:3]-centroid)**2
    #print(distance)
    distance = np.sum(distance, axis=1)
    #print(distance)
    indices = np.argsort(distance)[:overlap_num]
    #print(indices)
    return indices

#点群を可視化する関数
def visualize_pcd(pcd):
    #点群が輝度値を用いている場合は輝度値をスライスする
    pointcloud = pcd[:, :3]
    #open3dのPointCloudオブジェクトを生成
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pointcloud)
    pcd_o3d.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd_o3d],
                                      window_name="Point Cloud")
    
    
sample_point = 4096
k = 1024
overlap_num = 512

def DCPDataset(sample_point, k, overlap_num, data_path):
    file_names = os.listdir(data_path)
    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        ds_pcd = downsample_pcd(pcd, sample_point)
        indices = knn(ds_pcd, k)

pcd = load_ply("/mnt/d/SICK/pay-10-bucks/data/mylabs/raw/mylab001.ply")
ds_pcd = downsample_pcd(pcd, sample_point)
knn_indices = knn(ds_pcd, k)
visualize_pcd(pcd)

center_pcd_indices = get_pcd_around_centroid(ds_pcd, overlap_num)
center_pcd_indices = np.ones((sample_point, overlap_num))*center_pcd_indices
#なぜかfloat形に変換されるので後で治す
knn_indices = np.concatenate((knn_indices, center_pcd_indices) , axis=1)

candidates = []
for i, indice in enumerate(knn_indices):
    unique_indice = np.unique(indice)
    if(unique_indice.shape[0] <= 1074):
        candidates.append(i)

print(len(candidates))

print(candidates[5])
idx = candidates[5]
print(knn_indices[idx])


for idx in candidates:
    pcd_index = knn_indices[idx]
    pcd_index = pcd_index.astype(np.int64)
    pcd = ds_pcd[pcd_index]
    #visualize_pcd(pcd)

    