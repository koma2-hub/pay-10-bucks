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
def visualize_pcd(pcd1,pcd2):
    pointcloud1 = pcd1[:, :3]
    pointcloud2 = pcd2[:, :3]
    #open3dのPointCloudオブジェクトを生成
    pcd_o3d1 = o3d.geometry.PointCloud()
    pcd_o3d1.points = o3d.utility.Vector3dVector(pointcloud1)
    pcd_o3d1.paint_uniform_color([1, 0, 0])
        #open3dのPointCloudオブジェクトを生成
    pcd_o3d2 = o3d.geometry.PointCloud()
    pcd_o3d2.points = o3d.utility.Vector3dVector(pointcloud2)
    pcd_o3d2.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd_o3d1, pcd_o3d2],
                                      window_name="Point Cloud")
    
    
sample_point = 4096
k = 1024
overlap_num = 512

def DCPDataset(sample_point, k, overlap_num, data_path):
    datasets = []
    file_names = os.listdir(data_path)
    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        ds_pcd = downsample_pcd(pcd, sample_point)
        indices = knn(ds_pcd, k)

pcd = load_ply("/mnt/d/SICK/pay-10-bucks/data/mylabs/raw/mylab001.ply")
ds_pcd = downsample_pcd(pcd, sample_point)
knn_indices = knn(ds_pcd, k)

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
knn_indices = knn_indices.astype(np.int64)
"""
for idx in candidates:
    pcd_index = knn_indices[idx]
    pcd_index = pcd_index.astype(np.int64)
    pcd = ds_pcd[pcd_index]
    visualize_pcd(pcd, [0,1,0])
"""

idx0 = candidates[0]
idx1 = candidates[1]

idx2 = candidates[2]
idx3 = candidates[3]

pcd_index0 = knn_indices[idx0]
pcd_index1 = knn_indices[idx1]
pcd_index2 = knn_indices[idx2]
pcd_index3 = knn_indices[idx3]

pcd0 = ds_pcd[pcd_index0]
pcd1 = ds_pcd[pcd_index1]
pcd2 = ds_pcd[pcd_index2]
pcd3 = ds_pcd[pcd_index3]

print(pcd0.shape)
print(pcd1.shape)
print(pcd2.shape)
print(pcd3.shape)

visualize_pcd(pcd0, pcd1)
visualize_pcd(pcd0, pcd2)
visualize_pcd(pcd0, pcd3)
visualize_pcd(pcd1, pcd2)
visualize_pcd(pcd1, pcd3)
visualize_pcd(pcd2, pcd3)



    