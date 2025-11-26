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

#近傍店の探索を行う
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(1,0).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(1,0).contiguous()

    distance, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return distance, idx


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

def random_transform(pcd, translation_range = (-5, 5)):
    translation_vector = np.array([np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1])])
    
    pcd_translated = np.copy(pcd)
    pcd_translated[:,:3] = pcd_translated[:,:3] + translation_vector
    return pcd, pcd_translated, translation_vector

for i in range(10):
    trans_range = (-5, 5)
    trans_vec = np.array([np.random.uniform(trans_range[0], trans_range[1]),np.random.uniform(trans_range[0], trans_range[1]),np.random.uniform(trans_range[0], trans_range[1])])
    print(trans_vec)
