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

data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/processed"

data_files = os.listdir(data_path)

ply = load_ply(os.path.join(data_path, data_files[5]))
ds_ply = downsample_pcd(pointcloud=ply, downsample_point=8192)
ds_ply[:, :3] = ds_ply[:, :3] / 100
trans_vec = np.asarray([0, 10, 0])
trans_ply = ds_ply.copy()
trans_ply[:, :3] = trans_ply[:, :3] + trans_vec

visualize_pcd([ds_ply, trans_ply])




#
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    distance, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return distance, idx

def difference(pcd, k):
    """
    pcd -> (N, C)
    k -> int
    """
    #pcd_tensor -> (B, C, N)
    pcd_tensor = torch.from_numpy(pcd.astype(np.float32)).clone().to("cuda")
    pcd_tensor = pcd_tensor.unsqueeze(0)
    pcd_tensor = pcd_tensor.transpose(2, 1)
    dist, idx = knn(pcd_tensor[:3, 0], k = k)
    #pcd_tensor -> (C, N) dist:(B, N, k) -> (N, k) idx:(B, N, k) -> (N, k)
    for i in idx:
        #中心の点
        point = i[0]
        #近傍点の輝度とその距離
        intensities = pcd_tensor[3][i]
        distances = dist[point]
        #0除算対策
        distances[0] = 1

        #中心点と近傍点の輝度値の差をそれぞれのキョリで割ったものの総和を計算
        alpha = torch.sum(torch.abs(intensities - intensities[0]) / torch.abs(distances)) * 100
        #変化の割合を輝度値と置き換える
        pcd_tensor[3][point] = alpha
    pcd_numpy = pcd_tensor.to('cpu').detach().numpy().copy()
    return pcd_numpy


    

