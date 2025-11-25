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

data_path = "/mnt/d/SICK/pay-10-bucks/data/mylabs/processed"

data_files = os.listdir(data_path)

ply = load_ply(os.path.join(data_path, data_files[0]))
ds_ply = downsample_pcd(pointcloud=ply, downsample_point=8192)
ds_ply = ds_ply/100
t_vec = np.array([2,2,10])
pcd_trans = np.copy(ds_ply)
pcd_trans[:, :3] = pcd_trans[:, :3] + t_vec

visualize_pcd([ds_ply, pcd_trans])