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
from utils import load_ply, downsample_pcd

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
print(data_files)


for file in data_files:
    file_path = os.path.join(data_path, file)
    ply = load_ply(file_path)
    ply_ds = downsample_pcd(pointcloud=ply, downsample_point=4096, intensity=True)
    print(file_path)
    visualize_pcd([ply_ds])

