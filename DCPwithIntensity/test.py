import sys
import os
import numpy as np
import torch 
from utils import load_ply, downsample_pcd, knn

sample_point = 4096
k = 1024

pcd = load_ply("/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs/raw/mylab001.ply")
ds_pcd = downsample_pcd(pcd, sample_point)
print(ds_pcd.shape)
knn_indices = knn(ds_pcd, k)
print(knn_indices.shape)