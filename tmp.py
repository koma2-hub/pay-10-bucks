import os 
import numpy as np
import torch
import fpsample
from utils.data_utils import load_ply , knn

data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/robot_record_icn/raw"
file_names = os.listdir(data_path)
print(len(file_names))

for i,file_name in enumerate(file_names):
    print(i, file_name)
    file_path = os.path.join(data_path, file_name)
    pcd = load_ply(file_path)
    indices = knn(pcd, k=32)

    rng = np.random.default_rng()
    sample_point = rng.integers(pcd.shape[0])
    neighbor_indice = indices[sample_point]
    pcd_intensity = pcd[sample_point, -1]
    pcd_intensity_noise = pcd_intensity +  np.random.normal(0, 0.1, 32)
    print(pcd_intensity_noise)
    pcd_intensity_noise = np.maximum(0, pcd_intensity_noise)
    print(pcd_intensity_noise)