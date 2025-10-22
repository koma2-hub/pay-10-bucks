import sys
import os
import numpy as np
import torch 
from utils import load_ply, knn

data_dir = ""
data_list = os.listdir(data_dir)
k = 4096
overlapp = 512

for file in (data_list):
    data_path = os.path.join(data_dir, file)
    pcd = load_ply(data_path)
    
    indices = knn(pcd, k)


