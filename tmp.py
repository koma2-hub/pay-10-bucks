import os 
import numpy as np
import torch
import fpsample
from utils.data_utils import load_ply , knn

data_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/data/robot_record_icn/raw"
file_names = os.listdir(data_path)
print(len(file_names))

for i,file_path in enumerate(file_names):
    print(i, file_path)

