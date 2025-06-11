import numpy as np
from utils.data_utils import load_ply

pcd = load_ply('/mnt/c/Users/matsu/SICK/pay-10-bucks/dataset/robot_record_dataset/raw/robot_record0.ply')
print(type(pcd))