import os 
import torch

from utils.data_utils import load_ply

pcd = load_ply("./data/raw/robot_record0.ply")
pcd = torch.from_numpy(pcd)
print(pcd.shape[0])
