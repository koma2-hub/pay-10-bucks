import numpy as np
import time
import fpsample 
import open3d as o3d
from utils.data_utils import load_ply , farthest_point_sampling
import os,sys
sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import load_ply


theta = np.pi/6
degree = 0.52

sin30 = np.sin(theta)
print(sin30)
sin30 = np.sin(degree)
print(sin30)


