import os 
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import fpsample
from utils.data_utils import load_ply , knn

ply = load_ply("/mnt/c/Users/matsu/SICK/pay-10-bucks/data/mylabs_icn/raw/icn1024_mylab001.ply")

indices = knn(ply, k=32)

np.set_printoptions(threshold=sys.maxsize)
print(ply[indices])