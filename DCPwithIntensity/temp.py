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
from test import DCPDataset


# --- 1. データセットの準備 ---
processed_path = "/mnt/d/SICK/pay-10-bucks/data/mylabs/processed"

# ★ ディスクから読み込むDatasetを初期化 ★
train_dataset = DCPDataset(processed_path, intensity=True)
 
print(f"データセット準備完了。合計ペア数: {len(train_dataset)}")

# --- 2. DataLoader の作成 ---
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)
# --- 3. データの取り出し (NumPyでの利用) ---
# 最初の1バッチを取り出してみる
# iter(train_loader).next() と同じ
batch = next(iter(train_loader))

# データセットが返す8つの値が、バッチ化されて取り出される
# (batch_size, num_points, channels)
src_pcd_batch, tgt_pcd_batch, R_st_batch, t_st_batch, \
R_ts_batch, t_ts_batch, euler_st_batch, euler_ts_batch = batch
 
print(f"\n--- 最初のバッチ ---")
print(f"ソース点群 (Torch Tensor) の形状: {src_pcd_batch.shape}")
print(f"変換行列 (Torch Tensor) の形状: {R_st_batch.shape}")
print(f"並進 (Torch Tensor) の形状: {t_st_batch.shape}")

# ★ NumPyで使いたい場合 ★
# .numpy() を呼び出すだけです
src_pcd_numpy = src_pcd_batch.numpy()
print(f"\nNumPy配列に変換した形状: {src_pcd_numpy.shape}")