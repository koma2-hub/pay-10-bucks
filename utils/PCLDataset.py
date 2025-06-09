import os
import sys
sys.path.append(os.pardir)
import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
import fpsample
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as DataLoader
from torch_geometric.nn.pool import fps  # FPSの関数
from utils.data_utils import load_ply ,farthest_point_sampling

from torch_geometric.nn import knn
from torch_geometric.nn import max_pool_x

def downsample_pcd(pointcloud, downsample_point):
    fps_indices = fpsample.fps_sampling(pointcloud, downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :3]
    return downsampled_pc

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_cluster import fps
import open3d as o3d

import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_cluster import fps # これは使っていないかもしれませんが、もし使っているなら必要です
import open3d as o3d # これは使っていないかもしれませんが、もし使っているなら必要です
import numpy as np # numpyをインポート

# load_ply 関数と farthest_point_sampling 関数は変更なし

class MultiPointCloudDataset(InMemoryDataset):
    def __init__(self, root, num_points=1024, transform=None, pre_transform=None):
        self.num_points = num_points
        super().__init__(root, transform, pre_transform)
        # weights_only=False の設定は、torch==2.1.0 以降では非推奨または削除されている可能性があります。
        # 現在のPyTorch GeometricのバージョンとPyTorchのバージョンに合わせて、
        # `torch.load` の警告やエラーが出ないか確認してください。
        # もしエラーが出る場合は、この部分を `self.data, self.slices = torch.load(self.processed_paths[0])` に変更してみてください。
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.ply')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # すでにデータが存在する前提
        pass

    def process(self):
        data_list = []

        for fname in self.raw_file_names:
            path = os.path.join(self.raw_dir, fname)

            points = load_ply(path)  # (N, 4) - ここではfloat64である可能性が高い

            # --- ここに修正を追加 ---
            # NumPy配列を明示的にfloat32型に変換
            if points is not None:
                points = points.astype(np.float32)
            else:
                print(f"Skipping {fname} due to load_ply error.")
                continue # エラーが発生したファイルはスキップ

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # points[:, :3] はすでにfloat32なので、torch.from_numpy は float32 Tensor を作成する
            points_tensor = torch.from_numpy(points[:, :3]).to(device)

            # intensity も同様にfloat32になっているはず
            intensity = points[:, -1]
            intensity = torch.from_numpy(intensity).unsqueeze(1).to(device)

            # x も float32 Tensor になる
            x = torch.cat([points_tensor, intensity], dim=1)  # (N, 4)

            # farthest_point_sampling はNumPy配列の座標 (float32) で実行
            idx = farthest_point_sampling(points[:, :3], k = self.num_points)
            idx = torch.from_numpy(idx)
            
            # x_sampled はfloat32のままCPUに転送される
            x_sampled = x[idx].cpu()  # DataオブジェクトにはCPUテンソルを渡す

            print(len(data_list))
            data = Data(x=x_sampled)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
'''
dataset = MultiPointCloudDataset(root='/mnt/c/Users/matsu/SICK/pay-10-bucks/dataset/robot_record_dataset/', num_points=1024)
dataloader = DataLoader(dataset, batch_size = 10, shuffle=False)

batch = next(iter(dataloader))
print(batch)
assign_index = knn(x=batch.x[:, :3], y = batch.x[:, :3], k = 20, batch_x = batch.batch, batch_y=batch.batch)
print(assign_index.shape)
'''

'''
p = batch.x[assign_index[1, :],-1].unsqueeze(1)
q = batch.x[assign_index[0, :], -1].unsqueeze(1)
print(p.shape, q.shape)

feature = torch.cat([p, q-p], dim = 1)
print(feature.shape)
print(feature)

class EdgeConv(nn.Module):
    def __init__(self):
        super(EdgeConv, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2)
        )
    def forard(self, batch):
        assign_index = knn(x=batch.x[:, :3], y = batch.x[:, :3], k = 20, batch_x = batch.batch, batch_y=batch.batch)
'''




