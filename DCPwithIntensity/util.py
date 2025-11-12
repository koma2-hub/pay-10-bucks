#utils.py


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation

import fpsample

# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rot_mat, translation):
    # point_cloud は (B, C, L) 形状 (C=3 または C=4)
    # rot_mat は (B, 3, 3)
    # translation は (B, 3)
    
    # 1. 最初の3チャンネル (XYZ) だけをスライス
    xyz = point_cloud[:, :3, :]  # 形状: (B, 3, L)
    
    # 2. 3D座標 (XYZ) にのみ回転と並進を適用
    transformed_xyz = torch.matmul(rot_mat, xyz) + translation.unsqueeze(2) # 形状: (B, 3, L)
    
    # 3. チャンネル数に応じて処理を分岐
    if point_cloud.size(1) == 3:
        # 入力が 3 チャンネルだった場合
        return transformed_xyz
    else:
        # 入力が 4 チャンネル (XYZI) だった場合
        # 4チャンネル目 (Intensity) を取得
        intensity = point_cloud[:, 3:, :] # 形状: (B, 1, L)
        
        # 4. 変換後のXYZ と、元のIntensity を連結(cat)して (B, 4, L) に戻す
        return torch.cat((transformed_xyz, intensity), dim=1)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def downsample_pcd(pointcloud, downsample_point, intensity=False) -> np.ndarray:
    if pointcloud.shape[0] < downsample_point:
        downsample_point = pointcloud.shape[0]
    fps_indices = fpsample.fps_sampling(pointcloud[:,:3], downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :4]
    return downsampled_pc

def load_ply(filename, intensity=True):
    #.plyファイルを読み込み　点群(x, y, z, intensity)のnumpy配列を返す
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            header_index = None
            for i, line in enumerate(lines):
                if 'end_header' in line:
                    header_index = i
                    break
            if header_index is None:
                raise ValueError("PLYファイルのヘッダが正しく読み込めませんでした。")
            
            # ヘッダ以降の行を読み込み
            points = np.array([list(map(float, l.split())) for l in lines[header_index+1:]])
            if points.shape[1] < 4:
                # xyz + intensity(もしくは他の属性)がなければエラー
                raise ValueError(f"期待される列数に満たないデータが検出されました: {points.shape[1]}列")
            
            # [x, y, z, intensity] の形に整形
            # intensity が最後の列にあると仮定 (points[:, -1])
            points = np.concatenate([points[:, :3], points[:, -1].reshape(-1, 1)], axis=1)
            if(intensity):
                return points
            else:
                return points[:, :3]
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        # 必要に応じて sys.exit(1) などで終了するか、Noneを返す
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました(load_ply): {e}")
        return None
    

def knn(x: np.ndarray, k: int):
    """
    一つの点群データ `x` の各点について、k近傍点のインデックスをNumPyで計算する。

    Args:
        x: 点群データ。形状は (N, C)。
           N: 点の数
           C: 特徴量次元
        k: 探す近傍点の数。

    Returns:
        np.ndarray: 各点のk近傍点のインデックス。形状は (N, k)。
    """
    num_points = x.shape[0]
    x = x[:,:3]

    x_norm_sq = np.sum(x**2, axis=1, keepdims=True)  # 形状: (N, 1)
    dot_product = np.matmul(x, x.T)  # 形状: (N, N)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T  # 形状: (N, N)
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

    return indices