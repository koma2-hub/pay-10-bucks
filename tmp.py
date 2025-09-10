import os 
import numpy as np
import torch
import fpsample
from utils.data_utils import load_ply , knn

def save_ply(filename, pcd):
    """
    点群データを PLY ファイルとして保存する関数。
    出力フォーマットは以下の通り:
      property float32 x
      property float32 y
      property float32 z
      property uint8 r
      property uint8 g
      property uint8 b
      property float32 i
      
    数値の桁指定:
      - x, y, z: 小数点以下4桁まで
      - r, g, b: 整数 (常に 0)
      - i: 小数点以下2桁まで

    例:
      -1334.0197 -1060.7484 1785.6458 0 0 0 0.16

    入力:
      pcd: (N,3) または (N,4) の numpy 配列または torch.Tensor
           4列目が存在する場合は intensity として使用、なければ 0 とする。
    """
    # Noneチェック
    if pcd is None:
        print(f"保存対象が None のためスキップします: {filename}")
        return

    # torch.Tensor の場合は numpy 配列に変換
    if isinstance(pcd, torch.Tensor):
        pcd_np = pcd.cpu().numpy()
    else:
        pcd_np = np.asarray(pcd)

    # 入力の形状チェック (N,3) または (N,4)
    if pcd_np.ndim != 2 or pcd_np.shape[1] not in (3, 4):
        raise ValueError("pcd は (N,3) または (N,4) の形状である必要があります。")

    # 座標は float32 として取得
    xyz = pcd_np[:, :3].astype(np.float32)
    
    # intensity の取得: 4列目があればその値、なければ 0
    if pcd_np.shape[1] == 4:
        intensity = pcd_np[:, 3].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((pcd_np.shape[0], 1), dtype=np.float32)
    
    # r, g, b は uint8 の 0 として生成
    rgb = np.zeros((pcd_np.shape[0], 3), dtype=np.uint8)
    
    # x, y, z, r, g, b, i の順にデータを結合 (shape: (N,7))
    data = np.hstack((xyz, rgb, intensity))

    # PLY ヘッダーの作成
    header = f"""ply
format ascii 1.0
element vertex {data.shape[0]}
property float32 x
property float32 y
property float32 z
property uint8 r
property uint8 g
property uint8 b
property float32 i
end_header
"""
    # ファイルにヘッダーと各点のデータを書き出す
    with open(filename, "w") as f:
        f.write(header)
        for row in data:
            # 書式: x,y,z は小数点以下4桁、i は小数点以下2桁で出力
            f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])} {int(row[4])} {int(row[5])} {row[6]:.2f}\n")

file_path = '/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/lab_room004.ply'
pcd = load_ply('/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/lab_room003.ply')

save_ply(filename=file_path, pcd=pcd)



    
src = load_ply("/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/lab1.ply")
tgt = load_ply("/mnt/c/Users/matsu/SICK/pay-10-bucks/test_data/lab2.ply")

o3d_pcd_a.points = o3d.utility.Vector3dVector(src[:, :3])
o3d_pcd_b.points = o3d.utility.Vector3dVector(tgt[:, :3])

o3d_pcd_a.paint_uniform_color([0, 0.651, 0.929])
o3d_pcd_b.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_b],
                                    window_name="raw point cloud")


