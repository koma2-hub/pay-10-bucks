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
# ★ 'make_dataset' から DCPDataset をインポート
from make_dataset import DCPDataset

def visualize_pcd(pcd_list, window_name="PointCloud"):
    """
    Open3Dで点群リストを可視化する。
    入力は (C, L) または (B, C, L) のTorchテンソルを想定。
    """
    pcd_o3d_list = []
    
    device = torch.device("cpu")
    
    for pcd_tensor in pcd_list:
        
        # テンソルを (L, C) の numpy 配列に変換
        if pcd_tensor.dim() == 3:
            # (B, C, L) -> (L, C)
            # .detach() を呼び出して計算グラフから切り離す
            pointcloud_np = pcd_tensor.squeeze(0).cpu().detach().numpy().T
        else:
            # (C, L) -> (L, C)
            # .detach() を呼び出して計算グラフから切り離す
            pointcloud_np = pcd_tensor.cpu().detach().numpy().T
            
        # (L, 3) の XYZ 座標を取得
        pointcloud_xyz = pointcloud_np[:, :3]
        
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pointcloud_xyz)
        
        # 色をランダムに設定
        rgb = [random.uniform(0,1) for i in range(3)]
        pcd_obj.paint_uniform_color(rgb)
        pcd_o3d_list.append(pcd_obj)
        
    o3d.visualization.draw_geometries(pcd_o3d_list, window_name=window_name)

# --- 1. データセットの準備 ---
# ★ ユーザーの環境に合わせたデータセットパス
processed_path = "/mnt/c/Users/matsu/SICK/pay-10-bucks/myDCP/dataset/"

# ★ intensity=False を指定 (Trueでも動作するが、ここでは座標のみ確認)
dataset = DCPDataset(processed_path, intensity=False)

print(f"データセット準備完了。合計ペア数: {len(dataset)}")


# --- 2. データセットの正解トランスフォームを確認 ---
torch.set_printoptions(edgeitems=torch.inf)
num_to_check = 5
print(f"データセットからランダムに {num_to_check} ペアの正解変換を確認します...")

for i in range(num_to_check):
    random_data_idx = random.randint(0, len(dataset) - 1)
    data = dataset[random_data_idx]
    
    # src, tgt は (C, L) = (3, 1024) (intensity=False のため)
    # R_st は (3, 3)
    # t_st は (1, 3)
    src, tgt, R_st, t_st, R_ts, t_ts, _, _ = data
    
    print(f"\n--- チェック {i+1}/{num_to_check} (データ インデックス: {random_data_idx}) ---")
    print(f"src 形状: {src.shape}, tgt 形状: {tgt.shape}")
    print(f"R_st 形状: {R_st.shape}, t_st 形状: {t_st.shape}")

    # 1. 位置合わせ前の点群の表示 (src と tgt)
    # visualize_pcd は (C, L) 形式をそのまま受け取る
    print("表示 1/2: 位置合わせ前の点群 (src=ランダム色, tgt=ランダム色)")
    visualize_pcd([src, tgt], window_name=f"[{i+1}] Before Registration (src vs tgt)")
    
    # 2. src に正解の変換 (R_st, t_st) を適用
    # p' = R * p + t
    
    # ★ 修正点: src はすでに (3, 1024) なので、そのまま p として使う
    src_xyz = src[:3, :] # (intensity=False なので [:3,:] は不要だが、安全のため残す)
    
    # ★ 修正点: t_st (1, 3) を (3, 1) に転置 (ブロードキャストのため)
    t_st_vec = t_st.T  # (3, 1)

    # ★ 修正点: 正しい変換 R*p + t
    # (3, 3) @ (3, 1024) + (3, 1)
    transformed_src_xyz = torch.matmul(R_st, src_xyz) + t_st_vec
    
    # 3. 輝度値 (Intensity) を戻す (存在する場合)
    if src.shape[0] == 4:
        src_intensity = src[3:, :] # (1, L)
        transformed_src = torch.cat((transformed_src_xyz, src_intensity), dim=0)
    else:
        transformed_src = transformed_src_xyz

    # 4. 位置合わせ後の点群の表示 (transformed_src と tgt)
    # ★ transformed_src と tgt がぴったり重なれば、データセットは正しい ★
    print("表示 2/2: 位置合わせ後の点群 (transformed_src と tgt が重なるはず)")
    visualize_pcd([transformed_src, tgt], window_name=f"[{i+1}] After Registration (Transformed Src vs Tgt)")
    print("srcをtgtに位置合わせする点群", R_st)

print("\n--- チェック完了 ---")