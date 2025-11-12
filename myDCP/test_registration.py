from __future__ import print_function
import os
import gc
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
# from data import ModelNet40
import open3d as o3d
from model import DCP
from util import transform_point_cloud, npmat2euler
import numpy as np
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import random_split 


#点群を可視化する関数
def visualize_pcd(pcd_src, pcd_tgt):
    
    # ★ 修正: テンソルがGPUにある場合 .cpu() が必要
    device = torch.device("cpu")
    pcd_src_np = pcd_src.squeeze(0).cpu().detach().numpy().T
    pcd_src_np_xyz = pcd_src_np[:,:3]
    pcd_src_o3d = o3d.geometry.PointCloud()
    pcd_src_o3d.points = o3d.utility.Vector3dVector(pcd_src_np_xyz)
    pcd_src_o3d.paint_uniform_color([1,0,0])

    pcd_tgt_np = pcd_tgt.squeeze(0).cpu().detach().numpy().T
    pcd_tgt_np_xyz = pcd_tgt_np[:,:3]
    pcd_tgt_o3d = o3d.geometry.PointCloud()
    pcd_tgt_o3d.points = o3d.utility.Vector3dVector(pcd_tgt_np_xyz)
    pcd_tgt_o3d.paint_uniform_color([0,0,1])
        
    o3d.visualization.draw_geometries([pcd_src_o3d, pcd_tgt_o3d], window_name="Point Cloud Visualization")


class DCPDataset(Dataset):
    """
    前処理済みの .pt ファイルを読み込むデータセットクラス
    """
    def __init__(self, processed_dir, intensity=True, n_points=1024):
        self.processed_dir = processed_dir
        self.intensity = intensity
        self.n_points = n_points # n_points を受け取る (main.py から渡されるため)
        
        # processed_dir にある .pt ファイルのリストを作成
        self.file_paths = []
        for f in os.listdir(self.processed_dir):
            if f.endswith(".pt"):
                self.file_paths.append(os.path.join(self.processed_dir, f))
        
        self.file_paths.sort() # 順序を保証

    def __len__(self):
        # データセットの総数 (ファイルの数) を返す
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # weights_only=False は pickle 化されたNumpy配列を読み込むために必要
        data = torch.load(file_path, weights_only=False) 

        # 3. 辞書からテンソルを取得
        # (make_dcpDataset で (C, L) 形式で保存されているため、transpose 不要)
        src_pcd = torch.as_tensor(data['src_pcd']).float()
        tgt_pcd = torch.as_tensor(data['tgt_pcd']).float()
        
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_ts = torch.as_tensor(data['t_ts']).float().view(1, 3)
        R_ts = torch.as_tensor(data['R_ts']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(1, 3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        euler_ts = torch.as_tensor(data['euler_ts']).float()

        # 輝度値を除外する場合
        if not self.intensity:
            src_pcd = src_pcd[:3, :] # (C, L) 形式なので、:3 でスライス
            tgt_pcd = tgt_pcd[:3, :]
            
        return src_pcd, tgt_pcd, R_st, t_st, R_ts, t_ts, euler_st, euler_ts
    

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration Visualization')
    
    # --- 必要な引数を main.py からコピー ---
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment (default: exp)')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--cycle', type=bool, default=True, metavar='N',
                        help='Whether to use cycle consistency')
    
    # --- このスクリプト固有の引数 ---
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to preprocessed data directory (.pt files)')
    parser.add_argument('--use_intensity', action='store_true', default=True,
                        help='Use intensity channel (4 channels) instead of 3')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path (default: checkpoints/exp/models/model.best.t7)')
    parser.add_argument('--num_vis', type=int, default=5,
                        help='Number of pairs to visualize')

    args = parser.parse_args()

    # ★ 修正点: args.input_channels を定義
    args.input_channels = 4 if args.use_intensity else 3
    if(args.input_channels == 4):
        print("Using 4 input channels (with intensity)")
    else:
        print("Using 3 input channels (coordinates only)")

    print(f"Loading all data from: {args.data_path}")
    # ★ 修正点: 引数のタイポ修正 (useintensity -> use_intensity)
    dataset = DCPDataset(args.data_path, intensity=args.use_intensity)
    
    #モデルのロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DCP(args).to(device)
    
    if args.model_path == '':
        model_path = f'checkpoints/{args.exp_name}/models/model.best.t7'
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        print("まず main.py でモデルを学習させてください。")
        return

    print(f"Loading model from: {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=device)) # 'map_location' を追加
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
    else:
        print(f"Using device: {device}")

    #データセットからランダムにデータを抽出し,推測される変換行列から試しに位置合わせを行う.
    print(f"--- {args.num_vis} ペアの可視化を開始します ---")
    
    for i in range(args.num_vis):
        test_data_index = random.randint(0, len(dataset) - 1)
        print(f"\n[{i+1}/{args.num_vis}] visualizing pair {test_data_index}...")
        
        #test_data = dataset[i]
        test_data = dataset[test_data_index]
        net.eval()
        
        src, tgt, R_gt, t_gt, _, _, _, _ = test_data

        # ★ 修正点 1: バッチ次元 (B=1) を追加
        src = src.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)

        #R_gtとt_gtをGPUに送りバッチ次元を追加する
        R_gt = R_gt.unsqueeze(0).to(device)
        t_gt = t_gt.T.unsqueeze(0).to(device)
        
        # モデルで変換を予測
        rotation_ab_pred, translation_ab_pred, _, _ = net(src, tgt)

        # 1. 元の点群 (src と tgt) を表示
        print("表示 1/2: 元の点群 (src=ランダム色, tgt=ランダム色)")
        visualize_pcd(src, tgt)
        
        # ★ 修正点 2: 正しい変換ロジック (util.py と同じ)
        
        # (B, C, L) から (B, 3, L) の XYZ 座標をスライス
        src_xyz = src[:, :3, :] 
        
        # (B, 3, 3) と (B, 3, L) で matmul
        # (B, 3) の並進を (B, 3, 1) に unsqueeze して加算
        transformed_src_xyz = torch.matmul(rotation_ab_pred, src_xyz) + translation_ab_pred.unsqueeze(2)

        # 輝度値がある場合は、変換後のXYZに輝度値を結合
        if args.input_channels == 4:
            src_intensity = src[:, 3:, :] # (B, 1, L)
            transformed_src = torch.cat((transformed_src_xyz, src_intensity), dim=1)
        else:
            transformed_src = transformed_src_xyz
        #1.1　正しい位置合わせを行った場合の点群を表示
        transformed_src_true = torch.matmul(R_gt, src_xyz) + t_gt
        visualize_pcd(transformed_src_true, tgt)

        # 2. 位置合わせ後の点群 (transformed_src と tgt) を表示
        print("表示 2/2: 位置合わせ後の点群 (transformed_src=ランダム色, tgt=ランダム色)")
        visualize_pcd(transformed_src, tgt)


        print("回転行列の真値", R_gt)
        print("並進の真値",t_gt)

        print("回転行列の予測値", rotation_ab_pred)
        print("並進行列の予測値", translation_ab_pred)

if __name__ == '__main__':
    main()