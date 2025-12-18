#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d
from tqdm import tqdm

# 既存のモジュールからインポート
from model import PRNet
from data import PRNetDataset
from util import transform_point_cloud

def compute_errors(R_pred, t_pred, R_gt, t_gt):
    """
    回転行列と並進ベクトルの誤差を計算する
    """
    # 回転誤差 (角度: 度)
    R_diff = torch.matmul(R_pred, R_gt.transpose(2, 1))
    trace = R_diff.diagonal(dim1=1, dim2=2).sum(1)
    # 数値計算誤差で -1〜1 をわずかに超えるのを防ぐ
    trace = torch.clamp(trace, -1.0, 3.0) 
    # trace = 1 + 2cos(theta) -> theta = acos((trace-1)/2)
    rot_error_rad = torch.acos((trace - 1) / 2)
    rot_error_deg = torch.rad2deg(rot_error_rad)

    # 並進誤差 (L2ノルム)
    trans_error = torch.norm(t_pred - t_gt, dim=1)

    return rot_error_deg, trans_error

def visualize_registration(src, tgt, src_transformed):
    """
    Open3Dを用いて点群を可視化する
    src: 赤
    tgt: 青
    src_transformed (予測): 緑
    """
    # テンソルをnumpyに変換 (バッチの先頭[0]のみ使用)
    # [C, N] -> [N, 3] (XYZのみ抽出)
    src_np = src[0, :3, :].transpose(1, 0).detach().cpu().numpy()
    tgt_np = tgt[0, :3, :].transpose(1, 0).detach().cpu().numpy()
    pred_np = src_transformed[0, :3, :].transpose(1, 0).detach().cpu().numpy()

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(src_np)
    pcd_src.paint_uniform_color([1, 0, 0]) # 赤 (変換前)

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(tgt_np)
    pcd_tgt.paint_uniform_color([0, 0, 1]) # 青 (正解ターゲット)

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_np)
    pcd_pred.paint_uniform_color([0, 1, 0]) # 緑 (予測結果)

    print("可視化中... (ウィンドウを閉じると次に進みます)")
    print("赤: Source(入力), 青: Target(目標), 緑: Pred(予測結果)")
    o3d.visualization.draw_geometries([pcd_src, pcd_tgt, pcd_pred], 
                                      window_name="Registration Result",
                                      width=800, height=600)

def test(args):
    # 1. モデルの準備
    print(f"Loading model from {args.model_path}...")
    net = PRNet(args).cuda()
    
    # 重みのロード (DataParallelで保存されている場合の対応)
    if args.model_path != '':
        state_dict = torch.load(args.model_path)
        # キーに 'module.' がついている場合に取り除く処理
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        # PRNetは内部にacpnetを持つ構造なので、それにロードする
        net.acpnet.load_state_dict(new_state_dict)
    else:
        raise ValueError("モデルパス (--model_path) を指定してください")

    net.eval()

    # 2. データセットの準備
    test_dataset = PRNetDataset(args.data_path, intensity=args.use_intensity)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, # 可視化のため1ずつ処理したい場合は1にする
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Test data size: {len(test_dataset)}")

    rot_errors = []
    trans_errors = []

    # 3. 推論ループ
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # データの展開 (PRNetDatasetの戻り値に合わせて調整)
            # src, tgt, R_st, t_st, R_ts, t_ts, euler_st, euler_ts
            src, tgt, R_ab_gt, t_ab_gt, _, _, _, _ = [d.cuda() for d in data]

            # 推論実行
            # predictメソッドは (R_pred, t_pred) を返す
            R_pred, t_pred = net.predict(src, tgt, n_iters=args.n_iters)

            # 誤差計算
            r_err, t_err = compute_errors(R_pred, t_pred, R_ab_gt, t_ab_gt)
            
            rot_errors.append(r_err.cpu().numpy())
            trans_errors.append(t_err.cpu().numpy())

            # 可視化 (引数で指定された場合)
            if args.visualize:
                # 予測されたR, tを使ってsrcを変換
                src_transformed = transform_point_cloud(src, R_pred, t_pred)
                visualize_registration(src, tgt, src_transformed)
                
                # 全て表示すると大変なので、最初の数回だけ表示するなどの制御も可能
                # if i > 5: break 

    # 4. 結果の集計
    rot_errors = np.concatenate(rot_errors)
    trans_errors = np.concatenate(trans_errors)

    print("=========================================")
    print(f"Test Results on {len(test_dataset)} samples")
    print(f"Rotation Error (deg): Mean={np.mean(rot_errors):.4f}, Median={np.median(rot_errors):.4f}")
    print(f"Translation Error   : Mean={np.mean(trans_errors):.4f}, Median={np.median(trans_errors):.4f}")
    print("=========================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PRNet Testing')
    
    # 必須の引数
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.t7 file)')
    
    # モデル構造に関する引数 (学習時と同じにする必要があります)
    parser.add_argument('--exp_name', type=str, default='test_exp')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'])
    parser.add_argument('--attention', type=str, default='transformer', choices=['identity', 'transformer'])
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'])
    parser.add_argument('--n_emb_dims', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_ff_dims', type=int, default=1024)
    parser.add_argument('--n_keypoints', type=int, default=512)
    parser.add_argument('--n_subsampled_points', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--temp_factor', type=float, default=100)
    parser.add_argument('--cat_sampler', type=str, default='gumbel_softmax')
    
    # テスト実行に関する引数
    parser.add_argument('--n_iters', type=int, default=3, help='Number of iterations for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_intensity', action='store_true', help='Use intensity channel')
    parser.add_argument('--visualize', action='store_true', help='Visualize results with Open3D')
    
    # その他 (Loss計算などで使われるがテストには影響しないダミー)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--cycle_consistency_loss', type=float, default=0.1)
    parser.add_argument('--feature_alignment_loss', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='prnet')

    args = parser.parse_args()
    
    # 再現性のため
    torch.backends.cudnn.deterministic = True
    
    test(args)