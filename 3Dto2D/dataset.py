import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from tqdm import tqdm
import argparse

from utils import extract_near_edge_points,downsample_pcd


def random_rotation(pcd, rotation_range=(0, np.pi/6)):
    # (この関数は変更ありません)
    #ランダムな回転行列の生成
    angle_x = np.random.uniform(*rotation_range)
    angle_y = np.random.uniform(*rotation_range)
    angle_z = np.random.uniform(*rotation_range)

    sinx = np.sin(angle_x)
    cosx = np.cos(angle_x)
    siny = np.sin(angle_y)
    cosy = np.cos(angle_y)
    sinz = np.sin(angle_z)
    cosz = np.cos(angle_z)

    #各軸の回転行列
    rotation_x = np.array([[1, 0,    0],
                           [0, cosx, -sinx],
                           [0, sinx, cosx]])
    rotation_y = np.array([[cosy, 0, siny],
                           [0,    1, 0],
                           [-siny, 0, cosy]])
    rotation_z = np.array([[cosz, -sinz, 0],
                           [sinz, cosz,  0],
                           [0,    0,     1]])
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    pcd_rotated = pcd.copy()
    #点群が輝度値を持つときは座標のみ変換する
    #点群が輝度値を持たないときはそのまま変換する
    euler = np.asarray([angle_z, angle_y, angle_x])
    rotation_st = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])

    if(pcd.shape[1] != 3):
        pcd_rotated[:,:3] = rotation_st.apply(pcd[:,:3])
        return pcd, pcd_rotated, rotation_matrix, euler
    else:
        pcd_rotated = pcd_rotated.apply(pcd)
        return pcd, pcd_rotated, rotation_matrix, euler


def random_transform(pcd, translation_range = (-5, 5)):
    translation_vector = np.array([np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1])])
    
    pcd_translated = np.copy(pcd)
    pcd_translated[:,:3] = pcd_translated[:,:3] + translation_vector
    return pcd, pcd_translated, translation_vector


def sample_knn_patches_with_overlap(points_full, 
                                        num_points_k=1024):
    """
    (この関数は変更ありません)
    """
    
    N, D = points_full.shape
    K = num_points_k


    # 1. 座標データと KDTree を構築 (これは1回だけでよい)
    coords_xyz = points_full[:, :3]
    tree = KDTree(coords_xyz)


    center_index = np.random.randint(0, N)
    center_point = coords_xyz[center_index]
    _, indices_pcd = tree.query(center_point, k = K)

    points_src_patch = points_full[indices_pcd, :]
    points_tgt_patch = points_full[indices_pcd, :]

    return points_src_patch, points_tgt_patch

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
    
def rigit_transform(pcd):
    # (この関数は random_rotation と random_transform を呼ぶだけなので変更不要)
    pcd ,pcd_rotated, rotation_matrix, euler_zyx = random_rotation(pcd)
    _, pcd_rotated_tranfromed , transform_vector = random_transform(pcd_rotated)
    rotation_src_to_tgt = rotation_matrix
    rotation_tgt_to_src = rotation_src_to_tgt.T
    transform_tgt_to_src = -rotation_tgt_to_src.dot(transform_vector)
    euler_src_to_tgt = euler_zyx
    euler_tgt_to_src = -euler_zyx[::-1]
    return pcd, pcd_rotated_tranfromed, rotation_src_to_tgt,transform_vector,\
        rotation_tgt_to_src, transform_tgt_to_src, euler_src_to_tgt, euler_tgt_to_src
    

def make_Dataset(sample_point, k, sample_num, data_path, output_dir, pixel_size, edge_threshold, dilation_iter=2,intensity=True):
    os.makedirs(output_dir, exist_ok=True)

    file_names = os.listdir(data_path)
    print(f"対象ファイル:{file_names}")

    pair_counter = 0
    total_pair_to_generate = len(file_names) * sample_num
    pbar = tqdm(total=total_pair_to_generate, desc="Generating data pairs")

    for file in file_names:
        file_path = os.path.join(data_path, file)
        egde_pcd = extract_near_edge_points(file_path, pixel_size=pixel_size, edge_threshold=edge_threshold, dilation_iter=2)

        if egde_pcd is None:
            pbar.updata(sample_num)
            print("edge detection failed")
            continue

        ds_pcd = downsample_pcd(egde_pcd, sample_point)
        ds_pcd[:, :3] = ds_pcd[:, :3] / 128
        
        for i in range(sample_num):
            src_pcd, tgt_pcd = sample_knn_patches_with_overlap(ds_pcd, num_points_k = k)
            _, transformed_tgt, R_st, translation_st, \
            R_ts, translation_ts, euler_st, euler_ts = rigit_transform(tgt_pcd)

            #permutation
            src_pcd = np.random.permutation(src_pcd)
            transformed_tgt = np.random.permutation(transformed_tgt)
            # 4. (N, D) -> (C, L) 形式 (D, N) に転置して保存
            #    (D=4, N=1024)
            src_pcd = src_pcd.T
            transformed_tgt = transformed_tgt.T
            
            data_dict = {
                'src_pcd': src_pcd.astype(np.float32),
                'tgt_pcd': transformed_tgt.astype(np.float32),
                'R_st': R_st.astype(np.float32),
                't_st': translation_st.astype(np.float32),
                'R_ts': R_ts.astype(np.float32),
                't_ts': translation_ts.astype(np.float32),
                'euler_st': euler_st.astype(np.float32),
                'euler_ts': euler_ts.astype(np.float32)
            }
            
            output_filename = os.path.join(output_dir, f"pair_{pair_counter:06d}.pt")
            
            torch.save(data_dict, output_filename)
            
            pair_counter += 1
            
            pbar.update(1)


        

def main():
    parser = argparse.ArgumentParser(description='make dataset')

    parser.add_argument('--data_path', default='/mnt/d/SICK/pay-10-bucks/data/mylabs/processed')
    parser.add_argument('--output_path', default='/mnt/d/SICK/pay-10-bucks/3Dto2D/dataset')
    parser.add_argument('--dilation_iter', type=int, default=2)
    parser.add_argument('--sample_point', type=int, default=8192)
    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--pixel_size', type=float)
    parser.add_argument('--threshold', type=float)

    args = parser.parse_args()

    data_path = args.data_path
    output_path = args.output_path
    sample_point = args.sample_point
    sample_num = args.sample_num
    pixel_size = args.pixel_size
    edge_threshold = args.threshold
    dilation_iter = args.dilation_iter

    p = int(pixel_size*10)
    th = int(edge_threshold*10)

    output_path = output_path +'/' + 'pixelsize' + str(p) + '_threshold' + str(th) + '_dialation'
    
    print(f"データセットを{output_path}に生成します...")
    make_Dataset(
        sample_point = sample_point,
        k = 1024, 
        sample_num=sample_num,
        data_path=data_path,
        output_dir=output_path, 
        pixel_size=pixel_size, 
        edge_threshold=edge_threshold, 
        dilation_iter=dilation_iter,
        intensity=True
    )
    print("データセット生成完了.")

    print("\n---データセット読み込みテスト---")
    processed_path = output_path

    try:
        train_dataset = DCPDataset(processed_path, intensity=True)
        print(f"データセット準備完了。合計ペア数: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0 # ★ main.py 以外で num_workers > 0 を使うとエラーになることがあるため 0 に変更
        )
        
        batch = next(iter(train_loader))

        src_pcd_batch, tgt_pcd_batch, R_st_batch, t_st_batch, \
        R_ts_batch, t_ts_batch, euler_st_batch, euler_ts_batch = batch
        
        print(f"\n--- 最初のバッチ ---")
        print(f"ソース点群 (Torch Tensor) の形状: {src_pcd_batch.shape}")
        print(f"ターゲット点群 (Torch Tensor) の形状: {tgt_pcd_batch.shape}")
        print(f"変換行列 (Torch Tensor) の形状: {R_st_batch.shape}")
        print(f"並進 (Torch Tensor) の形状: {t_st_batch.shape}")
        
        # ★ 正規化された並進ベクトルの値を確認
        print(f"並進ベクトルのサンプル値 (t_st):\n {t_st_batch[:2]}")

    except Exception as e:
        print(f"データセットのテスト中にエラーが発生しました: {e}")
        print("トレースバック:")
        import traceback
        traceback.print_exc()   

if __name__ == '__main__':
    main()
