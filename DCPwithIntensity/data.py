import torch
import torch.utils.data as data
import numpy as np
import os

class PreTransformedDataset(data.Dataset):
    """
    すでに $X$, $Y$, $T$ のペアで保存されているデータを読み込むデータローダー。
    $X$ と $Y$ は (4, num_points) の形状を想定 (XYZI)。
    $T$ は (4, 4) の変換行列を想定。
    """
    def __init__(self, root_dir, partition='train'):
        self.root_dir = os.path.join(root_dir, partition)
        
        # 'src' (X), 'tgt' (Y), 'transform' (T) フォルダからファイルリストを取得
        self.src_files = sorted([os.path.join(self.root_dir, 'src', f) 
                                 for f in os.listdir(os.path.join(self.root_dir, 'src')) 
                                 if f.endswith('.npy')])
        self.tgt_files = sorted([os.path.join(self.root_dir, 'tgt', f) 
                                 for f in os.listdir(os.path.join(self.root_dir, 'tgt')) 
                                 if f.endswith('.npy')])
        self.transform_files = sorted([os.path.join(self.root_dir, 'transform', f) 
                                       for f in os.listdir(os.path.join(self.root_dir, 'transform')) 
                                       if f.endswith('.npy')])
        
        # ファイル数が一致しているか簡易チェック
        assert len(self.src_files) == len(self.tgt_files) == len(self.transform_files), \
               "Error: Mismatch in number of source, target, or transform files."

    def __len__(self):
        # データセットの総数を返す
        return len(self.src_files)

    def __getitem__(self, index):
        # 1. データをファイルから読み込む
        #    X と Y は (C, N) = (4, num_points) の形状を想定
        src_cloud = np.load(self.src_files[index])
        tgt_cloud = np.load(self.tgt_files[index])
        
        #    T は (4, 4) の形状を想定
        transform_matrix_gt = np.load(self.transform_files[index])

        # 2. 変換行列 $T$ (4x4) を $R$ (3x3) と $t$ (3x1) に分割
        #    DCPの損失関数 [cite: 214] は R と t を別々に扱うため
        R_gt = transform_matrix_gt[:3, :3]
        t_gt = transform_matrix_gt[:3, 3].reshape(3, 1) # (3,) -> (3, 1) に整形

        # 3. PyTorchテンソルに変換
        src_cloud = torch.from_numpy(src_cloud).float()
        tgt_cloud = torch.from_numpy(tgt_cloud).float()
        R_gt = torch.from_numpy(R_gt).float()
        t_gt = torch.from_numpy(t_gt).float()

        # 4. 4つの値をタプルとして返す
        return src_cloud, tgt_cloud, R_gt, t_gt