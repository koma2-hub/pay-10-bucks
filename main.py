import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch # DataとBatchクラスをインポート

# DGCNNFeatureExtractor モデルの定義 (以前の修正版を使用)
# get_graph_feature, knn_coords, knn_features, get_graph_feature_generic も同じファイルにあると仮定
from utils.model import DGCNNFeatureExtractor, get_graph_feature, knn_coords, knn_features, get_graph_feature_generic

# MultiPointCloudDataset の定義 (提供いただいたものを使用)
from utils.PCLDataset import MultiPointCloudDataset, load_ply, farthest_point_sampling


# ----------------------------------------------------------------------
# 以下は main.py の主要部分の修正例
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # データセットの準備 (例: データファイルが 'data/raw' ディレクトリにある場合)
    # dataset_root はご自身の環境に合わせて設定してください
    dataset_root = '/mnt/c/Users/matsu/SICK/pay-10-bucks/dataset/' 
    num_points = 1024 # 各点群の点数
    batch_size = 10 # バッチサイズ (1より大きい値でテスト推奨)

    dataset = MultiPointCloudDataset(root='/mnt/c/Users/matsu/SICK/pay-10-bucks/dataset/robot_record_dataset/', num_points=1024)
    # DataLoader の batch_size を適切に設定してください
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルのインスタンス化
    model = DGCNNFeatureExtractor(k=20, emb_dims=1024) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 特徴量抽出の実行 (例)
    model.eval() # 推論モードに設定
    all_features = []

    print(f"Using device: {device}")
    print(f"Number of batches: {len(dataloader)}")

    with torch.no_grad(): # 勾配計算を無効化
        for batch_idx, data_batch in enumerate(dataloader):
            print(f"\nProcessing batch {batch_idx+1}/{len(dataloader)}")
            print(f"Type of data_batch: {type(data_batch)}") # デバッグ用
            #print(data_batch) # デバッグ用: data_batch の内容を確認

            if isinstance(data_batch, Data):
                # DataLoaderのbatch_size=1の場合など、Dataオブジェクトが直接返される
                # (1, N, C) の形状にするためにunsqueeze(0)を追加
                x_input = data_batch.x.unsqueeze(0) # (1, num_points, 4)
                # マスクも手動で作成（全点が有効と仮定）
                mask = torch.ones(x_input.shape[0], x_input.shape[1], dtype=torch.bool, device=device)
                print(f"  Input treated as single Data object. Shape: {x_input.shape}")
            elif isinstance(data_batch, Batch):
                # DataLoaderがBatchオブジェクトを返す場合
                # x_dense: (B, N_padded, C)
                # mask: (B, N_padded)
                x_input, mask = data_batch.to_dense_batch()
                print(f"  Input treated as Batch object. Dense shape: {x_input.shape}")
            else:
                raise TypeError(f"Unexpected type for data_batch: {type(data_batch)}")

            # テンソルをデバイスに移動
            x_input = x_input.to(device)

            # モデルにデータを渡す
            features = model(x_input) 
            
            print(f"  Extracted features shape for this batch: {features.shape}")
            all_features.append(features.cpu())

    extracted_features = torch.cat(all_features, dim=0)
    print(f"\nTotal extracted features shape: {extracted_features.shape}")

    print("\nFeature extraction complete.")