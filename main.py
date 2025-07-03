# main.py
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader # 標準DataLoader
from torch_geometric.data import Data, Batch # PyGのData, Batchクラス
from torch_geometric.loader import DataLoader as GeometricDataLoader # PyGのDataLoader

# モデルとデータセットのインポート
from model import DGCNNLocalFeatureExtractor, get_graph_feature, knn_coords, knn_features, get_graph_feature_generic
from dataset_with_fps import ContrastivePointCloudDataset, load_ply, farthest_point_sampling, sample_patch_from_point_cloud, random_transform


# --- ハイパーパラメータ設定 ---
NUM_POINTS_PER_PATCH = 256 # 各パッチの点数
PATCH_RADIUS = None # パッチサンプリングの半径 (調整してください)
BATCH_SIZE = 32 # コントラスティブ学習のバッチサイズ (大きいほど良い)
EMB_DIMS = 1024 # DGCNNの埋め込み次元
PROJECTION_DIM = 128 # プロジェクションヘッドの出力次元
K_NEIGHBORS = 20 # DGCNNのK (K-NNグラフ構築)

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
TEMPERATURE = 0.07 # InfoNCE Lossの温度パラメータ

MODEL_SAVE_PATH = "dgcnn_local_feature_extractor_contrastive.pth"
DATA_ROOT_DIR = './dataset/robot_record_dataset/' # PLYファイルがあるdata/raw/の親ディレクトリ


def train_contrastive(model, dataloader, optimizer, device, temperature):
    model.train()
    total_loss = 0
    for batch_idx, (patch_A, patch_B) in enumerate(dataloader):
        # patch_A, patch_B は (B, N, C) のテンソル
        patch_A = patch_A.to(device)
        patch_B = patch_B.to(device)

        optimizer.zero_grad()

        # モデルからプロジェクションされた特徴量を取得 (訓練モードなのでprojected_featuresが返る)
        features_A = model(patch_A) # (B, N, projection_dim)
        features_B = model(patch_B) # (B, N, projection_dim)

        # パッチ全体の平均特徴量を取得 (簡易的なグローバル特徴量として扱う)
        # コントラスティブ学習では、パッチ内の全点から特徴を統合したり、
        # あるいはキーポイント検出をしてその特徴を使うことが多いですが、
        # ここではシンプルに平均特徴量を使用。
        features_A_global = features_A.mean(dim=1) # (B, projection_dim)
        features_B_global = features_B.mean(dim=1) # (B, projection_dim)
        
        # L2正規化
        features_A_global = F.normalize(features_A_global, p=2, dim=1)
        features_B_global = F.normalize(features_B_global, p=2, dim=1)

        # --- InfoNCE Loss の計算 ---
        # Positives: B_i に対する A_i
        l_pos = torch.einsum('bd,bd->b', features_A_global, features_B_global).unsqueeze(-1) # (B, 1)

        # Negatives: B_j (j != i) に対する A_i
        # einsum('bd,kd->bk') は A_i と B_k の全ての組み合わせの内積を計算
        l_neg = torch.einsum('bd,kd->bk', features_A_global, features_B_global) # (B, B)
        
        # logits の構成: [ポジティブの内積, ネガティブの内積 (バッチ内の他のサンプル)]
        # B_i と B_i の内積 (l_pos) は l_neg の対角成分なので、l_neg から対角成分を取り除く
        # もしくは、logits にl_posを直接結合し、ターゲットラベルでポジティブ位置を指定する
        
        # SimCLRスタイルのlogits構築 (対角成分がポジティブ)
        # logits = torch.matmul(features_A_global, features_B_global.T) / temperature # (B, B)
        # labels = torch.arange(BATCH_SIZE, dtype=torch.long, device=device) # 対角成分が正解

        # より一般的なInfoNCE (Positivesが最初に結合される形式)
        logits = torch.cat([l_pos, l_neg.fill_diagonal_(float('-inf'))], dim=1) # fill_diagonal_で対角成分をinfにして無視
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device) # ポジティブは常に0番目のインデックス

        loss = F.cross_entropy(logits / temperature, labels)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def extract_features(model, dataloader, device):
    model.eval() # 推論モードに設定 (local_featuresが返る)
    all_features = []
    with torch.no_grad():
        for batch_idx, (patch_A, patch_B) in enumerate(dataloader): # このdataloaderはコントラスティブ学習用だが、特徴抽出には片方で十分
            # 訓練済みモデルにデータを渡す (local_featuresを取得)
            # patch_A のみを使い、形状を (B, N, C) から (B, C, N) へ
            patch_A = patch_A.to(device)
            
            # モデルのforwardは (B, N, emb_dims) を返す
            features = model(patch_A) 
            
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. モデルのインスタンス化 (学習用)
    model = DGCNNLocalFeatureExtractor(k=K_NEIGHBORS, emb_dims=EMB_DIMS, projection_dim=PROJECTION_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 2. データセットとデータローダーの準備 (コントラスティブ学習用)
    # data/raw ディレクトリにPLYファイルがあることを確認してください
    if not os.path.exists(os.path.join(DATA_ROOT_DIR, 'raw')):
        print(f"Error: Directory '{os.path.join(DATA_ROOT_DIR, 'raw')}' not found.")
        print("Please place your .ply files in data/raw/ and ensure the path is correct.")
        exit()

    contrastive_dataset = ContrastivePointCloudDataset(
        root_dir=DATA_ROOT_DIR, 
        num_points_per_patch=NUM_POINTS_PER_PATCH, 
        patch_radius=PATCH_RADIUS
    )
    # 標準のDataLoaderを使用 (torch_geometric.data.DataLoaderではない)
    # Datasetがtorch_geometric.data.Dataを返さないため
    contrastive_dataloader = TorchDataLoader(
        contrastive_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        drop_last=True # InfoNCE Lossの計算で対角要素を扱うため、ドロップラストを有効にする
    )

    print(f"Number of batches for training: {len(contrastive_dataloader)}")

    # 3. モデルの学習ループ
    print("\n--- Starting Contrastive Learning Training ---")
    best_loss = float('inf')
    start = time.time()
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_contrastive(model, contrastive_dataloader, optimizer, device, TEMPERATURE)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

        # モデルの保存 (最も良い損失のモデルを保存)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH} (Loss: {best_loss:.4f})")

    learning_time = time.time() - start
    print("\n--- Contrastive Learning Training Complete ---")
    print("Total Learning Time:", learning_time)

    # 4. 学習済みモデルのロードと特徴量抽出
    print("\n--- Loading Trained Model for Feature Extraction ---")
    # 特徴量抽出時はプロジェクションヘッドを通さない方の出力 (emb_dims次元) を利用
    feature_extractor_model = DGCNNLocalFeatureExtractor(k=K_NEIGHBORS, emb_dims=EMB_DIMS, projection_dim=PROJECTION_DIM)
    
    # state_dictをロードする前に、モデルがデバイスに移動されていることを確認
    feature_extractor_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    feature_extractor_model.to(device)
    print("Model loaded successfully.")

    print("\n--- Starting Feature Extraction ---")
    # 特徴量抽出用のDataLoaderは、学習時と同じく ContrastivePointCloudDataset を利用
    # ただし、バッチサイズは任意で調整可能
    feature_extraction_dataloader = TorchDataLoader(
        contrastive_dataset, # ここでは同じデータセットを使うが、必要なら別途定義
        batch_size=BATCH_SIZE, 
        shuffle=False, # 特徴抽出時はシャッフル不要
        num_workers=4
    )
    
    extracted_features = extract_features(feature_extractor_model, feature_extraction_dataloader, device)
    
    print(f"Total extracted features shape: {extracted_features.shape}")
    print("\nFeature extraction complete.")

    # 抽出された特徴量の利用例
    # extracted_features の形状は (Total_Patches, emb_dims)
    # これらをRANSAC/ICPの前の特徴量マッチングに利用できます。
    # 例: extracted_features[0] は最初のパッチの特徴量
    #     extracted_features[BATCH_SIZE] は次のパッチの特徴量 など