import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# A. 準備フェーズ (モデル、Datasetクラス、損失関数の定義)
# ==============================================================================

class SubNetwork(nn.Module):
    def __init__(self, input_dim=32, embedding_dim=64):
        super(SubNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim)
        )
    def forward(self, x):
        return self.fc(x)


class SiameseNetwork(nn.Module):
    def __init__(self, sub_network):
        super(SiameseNetwork, self).__init__()
        self.sub_network = sub_network
    def forward(self, input1, input2):
        output1 = self.sub_network(input1)
        output2 = self.sub_network(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class VectorkICNDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.intensity1 = torch.from_numpy(data['intensity1'].astype(np.float32))
        self.intensity2 = torch.from_numpy(data['intensity2'].astype(np.float32))
        self.labels = torch.from_numpy(data['labels'].astype(np.float32))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.intensity1[idx], self.intensity2[idx], self.labels[idx]

# ==============================================================================
# B. データ準備フェーズ (データセットのロードと分割)
# ==============================================================================

# ハイパーパラメータ
epochs = 40
lr = 0.0005
batch_size = 32
input_dim = 32      # ★実際のデータに合わせてください
embedding_dim = 32  # ★調整可能なハイパーパラメータ

# 1. データセット全体をロード
full_dataset = VectorkICNDataset('/mnt/c/Users/matsu/SICK/pay-10-bucks/kICN_Dataset/vector_dataset_32points_noise005_2064.npz')

# 2. データセットを訓練用とテスト用に分割 (例: 80% 訓練, 20% テスト)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 3. それぞれのデータローダーを作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"データセット総数: {len(full_dataset)}")
print(f"訓練データ数: {len(train_dataset)}")
print(f"テストデータ数: {len(test_dataset)}")

# ==============================================================================
# C. 訓練フェーズ
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# モデル、損失関数、オプティマイザの初期化
sub_net = SubNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
model = SiameseNetwork(sub_network=sub_net).to(device)
criterion = ContrastiveLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

# 損失を保存するためのリスト
train_losses = []