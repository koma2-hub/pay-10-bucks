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
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
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
    def __init__(self, margin=2.0):
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
epochs = 30
lr = 0.0005
batch_size = 32
input_dim = 32      # ★実際のデータに合わせてください
embedding_dim = 32  # ★調整可能なハイパーパラメータ

# 1. データセット全体をロード
full_dataset = VectorkICNDataset('/mnt/c/Users/matsu/SICK/pay-10-bucks/kICN_Dataset/vector_dataset_32points_noise005_1520cr.npz')

# 2. データセットを訓練用とテスト用に分割 (例: 80% 訓練, 20% テスト)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 3. それぞれのデータローダーを作成
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

print("\n--- 訓練開始 ---")
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for intensity1, intensity2, labels_batch in train_dataloader: # ★訓練データローダーを使用
        intensity1, intensity2, labels_batch = intensity1.to(device), intensity2.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        output1, output2 = model(intensity1, intensity2)
        loss = criterion(output1, output2, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

print("--- 訓練完了 ---")

# 損失の可視化
plt.figure(figsize=(10, 5))
epoch_range = range(1, epochs + 1)
plt.plot(epoch_range, train_losses, marker='o', linestyle='-', label='Training Loss')
plt.title('vector_32points_noise005_1520cr')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epoch_range)
plt.grid(True)
plt.legend()
plt.savefig('/mnt/c/Users/matsu/SICK/pay-10-bucks/logs/vector_32points_noise005_1520cr_loss.png')
plt.show()

torch.save(model.state_dict(), '/mnt/c/Users/matsu/SICK/pay-10-bucks/models/vector_siamese_modelcr.pth')
print("モデルを '/mnt/c/Users/matsu/SICK/pay-10-bucks/models/vector_siamese_modelcr.pth' として保存しました。")
# ==============================================================================
# D. 評価フェーズ
# ==============================================================================
print("\n--- モデル評価開始 ---")
model.eval() # モデルを評価モードに設定

distances = []
labels = []

# テストデータローダーを使って距離とラベルを収集
with torch.no_grad():
    for intensity1, intensity2, label in test_dataloader: # ★テストデータローダーを使用
        intensity1, intensity2 = intensity1.to(device), intensity2.to(device)
        output1, output2 = model(intensity1, intensity2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        distances.extend(euclidean_distance.cpu().numpy())
        labels.extend(label.cpu().numpy())

distances = np.array(distances)
labels = np.array(labels)

positive_samples = np.sum(labels == 1)
negative_samples = np.sum(labels == 0)
print(f"\n--- テストデータのラベル分布 ---")
print(f"正例 (同じペア) の数: {positive_samples}")
print(f"負例 (違うペア) の数: {negative_samples}\n")

# 最適な閾値を探索
thresholds = np.arange(0, 4.0, 0.01) # 閾値の範囲と刻み幅
best_accuracy = 0
best_threshold = 0

print("\n--- 閾値ごとの正解率 ---")
print(" Threshold | Accuracy")
print("-----------|-----------")

for threshold in thresholds:
    # 閾値に基づいて予測 (距離 < 閾値なら1 (同じ), そうでなければ0 (違う))
    predictions = (distances < threshold).astype(int)
    
    # 正解率を計算
    accuracy = np.mean(predictions == labels)
    
    # 各閾値とそれに対応する正解率を表示
    print(f" {threshold:9.2f} | {accuracy * 100:6.2f}%")
    
    # 最高の正解率を更新していれば、記録する
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold


# 最終的な結果の出力
print("\n--- モデル評価最終結果 ---")
print(f"最適な閾値 (Best Threshold): {best_threshold:.2f}")
print(f"最高の正解率 (Best Accuracy on Test Set): {best_accuracy * 100:.2f}%")

# (評価最終結果の出力の後に追加)
import matplotlib.pyplot as plt

# --- ★距離の分布を可視化 ---
plt.figure(figsize=(10, 6))
# ラベルが1 (正例) のペアの距離をプロット
plt.hist(distances[labels == 1], bins=50, alpha=0.7, label='Positive Pairs (Same Object)')
# ラベルが0 (負例) のペアの距離をプロット
plt.hist(distances[labels == 0], bins=50, alpha=0.7, label='Negative Pairs (Different Objects)')

# 最適な閾値を線で表示
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.2f}')
plt.axvline(best_accuracy, label=f'Best Accuracy = {best_accuracy:.2f}')
plt.title('vector_32points_noise005_1520cr')
plt.xlabel('Euclidean Distance')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/c/Users/matsu/SICK/pay-10-bucks/logs/vector_32points_noise005_1520cr_distribution.png')
plt.show()