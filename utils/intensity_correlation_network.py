import torch
import torch.nn as nn

class ICN(nn.Module):
    def __init__(self, input_size):
        super(ICN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size*2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, x1, x2):
        # ベクトルを連結
        x = torch.cat([x1, x2], dim=1)
        output = self.net(x)
        output = torch.sigmoid(output)
        return output
