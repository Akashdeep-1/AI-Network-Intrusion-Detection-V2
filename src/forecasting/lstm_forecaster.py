"""
LSTM Forecaster — CPU-capable, genuinely predictive S(t)->S(t+k)
"""
from __future__ import annotations
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=96, num_layers=2, num_classes=9, horizon=5, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        # Predict k steps independently via horizon heads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, num_classes)) for _ in range(horizon)])
    def forward(self, x):  # [B, W, 36]
        _, (h_n, _) = self.lstm(x)  # h_n [layers, B, hidden]
        h = h_n[-1]  # [B, hidden] last layer
        h = self.dropout(h)
        logits = torch.stack([head(h) for head in self.heads], dim=1)  # [B, horizon, num_classes]
        return logits  # [B, K, C]

class GRUForecaster(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=96, num_layers=2, num_classes=9, horizon=5, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, num_classes)) for _ in range(horizon)])
    def forward(self, x):
        _, h_n = self.gru(x)
        h = h_n[-1]
        h = self.dropout(h)
        logits = torch.stack([head(h) for head in self.heads], dim=1)
        return logits
