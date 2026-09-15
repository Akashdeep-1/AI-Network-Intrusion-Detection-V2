"""
CICIDS LSTM/GRU forecaster — Dataset B S(t) R^21
Supports:
- Formulation A: state regression S(t+1..K) 21 dims -> MSE
- Formulation B: binary attack prob + high-level 9-class
Configurable lightweight CPU model.
"""
from __future__ import annotations
import torch, torch.nn as nn

class CICIDSForecaster(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, num_layers=2, horizon=5, dropout=0.2, mode="both"):
        """
        mode: 'state' (regression 21*K), 'bin' (attack prob K), 'high' (9*K), 'both' (state+bin)
        """
        super().__init__()
        self.horizon = horizon
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        # heads
        if mode in ("state","both"):
            # predict K*21 via separate MLPs per horizon (avoid accumulation)
            self.state_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
                for _ in range(horizon)
            ])
        if mode in ("bin","both"):
            self.bin_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))
                for _ in range(horizon)
            ])
        if mode == "high":
            self.high_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 9))
                for _ in range(horizon)
            ])

    def forward(self, x):  # [B,W,21]
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])  # [B, hidden]
        out = {}
        if hasattr(self, "state_heads"):
            s = torch.stack([hd(h) for hd in self.state_heads], dim=1)  # [B,K,21]
            out["state"] = s
        if hasattr(self, "bin_heads"):
            b = torch.stack([hd(h).squeeze(-1) for hd in self.bin_heads], dim=1)  # [B,K]
            out["bin_logit"] = b
        if hasattr(self, "high_heads"):
            hi = torch.stack([hd(h) for hd in self.high_heads], dim=1)  # [B,K,9]
            out["high_logit"] = hi
        return out

class GRUForecaster(CICIDSForecaster):
    def __init__(self, input_dim=21, hidden_dim=64, num_layers=2, horizon=5, dropout=0.2, mode="both"):
        nn.Module.__init__(self)
        self.horizon = horizon
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        if mode in ("state","both"):
            self.state_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim)) for _ in range(horizon)])
        if mode in ("bin","both"):
            self.bin_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1)) for _ in range(horizon)])
        if mode == "high":
            self.high_heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 9)) for _ in range(horizon)])
    def forward(self, x):
        _, h_n = self.gru(x)
        h = self.dropout(h_n[-1])
        out={}
        if hasattr(self, "state_heads"):
            out["state"]=torch.stack([hd(h) for hd in self.state_heads], dim=1)
        if hasattr(self, "bin_heads"):
            out["bin_logit"]=torch.stack([hd(h).squeeze(-1) for hd in self.bin_heads], dim=1)
        if hasattr(self, "high_heads"):
            out["high_logit"]=torch.stack([hd(h) for hd in self.high_heads], dim=1)
        return out
