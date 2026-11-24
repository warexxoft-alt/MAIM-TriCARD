"""Optional PyTorch implementation of M-AIM's cross-modal attention fusion (sklearn-compatible wrapper)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class _Net(nn.Module):
    def __init__(self, dims, d=32, heads=4, drop=0.2):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, d)) for k in dims])
        self.attn = nn.TransformerEncoderLayer(d, heads, 64, drop, batch_first=True)
        self.gate = nn.Linear(d * len(dims), len(dims))
        self.out = nn.Linear(d, 1)

    def forward(self, xs, return_gate=False):
        tok = torch.stack([e(x) for e, x in zip(self.enc, xs)], 1)      # (B, M, d)  modality tokens
        tok = self.attn(tok)                                            # cross-modal self-attention
        g = torch.softmax(self.gate(tok.flatten(1)), 1)                 # gated fusion weights
        z = (g.unsqueeze(-1) * tok).sum(1)
        o = self.out(z).squeeze(-1)
        return (o, g) if return_gate else o


class TorchMAIM(BaseEstimator, ClassifierMixin):
    def __init__(self, groups, epochs=60, lr=1e-3, d=32, seed=0):
        self.groups, self.epochs, self.lr, self.d, self.seed = groups, epochs, lr, d, seed

    def _prep(self, X, fit=False):
        out = []
        for g, c in self.groups.items():
            if fit:
                self.pp_[g] = (SimpleImputer(strategy="median").fit(X[c]),)
                Z = self.pp_[g][0].transform(X[c]); self.pp_[g] += (StandardScaler().fit(Z),)
            Z = self.pp_[g][1].transform(self.pp_[g][0].transform(X[c]))
            out.append(torch.tensor(Z, dtype=torch.float32))
        return out

    def fit(self, X, y):
        torch.manual_seed(self.seed); self.classes_ = np.array([0, 1]); self.pp_ = {}
        xs = self._prep(X, True); yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.net_ = _Net([x.shape[1] for x in xs], self.d)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - yt.mean()) / yt.mean()))
        n = len(yt)
        for _ in range(self.epochs):
            self.net_.train(); perm = torch.randperm(n)
            for i in range(0, n, 64):
                b = perm[i:i + 64]; opt.zero_grad()
                lossf(self.net_([x[b] for x in xs]), yt[b]).backward(); opt.step()
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(self._prep(X))).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)
