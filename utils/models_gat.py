from __future__ import annotations

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from utils.eval_helpers import ensure_X3D



class WindowGraphDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        return x[:, None], self.y[idx]


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        X = ensure_X3D(X)
        B, N, _ = X.shape
        Wh = self.W(X).view(B, N, self.heads, self.out_dim)
        e_src = torch.einsum('bnhf,hf->bnh', Wh, self.a_src)
        e_dst = torch.einsum('bjhf,hf->bjh', Wh, self.a_dst)
        e = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
        mask = (A > 0).unsqueeze(-1)
        e = e.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        H = torch.einsum('bijh,bjhf->bihf', alpha, Wh)
        H = F.elu(H)
        H = H.reshape(B, N, self.heads * self.out_dim)
        return H


class GATClassifier(nn.Module):
    def __init__(self, n_nodes, in_dim=1, hid=12, heads=2, out_dim=12, dropout=0.3):
        super().__init__()
        self.gat1 = DenseGATLayer(in_dim, hid, heads=heads, dropout=dropout)
        self.gat2 = DenseGATLayer(hid * heads, out_dim, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(out_dim, 1)

    def forward(self, X, A):
        H = self.gat1(X, A)
        H = self.dropout(H)
        H = self.gat2(H, A)
        G = H.mean(dim=1)
        return self.readout(G).squeeze(-1)
