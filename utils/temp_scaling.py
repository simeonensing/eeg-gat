from __future__ import annotations

import numpy as np
import torch
from torch import nn as nn



def temperature_scale_logits(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200):
    if logits.size == 0 or len(np.unique(y_true)) < 2:
        return 1.0
    T = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    y = torch.tensor(y_true.astype(np.float32))
    z = torch.tensor(logits.astype(np.float32))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / torch.clamp(T, min=1e-3), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T_star = float(torch.clamp(T, min=1e-3).detach().cpu().item())
    except Exception:
        T_star = 1.0
    return max(1e-3, T_star)
