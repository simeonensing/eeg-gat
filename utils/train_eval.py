from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.utils.data import DataLoader

from config import CFG
from utils.settings import (
    LABEL_SMOOTH_EPS,
)
from utils.eval_helpers import zscore_fit_apply, make_edge_batch, eval_loader_sklearn_metrics
from utils.models_gat import WindowGraphDataset, GATClassifier
from utils.scheduler import CosineWarmupScheduler


def run_one_split_with_tracking(
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        A: np.ndarray,
        lr: float, weight_decay: float,
        max_epochs: int, patience: int, batch_size: int,
        device: str,
        model_hparams: dict | None = None,
        seed: int | None = None
):
    """Enhanced version that tracks training history."""
    if model_hparams is None:
        model_hparams = dict(hid=12, heads=2, out=12, dropout=0.3)
    if seed is None: seed = CFG.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_nodes = X_train.shape[1]
    Xtr, Xva, scaler = zscore_fit_apply(X_train, X_val)

    train_ds = WindowGraphDataset(Xtr, y_train)
    val_ds = WindowGraphDataset(Xva, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = GATClassifier(n_nodes, in_dim=1, hid=model_hparams['hid'],
                          heads=model_hparams['heads'],
                          out_dim=model_hparams['out'],
                          dropout=model_hparams['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(opt, warmup_epochs=8, max_epochs=max_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val, best_state, no_improve = float('inf'), None, 0
    train_losses, val_losses = [], []

    for ep in range(1, max_epochs + 1):
        model.train()
        epoch_train_loss = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device).float()
            if LABEL_SMOOTH_EPS > 0:
                yb_smooth = (1.0 - LABEL_SMOOTH_EPS) * yb + LABEL_SMOOTH_EPS * 0.5
            else:
                yb_smooth = yb
            Ab = make_edge_batch(A, xb.shape[0], device)
            logits = model(xb, Ab)
            loss = criterion(logits, yb_smooth)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_train_loss.append(loss.item())

        train_losses.append(np.mean(epoch_train_loss))

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device).float()
                if LABEL_SMOOTH_EPS > 0:
                    yb_smooth = (1.0 - LABEL_SMOOTH_EPS) * yb + LABEL_SMOOTH_EPS * 0.5
                else:
                    yb_smooth = yb
                Ab = make_edge_batch(A, xb.shape[0], device)
                logits = model(xb, Ab)
                v_losses.append(criterion(logits, yb_smooth).item())
        va_loss = float(np.mean(v_losses)) if v_losses else float('nan')
        val_losses.append(va_loss)

        scheduler.step()
        if np.isfinite(va_loss) and va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience: break

    if best_state is not None: model.load_state_dict(best_state)

    tr_metrics = eval_loader_sklearn_metrics(model, train_loader, A, device)
    va_metrics = eval_loader_sklearn_metrics(model, val_loader, A, device)

    return dict(model=model, scaler=scaler, train=tr_metrics, val=va_metrics,
                train_losses=train_losses, val_losses=val_losses)

