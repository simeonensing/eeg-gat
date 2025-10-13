from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler




def ensure_X3D(X: torch.Tensor) -> torch.Tensor:
    if X.dim() == 2:  return X.unsqueeze(-1)
    if X.dim() == 3:  return X
    raise ValueError(f"X must be 2D/3D, got {tuple(X.shape)}")


@torch.no_grad()
def eval_loader_sklearn_metrics(model, loader, A, device):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        Ab = make_edge_batch(A, xb.shape[0], device)
        logits = model(xb, Ab)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
    if not all_logits:
        return dict(loss=float('nan'), acc=float('nan'), auc=float('nan'),
                    bacc=float('nan'), f1=float('nan'), prec=float('nan'), rec=float('nan'),
                    ap=float('nan'), ytrue=np.array([]), probs=np.array([]), logits=np.array([]))
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    p = 1.0 / (1.0 + np.exp(-logits))
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(y, yhat)
    bacc = balanced_accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    prec = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)
    aucv = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    eps = 1e-7
    bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return dict(loss=bce, acc=acc, auc=aucv, bacc=bacc, f1=f1, prec=prec, rec=rec,
                ap=ap, ytrue=y, probs=p, logits=logits)


def zscore_fit_apply(X_train: np.ndarray, X_val: np.ndarray):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler


def make_edge_batch(A: np.ndarray, batch_size: int, device: str):
    A_t = torch.from_numpy(A.astype(np.float32)).to(device)
    return A_t.unsqueeze(0).repeat(batch_size, 1, 1)
