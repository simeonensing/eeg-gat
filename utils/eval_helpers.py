from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler



from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

def write_fold_artifacts(prefix: str, fold: int, y_true, y_score, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # --- ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        outdir / f"roc_{prefix}_fold{fold}.csv", index=False
    )

    # --- PR ---
    rec, pre, _ = precision_recall_curve(y_true, y_score)
    pos_rate = float(np.mean(y_true == 1)) if y_true.size else np.nan
    pd.DataFrame({"recall": rec, "precision": pre, "pos_rate": pos_rate}).to_csv(
        outdir / f"pr_{prefix}_fold{fold}.csv", index=False
    )

    # --- Per-fold predictions (for confusion matrices) ---
    pd.DataFrame({"y_true": y_true, "prob": y_score, "fold": fold}).to_csv(
        outdir / f"predictions_{prefix}_fold{fold}.csv", index=False
    )

def finalize_after_cv(prefix: str, outdir: Path):
    """Build the aggregated files your plotters use."""
    # Concatenate per-fold predictions -> predictions_{prefix}.csv
    preds = []
    for p in sorted(outdir.glob(f"predictions_{prefix}_fold*.csv")):
        preds.append(pd.read_csv(p))
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(
            outdir / f"predictions_{prefix}.csv", index=False
        )

    # Build a macro PR curve if you want the “Average” trace in PR plots
    # (simple interpolation-based average)
    import numpy as np
    curves = []
    for p in sorted(outdir.glob(f"pr_{prefix}_fold*.csv")):
        df = pd.read_csv(p)
        if {"recall","precision"} <= set(df.columns):
            curves.append((df["recall"].to_numpy(float), df["precision"].to_numpy(float)))
    if curves:
        # Interpolate precision on a common recall grid, then average
        grid = np.linspace(0.0, 1.0, 200)
        precs = []
        for rec, pre in curves:
            # make precision a non-increasing envelope for consistency
            pre = np.maximum.accumulate(pre[::-1])[::-1]
            precs.append(np.interp(grid, rec, pre, left=1.0, right=pre[-1] if pre.size else np.nan))
        macro_pre = np.nanmean(np.vstack(precs), axis=0)
        pd.DataFrame({"recall": grid, "precision": macro_pre}).to_csv(
            outdir / f"pr_macro_{prefix}.csv", index=False
        )




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
