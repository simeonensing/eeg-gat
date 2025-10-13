from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, brier_score_loss


def _ensure_path(p: Path | str) -> Path:
    return Path(p).expanduser().resolve()


def _pr_unique_with_endpoint(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Precision–Recall with sklearn, then:
       (a) de-dup recall (keep first occurrence),
       (b) append terminal point (recall=1, precision=prevalence).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if y_true.size == 0 or y_prob.size == 0 or np.unique(y_true).size < 2:
        pos_rate = float(y_true.mean()) if y_true.size else float("nan")
        return np.array([0.0, 1.0]), np.array([1.0, pos_rate]), pos_rate

    pre, rec, _ = precision_recall_curve(y_true, y_prob)  # sklearn returns precision, recall, thresholds
    rec_u, idx = np.unique(rec, return_index=True)
    pre_u = pre[idx]
    pos_rate = float(np.mean(y_true == 1))
    if rec_u[-1] < 1.0:
        rec_u = np.r_[rec_u, 1.0]
        pre_u = np.r_[pre_u, pos_rate]
    return rec_u.astype(float), pre_u.astype(float), pos_rate


def write_fold_pr_csv(prefix: str, fold: int, y_true: np.ndarray, y_prob: np.ndarray, outdir: Path) -> None:
    """Per-fold PR CSV: recall, precision, pos_rate (exact post-processed PR used by plotting)."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rec_u, pre_u, pos_rate = _pr_unique_with_endpoint(y_true, y_prob)
    pd.DataFrame({"recall": rec_u, "precision": pre_u, "pos_rate": pos_rate}).to_csv(
        outdir / f"pr_{prefix}_fold{fold}.csv", index=False
    )


def build_macro_pr_csv(prefix: str, outdir: Path) -> None:
    """Interpolate each per-fold PR onto a common recall grid with left=1, right=prevalence.
       Save pr_macro_{prefix}.csv with (recall, precision).
    """
    outdir = Path(outdir)
    paths = sorted(outdir.glob(f"predictions_{prefix}_fold*.csv"))
    if not paths:
        return

    curves = []
    for p in paths:
        dfp = pd.read_csv(p)
        if {"y_true", "prob"} <= set(dfp.columns):
            y = dfp["y_true"].to_numpy(int)
            pr = dfp["prob"].to_numpy(float)
            rec_u, pre_u, pos_rate = _pr_unique_with_endpoint(y, pr)
            curves.append((rec_u, pre_u, pos_rate))
    if not curves:
        return

    grid = np.linspace(0.0, 1.0, 200)
    mats = []
    for rec_u, pre_u, pos_rate in curves:
        mats.append(np.interp(grid, rec_u, pre_u, left=1.0, right=pos_rate))
    macro_pre = np.mean(np.vstack(mats), axis=0)
    pd.DataFrame({"recall": grid, "precision": macro_pre}).to_csv(
        outdir / f"pr_macro_{prefix}.csv", index=False
    )


def write_fold_artifacts(
    prefix: str,
    fold: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outdir: Path,
    fpr: Optional[np.ndarray] = None,
    tpr: Optional[np.ndarray] = None
) -> None:
    """Write per-fold ROC/PR/predictions CSVs matching plotting_helpers expectations."""
    outdir = _ensure_path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    # Drop any NaN/Inf to avoid flat/invalid curves
    m = np.isfinite(y_true) & np.isfinite(y_prob)
    if m.size and not np.all(m):
        y_true, y_prob = y_true[m], y_prob[m]

    # ROC
    if fpr is None or tpr is None:
        if np.unique(y_true).size > 1 and y_prob.size:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
        else:
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    pd.DataFrame({"fpr": fpr.astype(float), "tpr": tpr.astype(float)}).to_csv(
        outdir / f"roc_{prefix}_fold{fold}.csv", index=False
    )

    # PR (exact reference-style processing)
    write_fold_pr_csv(prefix, fold, y_true, y_prob, outdir)

    # Predictions
    pd.DataFrame(
        {"y_true": y_true.astype(int), "prob": y_prob.astype(float), "fold": int(fold)}
    ).to_csv(outdir / f"predictions_{prefix}_fold{fold}.csv", index=False)


def _concat_predictions(prefix: str, outdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate predictions_*fold*.csv -> predictions_{prefix}.csv and probabilities_{prefix}.csv."""
    outdir = _ensure_path(outdir)
    paths = sorted(outdir.glob(f"predictions_{prefix}_fold*.csv"))
    if not paths:
        return np.array([]), np.array([])
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    df.to_csv(outdir / f"predictions_{prefix}.csv", index=False)
    # probabilities file used by histogram plot
    df[["y_true", "prob"]].to_csv(outdir / f"probabilities_{prefix}.csv", index=False)
    return df["y_true"].to_numpy(int), df["prob"].to_numpy(float)


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_mean_pred, frac_positives)."""
    if y_prob.size == 0:
        return np.array([]), np.array([])
    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(y_prob, bins, right=True)
    bin_means, frac_pos = [], []
    for b in range(1, n_bins + 1):
        mask = inds == b
        if not np.any(mask):
            continue
        bin_means.append(float(np.mean(y_prob[mask])))
        frac_pos.append(float(np.mean(y_true[mask] == 1)))
    return np.asarray(bin_means, float), np.asarray(frac_pos, float)


def _write_calibration(prefix: str, outdir: Path) -> None:
    """Create calibration_curve_{prefix}.csv and calibration_stats_{prefix}.csv."""
    y_true, y_prob = _concat_predictions(prefix, outdir)  # also ensures predictions/probabilities files
    if y_prob.size == 0:
        return
    # Curve
    x, y = _calibration_bins(y_true, y_prob, n_bins=10)
    pd.DataFrame({"bin_mean_pred": x, "frac_positives": y}).to_csv(
        _ensure_path(outdir) / f"calibration_curve_{prefix}.csv", index=False
    )
    # Stats (ECE, Brier)
    ece = 0.0
    if x.size:
        bins = np.linspace(0, 1, 11)
        inds = np.digitize(y_prob, bins, right=True)
        ece_accum, n = 0.0, y_prob.size
        for b in range(1, 11):
            m = inds == b
            if not np.any(m):
                continue
            conf = float(np.mean(y_prob[m]))
            accb = float(np.mean(y_true[m] == 1))
            ece_accum += (np.sum(m) / n) * abs(accb - conf)
        ece = float(ece_accum)
    brier = float(brier_score_loss(y_true, y_prob))
    pd.DataFrame([{"ece": ece, "brier": brier}]).to_csv(
        _ensure_path(outdir) / f"calibration_stats_{prefix}.csv", index=False
    )


def finalize_prefix(prefix: str, outdir: Path) -> None:
    """Build all aggregate artifacts for a prefix."""
    build_macro_pr_csv(prefix, outdir)
    _concat_predictions(prefix, outdir)  # also writes probabilities_{prefix}.csv
    _write_calibration(prefix, outdir)


def write_topomap_values_csv(importance_dict: dict[str, np.ndarray], ch_names: list[str], save_prefix: str, outdir: Path):
    """{save_prefix}_topomap_values.csv with columns: Method, Channel, Importance"""
    rows = []
    for method, importance in importance_dict.items():
        # importance shape: [folds x channels] or [N x channels]; take mean across samples
        avg_imp = np.mean(np.asarray(importance), axis=0)
        for ch, val in zip(ch_names, avg_imp):
            rows.append({"Method": method, "Channel": ch, "Importance": float(val)})
    df = pd.DataFrame(rows)
    path = Path(outdir) / f"{save_prefix}_topomap_values.csv"
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")


def write_topomap_consensus_csv(importance_dict: dict[str, np.ndarray], ch_names: list[str], save_prefix: str, outdir: Path):
    """{save_prefix}_consensus_topomap.csv with columns: Channel, ConsensusImportance"""
    all_norm = []
    for importance in importance_dict.values():
        avg_imp = np.mean(np.asarray(importance), axis=0)
        if np.max(avg_imp) > np.min(avg_imp):
            avg_imp = (avg_imp - np.min(avg_imp)) / (np.max(avg_imp) - np.min(avg_imp))
        else:
            avg_imp = np.zeros_like(avg_imp)
        all_norm.append(avg_imp)
    consensus = np.mean(np.stack(all_norm, axis=0), axis=0) if all_norm else np.zeros(len(ch_names))
    df = pd.DataFrame({"Channel": ch_names, "ConsensusImportance": consensus.astype(float)})
    path = Path(outdir) / f"{save_prefix}_consensus_topomap.csv"
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")
