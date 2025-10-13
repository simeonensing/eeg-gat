from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from mne.viz import plot_topomap
from utils.settings import (
    EXPLAIN_CLASS,
    USE_LOGIT_DELTAS_CLASSICAL,
)

from utils.eval_helpers import make_edge_batch


def xai_sign() -> float:
    """Return +1 if we explain 'active', -1 if we explain 'sham'."""
    return -1.0 if EXPLAIN_CLASS.lower() == "sham" else 1.0


def _to_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Safely convert probabilities to logits."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def occlusion_sensitivity_gat(model, X_val, A, device, occlusion_size: int = 1):
    """
    Channel-wise occlusion sensitivity for the GAT model.

    Returns an array shaped (n_samples, n_channels), where positive values indicate
    evidence for EXPLAIN_CLASS (per EXPLAIN_CLASS setting).
    """
    model.eval()
    n_samples, n_channels = X_val.shape[0], X_val.shape[1]

    # Baseline logits
    X_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    X_t = X_t.unsqueeze(-1) if X_t.dim() == 2 else X_t
    A_batch = make_edge_batch(A, n_samples, device)

    with torch.no_grad():
        baseline_logits = model(X_t, A_batch).detach().cpu().numpy()

    importance = np.zeros((n_samples, n_channels), dtype=np.float32)
    sgn = xai_sign()

    for ch in range(n_channels):
        X_occ = X_val.copy()
        X_occ[:, ch] = 0.0  # occlude this channel

        X_occ_t = torch.from_numpy(X_occ.astype(np.float32)).to(device)
        X_occ_t = X_occ_t.unsqueeze(-1) if X_occ_t.dim() == 2 else X_occ_t

        with torch.no_grad():
            occluded_logits = model(X_occ_t, A_batch).detach().cpu().numpy()

        # Importance = drop in logit when the channel is removed, signed to target class
        importance[:, ch] = sgn * (baseline_logits - occluded_logits)

    return importance


def gradient_input_gat(model, X_val, A, device):
    """
    Gradient × Input attribution for the GAT model.

    Returns an array shaped (n_samples, n_channels), where positive values indicate
    evidence for EXPLAIN_CLASS (per EXPLAIN_CLASS setting).
    """
    model.eval()
    n_samples, n_channels = X_val.shape[0], X_val.shape[1]

    X_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    X_t = X_t.unsqueeze(-1) if X_t.dim() == 2 else X_t
    X_t.requires_grad_(True)

    A_batch = make_edge_batch(A, n_samples, device)
    logits = model(X_t, A_batch)

    attributions = np.zeros((n_samples, n_channels), dtype=np.float32)
    sgn = xai_sign()

    for i in range(n_samples):
        model.zero_grad(set_to_none=True)
        if X_t.grad is not None:
            X_t.grad.zero_()

        # Backprop a unit gradient for logit of sample i
        logits[i].backward(retain_graph=True)

        if X_t.grad is not None:
            grad_i = X_t.grad[i].detach().cpu().numpy().squeeze()   # (n_channels,)
            inp_i = X_val[i]                                        # (n_channels,)
            attributions[i] = sgn * (grad_i * inp_i)

    return attributions


def occlusion_sensitivity_classical(model, X_val, n_channels: int, n_bands: int):
    """
    Channel-wise occlusion sensitivity for the classical model.

    Features are flattened as (n_channels * n_bands).
    Returns an array shaped (n_samples, n_channels), where positive values indicate
    evidence for EXPLAIN_CLASS (per EXPLAIN_CLASS setting).
    """
    n_samples = X_val.shape[0]
    sgn = xai_sign()

    # Baseline predictions
    baseline_probs = model.predict_proba(X_val)[:, 1]
    if USE_LOGIT_DELTAS_CLASSICAL:
        baseline = _to_logit(baseline_probs)
    else:
        baseline = baseline_probs

    importance = np.zeros((n_samples, n_channels), dtype=np.float32)
    feats_per_ch = n_bands

    for ch in range(n_channels):
        X_occ = X_val.copy()
        start, end = ch * feats_per_ch, (ch + 1) * feats_per_ch
        X_occ[:, start:end] = 0.0

        occluded_probs = model.predict_proba(X_occ)[:, 1]
        if USE_LOGIT_DELTAS_CLASSICAL:
            occluded = _to_logit(occluded_probs)
        else:
            occluded = occluded_probs

        # Importance = drop when channel removed, signed to target class
        importance[:, ch] = sgn * (baseline - occluded)

    return importance

def create_xai_summary_table(importance_dict, ch_names, save_prefix):
    """Create XAI summary table showing channel importance."""
    rows = []

    for method, importance in importance_dict.items():
        avg_importance = np.mean(importance, axis=0)
        std_importance = np.std(importance, axis=0)

        for ch_idx, ch_name in enumerate(ch_names):
            rows.append({
                'Method': method,
                'Channel': ch_name,
                'Mean_Importance': avg_importance[ch_idx],
                'Std_Importance': std_importance[ch_idx]
            })

    df = pd.DataFrame(rows)
    pivot_mean = df.pivot(index='Channel', columns='Method', values='Mean_Importance')

    save_path = SAVE_DIR / f"{save_prefix}_xai_table.csv"
    pivot_mean.to_csv(save_path)
    print(f"[Saved] {save_path}")

    return df
