from __future__ import annotations

import hashlib
import platform
import random
import sys

from datetime import datetime
from typing import List, Dict, Tuple

import matplotlib
import mne

import numpy as np
import optuna
import pandas as pd
import torch

from utils.settings import (
    USE_TEMP_SCALING,
    AVG_TWO_SEEDS_INNER,
    OUTER_SEEDS,
    USE_AP_SELECTION,
    LABEL_SMOOTH_EPS,
    EXPLAIN_CLASS,
    USE_LOGIT_DELTAS_CLASSICAL,
)


def write_reproducibility_csv(
    cfg: Config,
    used_ch_names: List[str],
    sfreq: float,
    outer_spec_by_win: Dict[int, Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int, int]],
    n_pairs_by_win: Dict[int, int],
):
    """
    Writes a single-row CSV with all salient details for exact re-runs.
    {SAVE_DIR}/reproducibility_metadata.csv
    """
    versions = _lib_versions()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # dataset
        "pre_active_path": cfg.pre_active_path,
        "pre_active_sha256": _sha256_file(cfg.pre_active_path),
        "post_sham_path": cfg.post_sham_path,
        "post_sham_sha256": _sha256_file(cfg.post_sham_path),
        "post_active_path": cfg.post_active_path,
        "post_active_sha256": _sha256_file(cfg.post_active_path),
        "montage_name": cfg.montage_name,
        "kept_channels": ";".join(used_ch_names),
        "sfreq_hz": sfreq,
        # spectral & GWT
        "bands": ";".join([f"{k}:[{v[0]}-{v[1]}]" for k, v in cfg.bands.items()]),
        "band_order": ";".join(cfg.band_order),
        "n_scales": cfg.n_scales,
        "s_max": cfg.s_max,
        "n_freqs": int(np.asarray(cfg.all_freqs).size),
        # windows & CV
        "window_grid_s": ";".join(map(str, cfg.window_grid)),
        "outer_folds_target": cfg.outer_folds_target,
        "inner_folds_target": cfg.inner_folds_target,
        "random_seed": cfg.random_seed,
        # training
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "patience": cfg.patience,
        "device": cfg.device,
        # split feasibility thresholds (outer/inner)
        "outer_block_size_pairs_default": cfg.block_size_pairs_default,
        "outer_embargo_blocks_default": cfg.embargo_blocks_outer_default,
        "outer_min_pos": cfg.min_pos_per_split,
        "outer_min_neg": cfg.min_neg_per_split,
        "outer_min_train": cfg.min_train_rows,
        "outer_min_val": cfg.min_val_rows,
        "outer_ratio_lo": cfg.ratio_lo,
        "inner_embargo_blocks": cfg.embargo_blocks_inner,
        "inner_min_pos": cfg.min_pos_per_split_inner,
        "inner_min_neg": cfg.min_neg_per_split_inner,
        "inner_min_train": cfg.min_train_rows_inner,
        "inner_min_val": cfg.min_val_rows_inner,
        "inner_ratio_lo": cfg.ratio_lo_inner,
        # purge
        "purge_pairs": cfg.purge_pairs,
        # toggles
        "USE_TEMP_SCALING": USE_TEMP_SCALING,
        "AVG_TWO_SEEDS_INNER": AVG_TWO_SEEDS_INNER,
        "OUTER_SEEDS": OUTER_SEEDS,
        "USE_AP_SELECTION": USE_AP_SELECTION,
        "LABEL_SMOOTH_EPS": LABEL_SMOOTH_EPS,
        "EXPLAIN_CLASS": EXPLAIN_CLASS,
        "USE_LOGIT_DELTAS_CLASSICAL": USE_LOGIT_DELTAS_CLASSICAL,
        # versions
        **{f"ver_{k}": v for k, v in versions.items()},
    }

    # Add per-window outer spec & pair counts in friendly columns
    # e.g., win_4s_outerK, win_4s_block_size, win_4s_embargo, win_4s_pairs
    for win_sec, (folds, used_K, used_bs, used_E) in sorted(outer_spec_by_win.items()):
        row[f"win_{win_sec}s_outerK"] = used_K
        row[f"win_{win_sec}s_block_size"] = used_bs
        row[f"win_{win_sec}s_embargo"] = used_E
        row[f"win_{win_sec}s_pairs"] = n_pairs_by_win.get(win_sec, np.nan)

    df = pd.DataFrame([row])
    path = SAVE_DIR / "reproducibility_metadata.csv"
    df.to_csv(path, index=False)
    print(f"[Saved] {path}")


def _lib_versions() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": getattr(np, "__version__", "N/A"),
        "pandas": getattr(pd, "__version__", "N/A"),
        "torch": getattr(torch, "__version__", "N/A"),
        "sklearn": getattr(__import__("sklearn"), "__version__", "N/A"),
        "mne": getattr(mne, "__version__", "N/A"),
        "matplotlib": getattr(matplotlib, "__version__", "N/A"),
        "optuna": getattr(optuna, "__version__", "N/A"),
        "pygsp": getattr(__import__("pygsp"), "__version__", "N/A") if gsp_filters is not None else "N/A",
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})"
    }


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def valid_split(
        y_tr: np.ndarray, y_va: np.ndarray,
        *, min_train: int, min_val: int, min_pos: int, min_neg: int, ratio_lo: float
) -> bool:
    if len(y_tr) < min_train or len(y_va) < min_val: return False
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2: return False
    n0_tr, n1_tr = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    n0_va, n1_va = int((y_va == 0).sum()), int((y_va == 1).sum())
    if min(n0_tr, n1_tr) < min_pos or min(n0_va, n1_va) < min_pos: return False
    if min(n0_tr, n1_tr) < min_neg or min(n0_va, n1_va) < min_neg: return False

    def ok_ratio(a, b):
        mn, mx = min(a, b), max(a, b)
        return (mn / (mx + 1e-9)) >= ratio_lo

    return ok_ratio(n0_tr, n1_tr) and ok_ratio(n0_va, n1_va)


def pair_ids_from_rows(row_idx: np.ndarray) -> np.ndarray:
    return (row_idx // 2).astype(int)


def hard_purge_train_rows(tr_idx: np.ndarray, va_idx: np.ndarray, purge_pairs: int) -> np.ndarray:
    if purge_pairs <= 0: return tr_idx
    trp = pair_ids_from_rows(tr_idx)
    vap = pair_ids_from_rows(va_idx)
    forbid = set(int(v) + d for v in np.unique(vap) for d in range(-purge_pairs, purge_pairs + 1))
    keep = np.array([pid not in forbid for pid in trp], dtype=bool)
    return tr_idx[keep]


def _sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "N/A"
