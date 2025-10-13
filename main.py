#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWT+GAT (graph) vs Classical DSP+ML with XAI analysis.

Enhanced with:
- Occlusion sensitivity analysis for both pipelines
- Gradient × Input attribution for GAT models
- Consensus visualizations across methods
- Topographic maps using MNE
- CSV export for LaTeX tables
- Validation loss learning curves
- Reproducibility metadata CSV
- Optuna trials CSVs (per-study & combined)

NEW:
- Writes per-fold ROC/PR/predictions CSVs for plotting helpers
- Creates aggregated PR macro, probabilities, and calibration CSVs
- Writes topomap CSVs (values + consensus) matching plotting expectations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")

# --- Global font settings (Times-like) ---
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",  # math matches Times
        "axes.unicode_minus": False,  # proper minus sign with some serif fonts
        # (optional, helps with vector exports)
        "pdf.fonttype": 42,  # TrueType in PDFs
        "ps.fonttype": 42,   # TrueType in PS
    }
)

# Sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

# Graph wavelets
try:
    from pygsp import filters as gsp_filters
    from pygsp import graphs
except Exception:
    graphs = None
    gsp_filters = None

# Optuna
import optuna
from optuna.exceptions import TrialPruned

# Internal modules
from config import CFG
from utils.blocking_splits import plan_inner, plan_outer
from utils.classical_cv import classical_nested_cv
from utils.gwt_features import build_gwt_feature_table
from utils.shared_preprocessing import align_channels, prep_raw
from utils.spectral_graph import (
    band_baseline_from_base,
    build_graph_info,
    full_power,
)
from utils.summary_helpers import export_results_to_csv, fmt_num, fmt_triplet
from utils.temp_scaling import temperature_scale_logits
from utils.train_eval import run_one_split_with_tracking
from utils.utils import (
    hard_purge_train_rows,
    set_all_seeds,
    valid_split,
    write_reproducibility_csv,
)
from utils.xai import (
    create_xai_summary_table,
    gradient_input_gat,
    occlusion_sensitivity_gat,
)
from utils.plotting_helpers import plot_all

# main.py
from utils.settings import (
    USE_TEMP_SCALING,
    AVG_TWO_SEEDS_INNER,
    OUTER_SEEDS,
    USE_AP_SELECTION,
)

# ========================== CONFIG ==========================
SAVE_DIR = Path(CFG.data.save_dir).expanduser().resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "figures").mkdir(parents=True, exist_ok=True)  # ensure plot dir

# ========================== NEW HELPERS (ARTIFACTS) ==========================

def _ensure_path(p: Path | str) -> Path:
    return Path(p).expanduser().resolve()

# ---------- PR helpers: match reference behavior ----------

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

# ---------- Core artifact writers ----------

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

# ---------- Topomap CSV writers (values + consensus) ----------

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

# ========================== GLOBAL TOGGLES (commented in settings) ==========================
# USE_TEMP_SCALING = True
# AVG_TWO_SEEDS_INNER = True
# OUTER_SEEDS = 3
# USE_AP_SELECTION = True

def main() -> None:
    from utils.settings import RANDOM_SEED
    set_all_seeds(RANDOM_SEED)

    print("[INFO] Loading and preprocessing dataset (SHARED for both pipelines)...")
    raw_base   = prep_raw(CFG.data.pre_active_path,  CFG.data.montage_name, CFG.data.keep_channels)
    raw_before = prep_raw(CFG.data.post_sham_path,   CFG.data.montage_name, CFG.data.keep_channels)
    raw_after  = prep_raw(CFG.data.post_active_path, CFG.data.montage_name, CFG.data.keep_channels)

    sf_common = min(raw_base.info["sfreq"], raw_before.info["sfreq"], raw_after.info["sfreq"])
    for r in (raw_base, raw_before, raw_after):
        if r.info["sfreq"] != sf_common:
            r.resample(sf_common, npad="auto")

    [raw_base, raw_before, raw_after], used_ch_names = align_channels(
        [raw_base, raw_before, raw_after], CFG.data.keep_channels
    )
    print(f"[INFO] CSD preprocessing complete. Channels: {used_ch_names}, sfreq: {sf_common}Hz")

    sfreq = float(sf_common)
    data_base_uV = (raw_base.get_data()   * 1e6).astype(np.float64, copy=False)
    data_bef_uV  = (raw_before.get_data() * 1e6).astype(np.float64, copy=False)
    data_aft_uV  = (raw_after.get_data()  * 1e6).astype(np.float64, copy=False)

    baseline_by_band = {
        bn: band_baseline_from_base(data_base_uV, sfreq, CFG.bands[bn], CFG.all_freqs, CFG.n_cycles)
        for bn in CFG.bands.keys()
    }
    power_bef = full_power(data_bef_uV, sfreq, CFG.all_freqs, CFG.n_cycles)
    power_aft = full_power(data_aft_uV, sfreq, CFG.all_freqs, CFG.n_cycles)

    G, meyer, s_vals, degree_centrality, info_mne = build_graph_info(
        raw_after, used_ch_names, CFG.data.montage_name, CFG.spectral.n_scales, CFG.spectral.s_max
    )

    feature_store: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    outer_spec_by_win: Dict[int, Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int, int]] = {}
    n_pairs_by_win: Dict[int, int] = {}
    usable_windows: List[int] = []

    for win_sec in CFG.cv.window_grid:
        try:
            X_ref, y_ref, A_ref, n_pairs_ref = build_gwt_feature_table(
                power_bef, power_aft, baseline_by_band, G, meyer, degree_centrality,
                sfreq, used_band=CFG.band_order[0], s_index=0, win_sec=win_sec,
            )
        except Exception as e:
            print(f"[WARN] win={win_sec}s failed GWT feature build ({e}); skipping.")
            continue

        block_size_candidates = sorted({
            CFG.cv.block_size_pairs_default,
            max(6, CFG.cv.block_size_pairs_default - 2),
            max(6, CFG.cv.block_size_pairs_default - 4),
            max(6, CFG.cv.block_size_pairs_default // 2),
            12, 10, 8,
        }, reverse=True)
        embargo_candidates = sorted({
            CFG.cv.embargo_blocks_outer_default,
            max(0, CFG.cv.embargo_blocks_outer_default - 1),
            max(0, CFG.cv.embargo_blocks_outer_default - 2),
            1, 0,
        }, reverse=True)

        folds, used_K, used_bs, used_E = plan_outer(
            y_rows=y_ref, n_pairs=n_pairs_ref,
            block_sizes_pairs=block_size_candidates,
            embargoes=embargo_candidates,
            K_target=CFG.cv.outer_folds_target,
        )
        if used_K < 2:
            print(f"[WARN] win={win_sec}s feasible outer folds={used_K}; skipping.")
            continue

        print(f"[INFO] win={win_sec}s -> SHARED outer_folds={used_K}, block_size_pairs={used_bs}, embargo={used_E}")
        outer_spec_by_win[win_sec] = (folds, used_K, used_bs, used_E)
        n_pairs_by_win[win_sec] = n_pairs_ref

        for band_name in CFG.band_order:
            for s_idx in range(len(s_vals)):
                X, y, A, _ = build_gwt_feature_table(
                    power_bef, power_aft, baseline_by_band, G, meyer, degree_centrality,
                    sfreq, used_band=band_name, s_index=s_idx, win_sec=win_sec,
                )
                feature_store[(band_name, s_idx, win_sec)] = (X, y, A, n_pairs_ref)
        usable_windows.append(win_sec)

    if not usable_windows:
        raise RuntimeError("No window size produced enough valid outer folds under constraints.")

    print("\n" + "=" * 80)
    print("GRAPH PIPELINE (GWT + GAT) WITH XAI")
    print("=" * 80)
    per_fold_preds_graph: List[Dict[str, np.ndarray]] = []
    per_fold_metrics_graph: List[Dict[str, float]] = []
    fold_histories_graph: List[Dict[str, List[float]]] = []
    xai_results_graph: List[Dict[str, np.ndarray]] = []
    best_params_graph: Dict[str, float | int | str] = {}

    optuna_best_rows: List[dict] = []
    optuna_all_trials_rows: List[pd.DataFrame] = []

    for win_sec in sorted(usable_windows):
        folds, used_K, used_bs, used_E = outer_spec_by_win[win_sec]
        print(f"\n=== [Graph] Window {win_sec}s: {used_K} outer folds (bs={used_bs}, embargo={used_E}) ===")
        any_key = (CFG.band_order[0], 0, win_sec)
        _, y_any, _, _ = feature_store[any_key]

        for ofold in range(used_K):
            tr_idx, va_idx = folds[ofold]
            tr_idx = hard_purge_train_rows(tr_idx, va_idx, purge_pairs=CFG.cv.purge_pairs)

            y_tr_full, y_va_full = y_any[tr_idx], y_any[va_idx]
            print(f"[Graph Outer {ofold + 1}/{used_K}] |train|={len(tr_idx)} |val|={len(va_idx)}")

            inner_folds = plan_inner(tr_idx, y_any, used_bs)
            if len(inner_folds) < 2:
                print("  [Graph] WARN: Not enough valid inner folds; skipping this outer fold.")
                continue

            def objective(trial: optuna.Trial) -> float:
                try:
                    band_name = trial.suggest_categorical("band", CFG.band_order)
                    s_idx     = trial.suggest_categorical("s_idx", list(range(len(s_vals))))
                    hid       = trial.suggest_categorical("hid", [8, 12, 16])
                    heads     = 2
                    out_dim   = trial.suggest_categorical("out", [8, 12])
                    dropout   = trial.suggest_float("dropout", 0.28, 0.38)

                    X_all, y_all, A, _ = feature_store[(band_name, s_idx, win_sec)]
                    scores: List[float] = []

                    for itr, iva in inner_folds:
                        Xtr, ytr = X_all[itr], y_all[itr]
                        Xva, yva = X_all[iva], y_all[iva]

                        if not valid_split(
                            ytr, yva,
                            min_train=CFG.cv.min_train_rows_inner,
                            min_val=CFG.cv.min_val_rows_inner,
                            min_pos=CFG.cv.min_pos_per_split_inner,
                            min_neg=CFG.cv.min_neg_per_split_inner,
                            ratio_lo=CFG.cv.ratio_lo_inner,
                        ):
                            continue

                        if AVG_TWO_SEEDS_INNER:
                            vals: List[float] = []
                            for sd in [CFG.cv.random_seed, CFG.cv.random_seed + 1]:
                                res = run_one_split_with_tracking(
                                    Xtr, ytr, Xva, yva, A,
                                    lr=CFG.train.lr, weight_decay=CFG.train.weight_decay,
                                    max_epochs=CFG.train.max_epochs, patience=CFG.train.patience,
                                    batch_size=CFG.train.batch_size,
                                    device=(CFG.train.resolved_device() if callable(CFG.train.resolved_device) else CFG.train.resolved_device),
                                    model_hparams=dict(hid=hid, heads=heads, out=out_dim, dropout=dropout),
                                    seed=sd,
                                )
                                p = 1.0 / (1.0 + np.exp(-res["val"]["logits"]))
                                if USE_AP_SELECTION and len(np.unique(yva)) > 1:
                                    vals.append(average_precision_score(yva, p))
                                else:
                                    eps = 1e-7
                                    bce = -np.mean(yva * np.log(p + eps) + (1 - yva) * np.log(1 - p + eps))
                                    vals.append(-bce)
                            if vals:
                                scores.append(-float(np.mean(vals)))
                        else:
                            res = run_one_split_with_tracking(
                                Xtr, ytr, Xva, yva, A,
                                lr=CFG.train.lr, weight_decay=CFG.train.weight_decay,
                                max_epochs=CFG.train.max_epochs, patience=CFG.train.patience,
                                batch_size=CFG.train.batch_size,
                                device=(CFG.train.resolved_device() if callable(CFG.train.resolved_device) else CFG.train.resolved_device),
                                model_hparams=dict(hid=hid, heads=heads, out=out_dim, dropout=dropout),
                            )
                            p = 1.0 / (1.0 + np.exp(-res["val"]["logits"]))
                            if USE_AP_SELECTION and len(np.unique(yva)) > 1:
                                scores.append(-average_precision_score(yva, p))
                            else:
                                eps = 1e-7
                                bce = -np.mean(yva * np.log(p + eps) + (1 - yva) * np.log(1 - p + eps))
                                scores.append(bce)

                    if not scores:
                        raise TrialPruned("No valid inner splits for this configuration.")
                    return float(np.mean(scores))
                except TrialPruned:
                    raise
                except Exception as e:  # noqa: BLE001
                    raise TrialPruned(str(e))

            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=CFG.cv.random_seed))
            study.optimize(objective, n_trials=20, show_progress_bar=False)

            # save all trials for this study (graph)
            try:
                df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs"))
            except TypeError:
                df_trials = study.trials_dataframe()
            df_trials.insert(0, "pipeline", "graph")
            df_trials.insert(1, "window_s", win_sec)
            df_trials.insert(2, "outer_fold", ofold + 1)
            trial_path = SAVE_DIR / f"optuna_graph_win{win_sec}_fold{ofold + 1}.csv"
            df_trials.to_csv(trial_path, index=False)
            print(f"[Saved] {trial_path}")

            best = study.best_trial
            best_row = {"pipeline": "graph", "window_s": win_sec, "outer_fold": ofold + 1,
                        "objective": float(best.value), **{f"param_{k}": v for k, v in best.params.items()}}
            optuna_best_rows.append(best_row)
            optuna_all_trials_rows.append(df_trials)

            b_band = best.params.get("band")
            b_sidx = best.params.get("s_idx")
            hid    = best.params["hid"]
            heads  = 2
            out_dim = best.params["out"]
            dropout = best.params["dropout"]

            if ofold == 0:
                best_params_graph = {"band": b_band, "scale": float(s_vals[b_sidx]),
                                     "hidden": hid, "heads": heads, "out_dim": out_dim, "dropout": dropout}

            X_best, y_best, A_best, _ = feature_store[(b_band, b_sidx, win_sec)]
            tr_idx, va_idx = folds[ofold]
            tr_idx = hard_purge_train_rows(tr_idx, va_idx, purge_pairs=CFG.cv.purge_pairs)
            Xtr_full, ytr_full = X_best[tr_idx], y_best[tr_idx]
            Xva_full, yva_full = X_best[va_idx], y_best[va_idx]

            logits_ens: List[np.ndarray] = []
            models_ens = []
            for s in range(OUTER_SEEDS):
                res = run_one_split_with_tracking(
                    Xtr_full, ytr_full, Xva_full, yva_full, A_best,
                    lr=CFG.train.lr, weight_decay=CFG.train.weight_decay,
                    max_epochs=CFG.train.max_epochs, patience=CFG.train.patience,
                    batch_size=CFG.train.batch_size,
                    device=(CFG.train.resolved_device() if callable(CFG.train.resolved_device) else CFG.train.resolved_device),
                    model_hparams=dict(hid=hid, heads=heads, out=out_dim, dropout=dropout),
                    seed=CFG.cv.random_seed + s,
                )
                logits_ens.append(res["val"]["logits"])
                models_ens.append(res["model"])
                if s == 0:
                    fold_histories_graph.append({"train_losses": res["train_losses"], "val_losses": res["val_losses"]})

            logits_val = np.mean(np.stack(logits_ens, axis=0), axis=0)

            if (USE_TEMP_SCALING and logits_val.size and len(np.unique(yva_full)) > 1):
                T_star = temperature_scale_logits(logits_val, yva_full)
                probs_cal = 1.0 / (1.0 + np.exp(-logits_val / T_star))
            else:
                probs_cal = 1.0 / (1.0 + np.exp(-logits_val))

            scaler = StandardScaler()
            Xva_scaled = scaler.fit_transform(Xva_full)
            model_xai = models_ens[0]
            importance_occ  = occlusion_sensitivity_gat(model_xai, Xva_scaled, A_best, CFG.train.device)
            importance_grad = gradient_input_gat(model_xai, Xva_scaled, A_best, CFG.train.device)
            xai_results_graph.append({"fold": ofold, "occlusion": importance_occ, "gradient_input": importance_grad})

            yv = yva_full
            pv = probs_cal
            yhat = (pv >= 0.5).astype(int)
            acc = accuracy_score(yv, yhat)
            bacc = balanced_accuracy_score(yv, yhat)
            f1   = f1_score(yv, yhat)
            prec_s = precision_score(yv, yhat, zero_division=0)
            rec_s  = recall_score(yv, yhat, zero_division=0)
            aucv = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")
            apv  = average_precision_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")

            if len(np.unique(yv)) > 1 and len(pv) > 0:
                fpr, tpr, _ = roc_curve(yv, pv)
            else:
                fpr, tpr = np.array([0, 1]), np.array([0, 1])

            # ---- per-fold artifacts for graph ----
            write_fold_artifacts(prefix="gwt_gat", fold=ofold + 1, y_true=yv, y_prob=pv,
                                 outdir=SAVE_DIR, fpr=fpr, tpr=tpr)

            per_fold_preds_graph.append(dict(ytrue=yv, probs=pv))
            per_fold_metrics_graph.append(dict(acc=acc, bacc=bacc, f1=f1, prec=prec_s, rec=rec_s, auc=aucv, ap=apv))

            print(f"  -> [Graph] Best({b_band}, s={float(s_vals[b_sidx]):.5f}) | Val Acc={acc:.3f} AUC={aucv:.3f} AP={apv:.3f}")

    # ---- Graph topomap CSVs ----
    if xai_results_graph:
        all_occlusion_g = np.concatenate([r["occlusion"] for r in xai_results_graph], axis=0)
        all_grad_g      = np.concatenate([r["gradient_input"] for r in xai_results_graph], axis=0)
        importance_dict_g = {"Occlusion": all_occlusion_g, "Gradient_x_Input": all_grad_g}
        create_xai_summary_table(importance_dict_g, used_ch_names, "gwt_gat", SAVE_DIR)
        write_topomap_values_csv(importance_dict_g, used_ch_names, "gwt_gat", SAVE_DIR)
        write_topomap_consensus_csv(importance_dict_g, used_ch_names, "gwt_gat", SAVE_DIR)

    print("\n" + "=" * 80)
    print("CLASSICAL PIPELINE (DSP + ML) WITH XAI")
    print("=" * 80)
    (
        per_fold_preds_classical,
        per_fold_metrics_classical,
        fold_histories_classical,
    ) = classical_nested_cv(
        raw_before=raw_before,
        raw_after=raw_after,
        raw_base=raw_base,
        outer_spec_by_win=outer_spec_by_win,
        n_pairs_by_win=n_pairs_by_win,
        info_mne=info_mne,
        ch_names=used_ch_names,
        optuna_best_rows=optuna_best_rows,
        optuna_all_trials_rows=optuna_all_trials_rows,
    )

    # Hook for classical XAI (if available later)
    xai_results_classical: List[Dict[str, np.ndarray]] = []

    # ---- per-fold artifacts for classical ----
    for i, pf in enumerate(per_fold_preds_classical, 1):
        yv = pf.get("ytrue")
        pv = pf.get("probs")
        if yv is None or pv is None:
            continue
        if len(np.unique(yv)) > 1 and len(pv) > 0:
            fpr, tpr, _ = roc_curve(yv, pv)
        else:
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
        write_fold_artifacts(prefix="classical", fold=i, y_true=yv, y_prob=pv, outdir=SAVE_DIR, fpr=fpr, tpr=tpr)

    # If you later produce classical XAI arrays, populate xai_results_classical and these will write CSVs:
    if xai_results_classical:
        all_occ_c = np.concatenate([r["occlusion"] for r in xai_results_classical], axis=0)
        importance_dict_c = {"Occlusion": all_occ_c}
        write_topomap_values_csv(importance_dict_c, used_ch_names, "classical", SAVE_DIR)
        write_topomap_consensus_csv(importance_dict_c, used_ch_names, "classical", SAVE_DIR)

    # ---- Summaries ----
    best_params_class: Dict[str, float | int | str] = {}
    cls_only = [r for r in optuna_best_rows if r.get("pipeline") == "classical"]
    if cls_only:
        best_row = min(cls_only, key=lambda r: r["objective"])
        best_params_class = {k.replace("param_", ""): v for k, v in best_row.items() if k.startswith("param_")}
    if not best_params_class:
        best_params_class = {"model": "LogReg/SVC", "note": "See optuna_best_trials.csv"}

    def summarize(metrics_list: List[dict]) -> Dict[str, Tuple[float, float, int]]:
        if not metrics_list:
            return {}
        summary: Dict[str, Tuple[float, float, int]] = {}
        for k in ["acc", "bacc", "f1", "prec", "rec", "auc", "ap"]:
            vals = [m[k] for m in metrics_list if np.isfinite(m[k])]
            if vals:
                summary[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))
        return summary

    sum_graph = summarize(per_fold_metrics_graph)
    sum_class = summarize(per_fold_metrics_classical)

    # Macro PR (compute from saved predictions using same routine the plotter will read)
    build_macro_pr_csv("gwt_gat", SAVE_DIR)
    build_macro_pr_csv("classical", SAVE_DIR)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Outer-fold validation)")
    print("=" * 80)
    print(f"{'Metric':<12} {'Graph (GWT+GAT)':<28} {'Classical (DSP+ML)':<28}")
    for k, label in [
        ("acc",  "Accuracy"),
        ("bacc", "BalancedAcc"),
        ("f1",   "F1"),
        ("prec", "Precision"),
        ("rec",  "Recall"),
        ("auc",  "ROC AUC"),
        ("ap",   "AP"),
    ]:
        print(f"{label:<12} {fmt_triplet(sum_graph.get(k)):<28} {fmt_triplet(sum_class.get(k)):<28}")

    # == Reproducibility CSV ==
    write_reproducibility_csv(CFG, used_ch_names, sfreq, outer_spec_by_win, n_pairs_by_win)

    # == Optuna combined CSVs (best + all) ==
    if optuna_best_rows:
        df_best = pd.DataFrame(optuna_best_rows)
        path_best = SAVE_DIR / "optuna_best_trials.csv"
        df_best.to_csv(path_best, index=False)
        print(f"[Saved] {path_best}")

    if optuna_all_trials_rows:
        try:
            df_all = pd.concat(optuna_all_trials_rows, ignore_index=True)
            path_all = SAVE_DIR / "optuna_all_trials.csv"
            df_all.to_csv(path_all, index=False)
            print(f"[Saved] {path_all}")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Could not concatenate all trials: {e}")

    export_results_to_csv(
        sum_graph, sum_class, float("nan"), float("nan"),  # macro areas omitted to avoid double definitions
        best_params_graph, best_params_class,
    )

    # ---- Build aggregated artifacts for plotting helpers (calibration, probabilities) ----
    finalize_prefix("gwt_gat", SAVE_DIR)
    finalize_prefix("classical", SAVE_DIR)

    # Generate figures (reads the files we just wrote)
    plot_all()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {SAVE_DIR.absolute()}")


if __name__ == "__main__":
    main()
