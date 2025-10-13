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
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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
        "ps.fonttype": 42,  # TrueType in PS
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

# ...rest of your imports and code unchanged...


# ========================== GLOBAL TOGGLES ==========================
# USE_TEMP_SCALING = True
# AVG_TWO_SEEDS_INNER = True
# OUTER_SEEDS = 3
# USE_AP_SELECTION = True
# LABEL_SMOOTH_EPS = 0.05
# EXPLAIN_CLASS = "active"  # or "SHAM"
# USE_LOGIT_DELTAS_CLASSICAL = True

# ========================== CONFIG ==========================
SAVE_DIR = Path(CFG.data.save_dir)


def main() -> None:
    from utils.settings import RANDOM_SEED
    set_all_seeds(RANDOM_SEED)

    print(
        "[INFO] Loading and preprocessing dataset (SHARED for both pipelines)..."
    )
    raw_base = prep_raw(
        CFG.data.pre_active_path, CFG.data.montage_name, CFG.data.keep_channels
    )
    raw_before = prep_raw(
        CFG.data.post_sham_path, CFG.data.montage_name, CFG.data.keep_channels
    )
    raw_after = prep_raw(
        CFG.data.post_active_path, CFG.data.montage_name, CFG.data.keep_channels
    )

    sf_common = min(
        raw_base.info["sfreq"],
        raw_before.info["sfreq"],
        raw_after.info["sfreq"],
    )
    for r in (raw_base, raw_before, raw_after):
        if r.info["sfreq"] != sf_common:
            r.resample(sf_common, npad="auto")

    [raw_base, raw_before, raw_after], used_ch_names = align_channels(
        [raw_base, raw_before, raw_after], CFG.data.keep_channels
    )
    print(
        f"[INFO] CSD preprocessing complete. "
        f"Channels: {used_ch_names}, sfreq: {sf_common}Hz"
    )

    sfreq = float(sf_common)
    data_base_uV = (raw_base.get_data() * 1e6).astype(
        np.float64, copy=False
    )
    data_bef_uV = (raw_before.get_data() * 1e6).astype(
        np.float64, copy=False
    )
    data_aft_uV = (raw_after.get_data() * 1e6).astype(
        np.float64, copy=False
    )

    baseline_by_band = {
        bn: band_baseline_from_base(
            data_base_uV, sfreq, CFG.bands[bn], CFG.all_freqs, CFG.n_cycles
        )
        for bn in CFG.bands.keys()
    }
    power_bef = full_power(data_bef_uV, sfreq, CFG.all_freqs, CFG.n_cycles)
    power_aft = full_power(data_aft_uV, sfreq, CFG.all_freqs, CFG.n_cycles)

    G, meyer, s_vals, degree_centrality, info_mne = build_graph_info(
        raw_after, used_ch_names, CFG.data.montage_name, CFG.spectral.n_scales, CFG.spectral.s_max
    )

    feature_store: Dict[
        Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, int]
    ] = {}
    outer_spec_by_win: Dict[
        int, Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int, int]
    ] = {}
    n_pairs_by_win: Dict[int, int] = {}
    usable_windows: List[int] = []

    for win_sec in getattr(CFG.cv, "window_grid", [2, 3, 4]):
        try:
            X_ref, y_ref, A_ref, n_pairs_ref = build_gwt_feature_table(
                power_bef,
                power_aft,
                baseline_by_band,
                G,
                meyer,
                degree_centrality,
                sfreq,
                used_band=CFG.band_order[0],
                s_index=0,
                win_sec=win_sec,
            )
        except Exception as e:
            print(
                f"[WARN] win={win_sec}s failed GWT feature build ({e}); "
                f"skipping."
            )
            continue

        block_size_candidates = sorted(
            {
                CFG.cv.block_size_pairs_default,
                max(6, CFG.cv.block_size_pairs_default - 2),
                max(6, CFG.cv.block_size_pairs_default - 4),
                max(6, CFG.cv.block_size_pairs_default // 2),
                12,
                10,
                8,
            },
            reverse=True,
        )
        embargo_candidates = sorted(
            {
                CFG.cv.embargo_blocks_outer_default,
                max(0, CFG.cv.embargo_blocks_outer_default - 1),
                max(0, CFG.cv.embargo_blocks_outer_default - 2),
                1,
                0,
            },
            reverse=True,
        )

        folds, used_K, used_bs, used_E = plan_outer(
            y_rows=y_ref,
            n_pairs=n_pairs_ref,
            block_sizes_pairs=block_size_candidates,
            embargoes=embargo_candidates,
            K_target=CFG.cv.outer_folds_target,
        )
        if used_K < 2:
            print(
                f"[WARN] win={win_sec}s feasible outer folds={used_K}; "
                f"skipping."
            )
            continue

        print(
            f"[INFO] win={win_sec}s -> SHARED outer_folds={used_K}, "
            f"block_size_pairs={used_bs}, embargo={used_E}"
        )
        outer_spec_by_win[win_sec] = (folds, used_K, used_bs, used_E)
        n_pairs_by_win[win_sec] = n_pairs_ref

        for band_name in CFG.band_order:
            for s_idx in range(len(s_vals)):
                X, y, A, _ = build_gwt_feature_table(
                    power_bef,
                    power_aft,
                    baseline_by_band,
                    G,
                    meyer,
                    degree_centrality,
                    sfreq,
                    used_band=band_name,
                    s_index=s_idx,
                    win_sec=win_sec,
                )
                feature_store[(band_name, s_idx, win_sec)] = (
                    X,
                    y,
                    A,
                    n_pairs_ref,
                )
        usable_windows.append(win_sec)

    if not usable_windows:
        raise RuntimeError(
            "No window size produced enough valid outer folds "
            "under constraints."
        )

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
        print(
            f"\n=== [Graph] Window {win_sec}s: {used_K} outer folds "
            f"(bs={used_bs}, embargo={used_E}) ==="
        )
        any_key = (CFG.band_order[0], 0, win_sec)
        _, y_any, _, _ = feature_store[any_key]

        for ofold in range(used_K):
            tr_idx, va_idx = folds[ofold]
            tr_idx = hard_purge_train_rows(tr_idx, va_idx, purge_pairs=CFG.cv.purge_pairs)

            y_tr_full, y_va_full = y_any[tr_idx], y_any[va_idx]
            print(
                f"[Graph Outer {ofold + 1}/{used_K}] "
                f"|train|={len(tr_idx)} |val|={len(va_idx)}"
            )

            inner_folds = plan_inner(tr_idx, y_any, used_bs)
            if len(inner_folds) < 2:
                print(
                    "  [Graph] WARN: Not enough valid inner folds; "
                    "skipping this outer fold."
                )
                continue

            def objective(trial: optuna.Trial) -> float:
                try:
                    band_name = trial.suggest_categorical(
                        "band", CFG.band_order
                    )
                    s_idx = trial.suggest_categorical(
                        "s_idx", list(range(len(s_vals)))
                    )
                    hid = trial.suggest_categorical("hid", [8, 12, 16])
                    heads = 2
                    out_dim = trial.suggest_categorical("out", [8, 12])
                    dropout = trial.suggest_float("dropout", 0.28, 0.38)

                    X_all, y_all, A, _ = feature_store[
                        (band_name, s_idx, win_sec)
                    ]
                    scores: List[float] = []

                    for itr, iva in inner_folds:
                        Xtr, ytr = X_all[itr], y_all[itr]
                        Xva, yva = X_all[iva], y_all[iva]

                        if not valid_split(
                            ytr,
                            yva,
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
                                    Xtr,
                                    ytr,
                                    Xva,
                                    yva,
                                    A,
                                    lr=CFG.train.lr,
                                    weight_decay=CFG.train.weight_decay,
                                    max_epochs=CFG.train.max_epochs,
                                    patience=CFG.train.patience,
                                    batch_size=CFG.train.batch_size,
                                    device=CFG.train.device,
                                    model_hparams=dict(
                                        hid=hid,
                                        heads=heads,
                                        out=out_dim,
                                        dropout=dropout,
                                    ),
                                    seed=sd,
                                )
                                p = 1.0 / (1.0 + np.exp(-res["val"]["logits"]))
                                if USE_AP_SELECTION and len(np.unique(yva)) > 1:
                                    vals.append(
                                        average_precision_score(yva, p)
                                    )
                                else:
                                    eps = 1e-7
                                    bce = -np.mean(
                                        yva * np.log(p + eps)
                                        + (1 - yva) * np.log(1 - p + eps)
                                    )
                                    vals.append(-bce)

                            if vals:
                                if USE_AP_SELECTION:
                                    scores.append(-float(np.mean(vals)))
                                else:
                                    scores.append(-float(np.mean(vals)))
                        else:
                            res = run_one_split_with_tracking(
                                Xtr,
                                ytr,
                                Xva,
                                yva,
                                A,
                                lr=CFG.train.lr,
                                weight_decay=CFG.train.weight_decay,
                                max_epochs=CFG.train.max_epochs,
                                patience=CFG.train.patience,
                                batch_size=CFG.train.batch_size,
                                device=CFG.train.device,
                                model_hparams=dict(
                                    hid=hid,
                                    heads=heads,
                                    out=out_dim,
                                    dropout=dropout,
                                ),
                            )
                            p = 1.0 / (1.0 + np.exp(-res["val"]["logits"]))
                            if USE_AP_SELECTION and len(np.unique(yva)) > 1:
                                scores.append(
                                    -average_precision_score(yva, p)
                                )
                            else:
                                eps = 1e-7
                                bce = -np.mean(
                                    yva * np.log(p + eps)
                                    + (1 - yva) * np.log(1 - p + eps)
                                )
                                scores.append(bce)

                    if not scores:
                        raise TrialPruned(
                            "No valid inner splits for this configuration."
                        )
                    return float(np.mean(scores))

                except TrialPruned:
                    raise
                except Exception as e:  # noqa: BLE001
                    raise TrialPruned(str(e))

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=CFG.cv.random_seed),
            )
            study.optimize(objective, n_trials=20, show_progress_bar=False)

            # save all trials for this study (graph)
            try:
                df_trials = study.trials_dataframe(
                    attrs=(
                        "number",
                        "value",
                        "state",
                        "params",
                        "user_attrs",
                        "system_attrs",
                    )
                )
            except TypeError:
                df_trials = study.trials_dataframe()

            df_trials.insert(0, "pipeline", "graph")
            df_trials.insert(1, "window_s", win_sec)
            df_trials.insert(2, "outer_fold", ofold + 1)

            trial_path = SAVE_DIR / f"optuna_graph_win{win_sec}_fold{ofold + 1}.csv"  # noqa: E501
            df_trials.to_csv(trial_path, index=False)
            print(f"[Saved] {trial_path}")

            best = study.best_trial
            best_row = {
                "pipeline": "graph",
                "window_s": win_sec,
                "outer_fold": ofold + 1,
                "objective": float(best.value),
                **{f"param_{k}": v for k, v in best.params.items()},
            }
            optuna_best_rows.append(best_row)
            optuna_all_trials_rows.append(df_trials)

            b_band = best.params.get("band")
            b_sidx = best.params.get("s_idx")
            hid = best.params["hid"]
            heads = 2
            out_dim = best.params["out"]
            dropout = best.params["dropout"]

            if ofold == 0:
                best_params_graph = {
                    "band": b_band,
                    "scale": float(s_vals[b_sidx]),
                    "hidden": hid,
                    "heads": heads,
                    "out_dim": out_dim,
                    "dropout": dropout,
                }

            X_best, y_best, A_best, _ = feature_store[(b_band, b_sidx, win_sec)]
            Xtr_full, ytr_full = X_best[tr_idx], y_best[tr_idx]
            Xva_full, yva_full = X_best[va_idx], y_best[va_idx]

            logits_ens: List[np.ndarray] = []
            models_ens = []
            for s in range(OUTER_SEEDS):
                res = run_one_split_with_tracking(
                    Xtr_full,
                    ytr_full,
                    Xva_full,
                    yva_full,
                    A_best,
                    lr=CFG.train.lr,
                    weight_decay=CFG.train.weight_decay,
                    max_epochs=CFG.train.max_epochs,
                    patience=CFG.train.patience,
                    batch_size=CFG.train.batch_size,
                    device=CFG.train.device,
                    model_hparams=dict(
                        hid=hid, heads=heads, out=out_dim, dropout=dropout
                    ),
                    seed=CFG.cv.random_seed + s,
                )
                logits_ens.append(res["val"]["logits"])
                models_ens.append(res["model"])
                if s == 0:
                    fold_histories_graph.append(
                        {
                            "train_losses": res["train_losses"],
                            "val_losses": res["val_losses"],
                        }
                    )

            logits_val = np.mean(np.stack(logits_ens, axis=0), axis=0)

            if (
                USE_TEMP_SCALING
                and logits_val.size
                and len(np.unique(yva_full)) > 1
            ):
                T_star = temperature_scale_logits(logits_val, yva_full)
                probs_cal = 1.0 / (1.0 + np.exp(-logits_val / T_star))
            else:
                probs_cal = 1.0 / (1.0 + np.exp(-logits_val))

            scaler = StandardScaler()
            Xva_scaled = scaler.fit_transform(Xva_full)

            model_xai = models_ens[0]
            importance_occ = occlusion_sensitivity_gat(
                model_xai, Xva_scaled, A_best, CFG.train.device
            )
            importance_grad = gradient_input_gat(
                model_xai, Xva_scaled, A_best, CFG.train.device
            )

            xai_results_graph.append(
                {
                    "fold": ofold,
                    "occlusion": importance_occ,
                    "gradient_input": importance_grad,
                }
            )

            yv = yva_full
            pv = probs_cal
            yhat = (pv >= 0.5).astype(int)
            acc = accuracy_score(yv, yhat)
            bacc = balanced_accuracy_score(yv, yhat)
            f1 = f1_score(yv, yhat)
            prec_s = precision_score(yv, yhat, zero_division=0)
            rec_s = recall_score(yv, yhat, zero_division=0)
            aucv = (
                roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")
            )
            apv = (
                average_precision_score(yv, pv)
                if len(np.unique(yv)) > 1
                else float("nan")
            )

            if len(np.unique(yv)) > 1 and len(pv) > 0:
                fpr, tpr, _ = roc_curve(yv, pv)
                roc_auc_val = auc(fpr, tpr)
                prec, rec, _ = precision_recall_curve(yv, pv)
            else:
                fpr, tpr, roc_auc_val = (
                    np.array([0, 1]),
                    np.array([0, 1]),
                    float("nan"),
                )
                prec, rec = np.array([1.0]), np.array([0.0])

            pos_rate = float(yv.mean()) if yv.size else float("nan")

            per_fold_preds_graph.append(
                dict(
                    ytrue=yv,
                    probs=pv,
                    fpr=fpr,
                    tpr=tpr,
                    roc_auc=roc_auc_val,
                    rec=rec,
                    prec=prec,
                    ap=apv,
                    pos_rate=pos_rate,
                )
            )
            per_fold_metrics_graph.append(
                dict(
                    acc=acc,
                    bacc=bacc,
                    f1=f1,
                    prec=prec_s,
                    rec=rec_s,
                    auc=aucv,
                    ap=apv,
                )
            )

            print(
                f"  -> [Graph] Best({b_band}, s={float(s_vals[b_sidx]):.5f}) "
                f"| Val Acc={acc:.3f} AUC={aucv:.3f} AP={apv:.3f}"
            )

    if xai_results_graph:
        all_occlusion_g = np.concatenate(
            [r["occlusion"] for r in xai_results_graph], axis=0
        )
        all_grad_g = np.concatenate(
            [r["gradient_input"] for r in xai_results_graph], axis=0
        )
        importance_dict_g = {
            "Occlusion": all_occlusion_g,
            "Gradient_x_Input": all_grad_g,
        }
        create_xai_summary_table(importance_dict_g, used_ch_names, "gwt_gat")

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

    best_params_class: Dict[str, float | int | str] = {}
    cls_only = [r for r in optuna_best_rows if r.get("pipeline") == "classical"]
    if cls_only:
        best_row = min(cls_only, key=lambda r: r["objective"])
        best_params_class = {
            k.replace("param_", ""): v
            for k, v in best_row.items()
            if k.startswith("param_")
        }
    if not best_params_class:
        best_params_class = {
            "model": "LogReg/SVC",
            "note": "See optuna_best_trials.csv",
        }

    def summarize(metrics_list: List[dict]) -> Dict[str, Tuple[float, float, int]]:
        if not metrics_list:
            return {}
        summary: Dict[str, Tuple[float, float, int]] = {}
        for k in ["acc", "bacc", "f1", "prec", "rec", "auc", "ap"]:
            vals = [m[k] for m in metrics_list if np.isfinite(m[k])]
            if vals:
                summary[k] = (
                    float(np.mean(vals)),
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    len(vals),
                )
        return summary

    def macro_pr_area(per_fold_preds: List[dict]) -> float:
        if not per_fold_preds:
            return float("nan")
        r_grid = np.linspace(0, 1, 200)
        precs_grid: List[np.ndarray] = []
        for pf in per_fold_preds:
            rec, prec, pos_rate = pf["rec"], pf["prec"], pf["pos_rate"]
            if rec.size > 1:
                rec_u, idx = np.unique(rec, return_index=True)
                prec_u = prec[idx]
                if rec_u[-1] < 1.0:
                    rec_u = np.r_[rec_u, 1.0]
                    prec_u = np.r_[prec_u, pos_rate]
                precs_grid.append(
                    np.interp(r_grid, rec_u, prec_u, left=1.0, right=pos_rate)
                )
        return (
            float(np.trapezoid(np.mean(precs_grid, axis=0), r_grid))
            if precs_grid
            else float("nan")
        )

    sum_graph = summarize(per_fold_metrics_graph)
    sum_class = summarize(per_fold_metrics_classical)
    macro_graph = macro_pr_area(per_fold_preds_graph)
    macro_class = macro_pr_area(per_fold_preds_classical)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Outer-fold validation)")
    print("=" * 80)
    print(f"{'Metric':<12} {'Graph (GWT+GAT)':<28} {'Classical (DSP+ML)':<28}")
    for k, label in [
        ("acc", "Accuracy"),
        ("bacc", "BalancedAcc"),
        ("f1", "F1"),
        ("prec", "Precision"),
        ("rec", "Recall"),
        ("auc", "ROC AUC"),
        ("ap", "AP"),
    ]:
        print(
            f"{label:<12} {fmt_triplet(sum_graph.get(k)):<28} "
            f"{fmt_triplet(sum_class.get(k)):<28}"
        )
    print(f"{'Macro PR':<12} {fmt_num(macro_graph, 28)}{fmt_num(macro_class, 28)}")

    # == Reproducibility CSV ==
    write_reproducibility_csv(
        CFG, used_ch_names, sfreq, outer_spec_by_win, n_pairs_by_win
    )

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
        sum_graph,
        sum_class,
        macro_graph,
        macro_class,
        best_params_graph,
        best_params_class,
    )

    plot_all()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {SAVE_DIR.absolute()}")


if __name__ == "__main__":
    main()
