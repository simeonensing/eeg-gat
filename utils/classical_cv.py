from __future__ import annotations

from typing import Dict, Tuple, List

import mne
import numpy as np
import optuna
import pandas as pd
from optuna import TrialPruned
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve

from config import CFG
from pathlib import Path
from config import CFG
from utils.settings import USE_AP_SELECTION
SAVE_DIR = Path(CFG.data.save_dir)
from utils.blocking_splits import plan_inner
from utils.classical_features import build_classical_feature_table_baseline_norm, make_logreg, make_svc, \
    sanity_shuffle_ap, eval_sklearn_model
from utils.utils import hard_purge_train_rows, valid_split
from utils.xai import occlusion_sensitivity_classical, create_xai_summary_table

# main.py
from utils.settings import (
    USE_AP_SELECTION,
)



def classical_nested_cv(
        raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, raw_base: mne.io.BaseRaw,
        outer_spec_by_win: Dict[int, Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int, int]],
        n_pairs_by_win: Dict[int, int], info_mne, ch_names,
        optuna_best_rows: List[dict] | None = None,
        optuna_all_trials_rows: List[pd.DataFrame] | None = None
):
    print("\n[Classical] Starting with SAME CSD preprocessing and SAME outer folds as graph...")
    per_fold_preds = []
    per_fold_metrics = []
    fold_histories = []
    xai_results = []

    for win_sec, (folds, used_K, used_bs, used_E) in sorted(outer_spec_by_win.items()):
        n_pairs_ref = n_pairs_by_win[win_sec]
        try:
            X_all, y_all, n_pairs = build_classical_feature_table_baseline_norm(
                raw_before, raw_after, raw_base, win_sec, n_pairs_reference=n_pairs_ref
            )
        except Exception as e:
            print(f"[Classical] win={win_sec}s failed feature build ({e}); skipping window.")
            continue

        expected_rows = n_pairs_ref * 2
        if X_all.shape[0] != expected_rows:
            print(f"[Classical] ERROR: Row mismatch! Expected {expected_rows}, got {X_all.shape[0]}. SKIPPING.")
            continue

        print(f"\n[Classical] Window {win_sec}s: using {used_K} SHARED outer folds (bs={used_bs}, embargo={used_E})")

        n_channels = len(ch_names)
        n_bands = len(CFG.bands)

        for ofold in range(used_K):
            tr_idx_base, va_idx = folds[ofold]
            if np.max(tr_idx_base) >= len(y_all) or np.max(va_idx) >= len(y_all):
                print(f"[Classical Outer {ofold + 1}/{used_K}] ERROR: Indices out of bounds! SKIPPING.")
                continue

            actual_purge = CFG.cv.purge_pairs
            tr_idx = hard_purge_train_rows(tr_idx_base, va_idx, purge_pairs=actual_purge)
            if tr_idx.size == 0:
                print(f"[Classical Outer {ofold + 1}/{used_K}] SKIPPED: purging removed all training dataset.")
                continue

            y_tr, y_va = y_all[tr_idx], y_all[va_idx]
            if not valid_split(
                    y_tr, y_va, min_train=CFG.cv.min_train_rows, min_val=CFG.cv.min_val_rows,
                    min_pos=CFG.cv.min_pos_per_split, min_neg=CFG.cv.min_neg_per_split, ratio_lo=CFG.cv.ratio_lo
            ):
                print(f"[Classical Outer {ofold + 1}/{used_K}] SKIPPED: split failed validation after purge.")
                continue

            print(
                f"[Classical Outer {ofold + 1}/{used_K}] |train|={len(tr_idx)} |val|={len(va_idx)} [purge±{actual_purge}]")

            inner_folds = plan_inner(tr_idx, y_all, used_bs)
            if len(inner_folds) < 2:
                print(f"  [Classical] WARN: insufficient inner folds; skipping outer fold {ofold + 1}.")
                continue

            def objective(trial: optuna.Trial):
                try:
                    model_name = trial.suggest_categorical("model", ["logreg", "svc"])
                    if model_name == "logreg":
                        C = trial.suggest_float("C", 1e-3, 0.5, log=True)
                        max_pca = trial.suggest_int("max_pca", 5, 12)
                        factory = lambda: make_logreg(C=C, max_pca=max_pca)
                    else:
                        C = trial.suggest_float("C", 1e-3, 0.5, log=True)
                        max_pca = trial.suggest_int("max_pca", 5, 12)
                        factory = lambda: make_svc(C=C, gamma="scale", max_pca=max_pca)

                    scores = []
                    for itr, iva in inner_folds:
                        Xtr, ytr = X_all[itr], y_all[itr]
                        Xva, yva = X_all[iva], y_all[iva]
                        if not valid_split(
                                ytr, yva, min_train=CFG.cv.min_train_rows_inner, min_val=CFG.cv.min_val_rows_inner,
                                min_pos=CFG.cv.min_pos_per_split_inner, min_neg=CFG.cv.min_neg_per_split_inner,
                                ratio_lo=CFG.cv.ratio_lo_inner
                        ): continue
                        m = factory()
                        m.fit(Xtr, ytr)
                        p = m.predict_proba(Xva)[:, 1]
                        if USE_AP_SELECTION:
                            if len(np.unique(yva)) > 1:
                                scores.append(-average_precision_score(yva, p))
                        else:
                            eps = 1e-7
                            bce = -np.mean(yva * np.log(p + eps) + (1 - yva) * np.log(1 - p + eps))
                            if np.isfinite(bce): scores.append(bce)

                    if not scores: raise TrialPruned("No valid inner splits.")
                    return float(np.mean(scores))
                except TrialPruned:
                    raise
                except Exception as e:
                    raise TrialPruned(str(e))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=CFG.cv.random_seed))
            study.optimize(objective, n_trials=15, show_progress_bar=False)

            # save all trials for this study (classical)
            try:
                df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "system_attrs"))
            except TypeError:
                df_trials = study.trials_dataframe()
            df_trials.insert(0, "pipeline", "classical")
            df_trials.insert(1, "window_s", win_sec)
            df_trials.insert(2, "outer_fold", ofold + 1)
            trial_path = SAVE_DIR / f"optuna_classical_win{win_sec}_fold{ofold+1}.csv"
            df_trials.to_csv(trial_path, index=False)
            print(f"[Saved] {trial_path}")

            best = study.best_trial
            if optuna_best_rows is not None:
                optuna_best_rows.append({
                    "pipeline": "classical",
                    "window_s": win_sec,
                    "outer_fold": ofold + 1,
                    "objective": float(best.value),
                    **{f"param_{k}": v for k, v in best.params.items()}
                })
            if optuna_all_trials_rows is not None:
                optuna_all_trials_rows.append(df_trials)

            if best.params.get("model") == "logreg":
                mk_best = lambda: make_logreg(C=best.params["C"], max_pca=best.params["max_pca"])
            else:
                mk_best = lambda: make_svc(C=best.params["C"], gamma="scale", max_pca=best.params["max_pca"])

            Xtr_full, ytr_full = X_all[tr_idx], y_all[tr_idx]
            Xva_full, yva_full = X_all[va_idx], y_all[va_idx]

            ap_sanity = sanity_shuffle_ap(mk_best, Xtr_full, ytr_full, Xva_full, yva_full, n=3)
            print(f"  [Classical] Sanity AP (shuffled labels): {ap_sanity:.3f}")

            model = mk_best()
            model.fit(Xtr_full, ytr_full)
            metrics = eval_sklearn_model(model, Xva_full, yva_full)
            print(f"  -> Classical Val Acc={metrics['acc']:.3f} AUC={metrics['auc']:.3f} AP={metrics['ap']:.3f}")

            importance_occ = occlusion_sensitivity_classical(model, Xva_full, n_channels, n_bands)
            xai_results.append({
                'fold': ofold,
                'occlusion': importance_occ
            })

            yv, pv = metrics['ytrue'], metrics['probs']
            if len(np.unique(yv)) > 1 and len(pv) > 0:
                fpr, tpr, _ = roc_curve(yv, pv); roc_auc = auc(fpr, tpr)
                prec, rec, _ = precision_recall_curve(yv, pv); ap_val = average_precision_score(yv, pv)
            else:
                fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), float('nan')
                prec, rec, ap_val = np.array([1.0]), np.array([0.0]), float('nan')

            pos_rate = float(yv.mean()) if yv.size else float('nan')
            per_fold_preds.append(dict(
                ytrue=yv, probs=pv, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                rec=rec, prec=prec, ap=ap_val, pos_rate=pos_rate
            ))
            per_fold_metrics.append(metrics)

    if xai_results:
        all_occlusion = np.concatenate([r['occlusion'] for r in xai_results], axis=0)
        importance_dict = {'Occlusion': all_occlusion}
        create_xai_summary_table(importance_dict, ch_names, "classical", SAVE_DIR)

    return per_fold_preds, per_fold_metrics, fold_histories
