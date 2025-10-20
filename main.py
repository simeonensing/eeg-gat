#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GWT+GAT (graph) vs Classical DSP+ML with XAI analysis.
(Full file with robust dashboard launch fallbacks.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import contextlib
import os
import sys
os.system("bash scripts/cleanup_dashboards.sh --force --keep-results")

# --- Reproducibility toggles ---
os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # determinism on CUDA
try:
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

# ---- sensible defaults for local dashboards (override via env) ----
os.environ.setdefault("AUTO_DASHBOARDS", "1")
os.environ.setdefault("DASH_HOST", "127.0.0.1")
os.environ.setdefault("OPTUNA_PORT", "8080")
os.environ.setdefault("MLFLOW_PORT", "5000")
os.environ.setdefault("TB_PORT", "6006")

import numpy as np
import pandas as pd

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")

from utils.results_writer import (
    build_macro_pr_csv,
    write_fold_artifacts,
    finalize_prefix,
    write_topomap_values_csv,
    write_topomap_consensus_csv,
)

# --- Global font settings (Times-like) ---
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Sklearn
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
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

# TensorBoard (Optuna integration)
try:
    from optuna.integration import TensorBoardCallback  # Optuna >=2.9
except Exception:  # pragma: no cover
    try:
        from optuna.integration.tensorboard import TensorBoardCallback  # older layout
    except Exception:
        TensorBoardCallback = None

# MLflow (optional)
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

# Internal modules
from config import CFG
from utils.blocking_splits import plan_inner, plan_outer
from utils.classical_cv import classical_nested_cv
from utils.gwt_features import build_gwt_feature_table
from utils.shared_preprocessing import align_channels, prep_raw
from utils.spectral_graph import band_baseline_from_base, build_graph_info, full_power
from utils.summary_helpers import export_results_to_csv, fmt_triplet
from utils.temp_scaling import temperature_scale_logits
from utils.train_eval import run_one_split_with_tracking
from utils.utils import (
    hard_purge_train_rows, set_all_seeds, valid_split, write_reproducibility_csv,
)
from utils.xai import create_xai_summary_table, gradient_input_gat, occlusion_sensitivity_gat
from utils.plotting_helpers import plot_all

# main.py
from utils.settings import USE_TEMP_SCALING, AVG_TWO_SEEDS_INNER, OUTER_SEEDS, USE_AP_SELECTION

# ========================== CONFIG ==========================
SAVE_DIR = Path(CFG.data.save_dir).expanduser().resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "figures").mkdir(parents=True, exist_ok=True)

# ---------------- Tracking helpers ----------------
def _abs_optuna_url(storage_url: str | None) -> str:
    """Absolute Optuna DSN for display."""
    from pathlib import Path as _P
    if not storage_url:
        return ""
    if storage_url.startswith("sqlite:///"):
        raw = storage_url[len("sqlite:///"):]
        p = _P(raw)
        if not p.is_absolute():
            p = (_P.cwd() / p).resolve()
        return f"sqlite:///{p.as_posix()}"
    if storage_url.startswith("journal://"):
        raw = storage_url[len("journal://"):]
        p = _P(raw)
        if not p.is_absolute():
            p = (_P.cwd() / p).resolve()
        return f"journal://{p.as_posix()}"
    return storage_url

def _abs_path_from_uri(uri: str | None) -> str:
    """Resolve MLflow/TensorBoard URIs (file:/ or relative) to absolute POSIX paths."""
    if not uri:
        return ""
    from pathlib import Path as _P
    if uri.startswith("file:"):
        raw = uri[len("file:"):]
        p = _P(raw)
        if not p.is_absolute():
            p = (_P.cwd() / p).resolve()
        return f"file:{p.as_posix()}"
    return uri

# ---- one-time dashboard launchers ------------------------------------------------
_DASH_ONCE = False
_OPTUNA_DASH_ONCE = False
_MLFLOW_DASH_ONCE = False
_TB_DASH_ONCE = False

def _pick_port(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _spawn_first(cmd_variants: list[list[str]]) -> bool:
    """Try each command variant until one launches; silence output."""
    import subprocess
    for cmd in cmd_variants:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False

def _launch_optuna_dashboard_once(storage_url: str | None) -> None:
    """Launch Optuna Dashboard; robust to PATH/env differences."""
    import shutil
    from pathlib import Path as _P
    global _OPTUNA_DASH_ONCE
    if _OPTUNA_DASH_ONCE or not storage_url:
        return

    host = os.environ.get("DASH_HOST", "127.0.0.1")
    oport = _pick_port("OPTUNA_PORT", 8080)

    # Normalize DSN and ensure sqlite file exists
    optuna_abs = storage_url
    if storage_url.startswith("sqlite:///"):
        raw = storage_url[len("sqlite:///"):]
        p = _P(raw)
        if not p.is_absolute():
            p = (_P.cwd() / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
        optuna_abs = f"sqlite:///{p.as_posix()}"

    # Candidate launchers: env override, PATH CLI, known absolute, python -m
    cli_env = os.environ.get("OPTUNA_DASH_PATH")  # e.g. /Scratch/local/bin/optuna-dashboard
    cli_path = shutil.which("optuna-dashboard")
    candidates: list[list[str]] = []
    if cli_env:
        candidates.append([cli_env, "--host", host, "--port", str(oport), optuna_abs])
    if cli_path:
        candidates.append([cli_path, "--host", host, "--port", str(oport), optuna_abs])
    # common local bin fallback (your system had this earlier)
    # last resort: use the current Python env (works only if module installed there)
    candidates.append([sys.executable, "-m", "optuna_dashboard.app", "--host", host, "--port", str(oport), optuna_abs])
    candidates.append([sys.executable, "-m", "optuna_dashboard",      "--host", host, "--port", str(oport), optuna_abs])

    if _spawn_first(candidates):
        print(f"[DASH] optuna    ⇒ http://{host}:{oport}")
        print(f"[INFO] Optuna Dashboard URL: http://{host}:{oport}")
        print(f"       (storage={optuna_abs})")
        _OPTUNA_DASH_ONCE = True
    else:
        print("➤ Optuna Dashboard could not be launched automatically.")
        print("  Try one of these (depending on where it's installed):")
        print(f"  {sys.executable} -m pip install --upgrade optuna-dashboard")
        print(f"  optuna-dashboard --host {host} --port {oport} \"{optuna_abs}\"")
        print(f"  {sys.executable} -m optuna_dashboard.app --host {host} --port {oport} \"{optuna_abs}\"")

def _launch_mlflow_ui_once(tracking_uri: str | None) -> None:
    import shutil
    global _MLFLOW_DASH_ONCE
    if _MLFLOW_DASH_ONCE or not tracking_uri:
        return
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    mport = _pick_port("MLFLOW_PORT", 5000)
    cli_env = os.environ.get("MLFLOW_CLI_PATH")  # optional override
    cli_path = shutil.which("mlflow")
    uri = _abs_path_from_uri(tracking_uri)
    candidates: list[list[str]] = []
    if cli_env:
        candidates.append([cli_env, "ui", "--backend-store-uri", uri, "--host", host, "--port", str(mport)])
    if cli_path:
        candidates.append([cli_path, "ui", "--backend-store-uri", uri, "--host", host, "--port", str(mport)])
    candidates.append([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--host", host, "--port", str(mport)])

    if _spawn_first(candidates):
        print(f"[DASH] mlflow    ⇒ http://{host}:{mport}")
        print(f"[INFO] MLflow UI URL:       http://{host}:{mport}")
        print(f"       (backend-store-uri={uri})")
        _MLFLOW_DASH_ONCE = True
    else:
        print("[DASH] MLflow UI could not be launched automatically.")
        print(f"  {sys.executable} -m pip install --upgrade mlflow")
        print(f"  mlflow ui --backend-store-uri \"{uri}\" --host {host} --port {mport}")

def _launch_tensorboard_once(tb_dir: Path | None) -> None:
    import shutil
    global _TB_DASH_ONCE
    if _TB_DASH_ONCE or not tb_dir:
        return
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    tport = _pick_port("TB_PORT", 6006)
    cli_env = os.environ.get("TB_CLI_PATH")  # optional override
    cli_path = shutil.which("tensorboard")
    candidates: list[list[str]] = []
    if cli_env:
        candidates.append([cli_env, "--logdir", str(tb_dir), "--host", host, "--port", str(tport)])
    if cli_path:
        candidates.append([cli_path, "--logdir", str(tb_dir), "--host", host, "--port", str(tport)])
    candidates.append([sys.executable, "-m", "tensorboard.main", "--logdir", str(tb_dir), "--host", host, "--port", str(tport)])

    if _spawn_first(candidates):
        print(f"[DASH] tboard    ⇒ http://{host}:{tport}")
        print(f"[INFO] TensorBoard URL:     http://{host}:{tport}")
        print(f"       (logdir={tb_dir})")
        _TB_DASH_ONCE = True
    else:
        print("[DASH] TensorBoard could not be launched automatically.")
        print(f"  {sys.executable} -m pip install --upgrade tensorboard")
        print(f"  tensorboard --logdir \"{tb_dir}\" --host {host} --port {tport}")

def _maybe_launch_all_dashboards(tracking: dict) -> None:
    """Call once per process to auto-launch + print URLs for all dashboards."""
    auto = os.environ.get("AUTO_DASHBOARDS", "1").lower() not in ("0", "false", "no")
    if auto:
        _launch_optuna_dashboard_once(tracking.get("storage_url"))
        _launch_mlflow_ui_once(tracking.get("ml_tracking"))
        _launch_tensorboard_once(tracking.get("tb_root"))
    else:
        host = os.environ.get("DASH_HOST", "127.0.0.1")
        print(f"[INFO] AUTO_DASHBOARDS disabled. Use these:")
        if tracking.get("storage_url"):
            print(f"  optuna-dashboard --host {host} --port {_pick_port('OPTUNA_PORT',8080)} \"{_abs_optuna_url(tracking['storage_url'])}\"")
        if tracking.get("ml_tracking"):
            print(f"  mlflow ui --backend-store-uri \"{_abs_path_from_uri(tracking['ml_tracking'])}\" --host {host} --port {_pick_port('MLFLOW_PORT',5000)}")
        if tracking.get("tb_root"):
            print(f"  tensorboard --logdir \"{tracking['tb_root']}\" --host {host} --port {_pick_port('TB_PORT',6006)}")

def _resolve_tracking():
    """Read optional settings from CFG and materialize paths."""
    # -------- Optuna --------
    optuna_cfg = getattr(CFG, "optuna", object())
    storage_url = getattr(optuna_cfg, "storage_url", None)  # None => in-memory
    study_prefix = getattr(optuna_cfg, "study_prefix", "study")
    n_trials = getattr(optuna_cfg, "n_trials", 20)
    seed = getattr(optuna_cfg, "seed", getattr(CFG.cv, "random_seed", 42))
    # -------- MLflow --------
    mlflow_cfg = getattr(CFG, "mlflow", object())
    ml_tracking = getattr(mlflow_cfg, "tracking_uri", None)
    ml_experiment = getattr(mlflow_cfg, "experiment_name", "EEG-GWT-GAT")
    ml_tags = getattr(mlflow_cfg, "tags", {}) or {}
    # -------- TensorBoard --------
    tb_cfg = getattr(CFG, "tensorboard", object())
    tb_root = Path(getattr(tb_cfg, "log_dir", SAVE_DIR / "tb")).expanduser().resolve()
    tb_root.mkdir(parents=True, exist_ok=True)
    return {
        "storage_url": storage_url,
        "study_prefix": study_prefix,
        "n_trials": int(n_trials),
        "seed": int(seed),
        "ml_tracking": ml_tracking,
        "ml_experiment": ml_experiment,
        "ml_tags": dict(ml_tags),
        "tb_root": tb_root,
    }

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

            # ---------- Integrations (Optuna/MLflow/TensorBoard) ----------
            tracking = _resolve_tracking()

            # Launch dashboards once per process and print URLs
            global _DASH_ONCE
            if not _DASH_ONCE:
                _maybe_launch_all_dashboards(tracking)
                _DASH_ONCE = True

            # TensorBoard dashboard hint for this run's subdir
            _tb_root = getattr(tracking['tb_root'], 'as_posix', lambda: str(tracking['tb_root']))()
            print(f"[INFO] TensorBoard log directory: {_tb_root}")
            print("[INFO] Launch TensorBoard with:")
            print(f"       tensorboard --logdir \"{_tb_root}\" --port {os.environ.get('TB_PORT','6006')}")

            # Name the study deterministically
            time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
            study_name = f"{tracking['study_prefix']}_graph_win{win_sec}_fold{ofold+1}_{time_tag}"

            # Optional: MLflow setup (parent run)
            active_mlflow_parent = None
            if tracking["ml_tracking"] and mlflow is not None:
                mlflow.set_tracking_uri(tracking["ml_tracking"])
                mlflow.set_experiment(tracking["ml_experiment"])
                print(f"[INFO] MLflow tracking URI: {tracking['ml_tracking']}")
                print(f"[INFO] Launch MLflow UI with:")
                print(f"       mlflow ui --backend-store-uri \"{_abs_path_from_uri(tracking['ml_tracking'])}\" --port {os.environ.get('MLFLOW_PORT','5000')}")
                active_mlflow_parent = mlflow.start_run(run_name=study_name)
                mlflow.set_tags({
                    "pipeline": "graph",
                    "window_s": win_sec,
                    "outer_fold": ofold + 1,
                    **tracking["ml_tags"],
                })

            # TensorBoard callback (per-trial)
            tb_cb = None
            tb_dir = tracking["tb_root"] / f"{study_name}"
            print(f"[INFO] TensorBoard log directory: {tb_dir}")
            if TensorBoardCallback is None:
                print("[WARN] Optuna TensorBoardCallback is unavailable. Install tensorboard:")
                print("       pip install tensorboard  # or: pip install tensorboardX")

            if TensorBoardCallback is not None:
                tb_cb = TensorBoardCallback(str(tb_dir), metric_name="objective")

            # Objective (with manual nested MLflow run per trial)
            def objective(trial: optuna.Trial) -> float:
                if mlflow is not None and tracking["ml_tracking"]:
                    child_ctx = mlflow.start_run(run_name=f"trial-{trial.number}", nested=True)
                else:
                    child_ctx = contextlib.nullcontext()

                with child_ctx:
                    try:
                        trial.set_user_attr("window_s", win_sec)
                        trial.set_user_attr("outer_fold", ofold + 1)

                        band_name = trial.suggest_categorical("band", CFG.band_order)
                        s_idx     = trial.suggest_categorical("s_idx", list(range(len(s_vals))))
                        hid       = trial.suggest_categorical("hid", [8, 12, 16])
                        heads     = 2
                        out_dim   = trial.suggest_categorical("out", [8, 12])
                        dropout   = trial.suggest_float("dropout", 0.28, 0.38)

                        if mlflow is not None and tracking["ml_tracking"]:
                            mlflow.log_params({
                                "band": band_name, "s_idx": s_idx, "hid": hid, "heads": heads, "out": out_dim, "dropout": dropout,
                            })

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

                        value = float(np.mean(scores))
                        trial.report(value, step=0)
                        # --- Always write at least one scalar to TensorBoard ---
                        try:
                            # Prefer PyTorch’s built-in writer if available
                            from torch.utils.tensorboard import SummaryWriter as _TBWriter  # type: ignore
                            _w = _TBWriter(log_dir=str(tb_dir))
                            _w.add_scalar("objective", float(value), trial.number)
                            _w.flush();
                            _w.close()
                        except Exception:
                            try:
                                # Fallback to tensorboardX if installed
                                from tensorboardX import SummaryWriter as _TBWriter  # type: ignore
                                _w = _TBWriter(logdir=str(tb_dir))
                                _w.add_scalar("objective", float(value), trial.number)
                                _w.flush();
                                _w.close()
                            except Exception as _e:
                                print(f"[WARN] Could not write TensorBoard scalar: {_e}")

                        if mlflow is not None and tracking["ml_tracking"]:
                            mlflow.log_metric("objective", value)

                        return value

                    except TrialPruned:
                        raise
                    except Exception as e:  # noqa: BLE001
                        if mlflow is not None and tracking["ml_tracking"]:
                            mlflow.set_tag("error", str(e))
                        raise TrialPruned(str(e))

            # Create study (named). Dashboard-ready if storage is set.
            sampler = optuna.samplers.TPESampler(seed=tracking["seed"])
            if tracking["storage_url"]:
                study = optuna.create_study(
                    direction="minimize", sampler=sampler,
                    storage=tracking["storage_url"], study_name=study_name, load_if_exists=True,
                )
                print(f"[INFO] Optuna storage: {tracking['storage_url']}")
                print("[INFO] Launch dashboard with:")
                print(f"       optuna-dashboard \"{_abs_optuna_url(tracking['storage_url'])}\"")
            else:
                study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)

            # Attach metadata
            study.set_user_attr("pipeline", "graph")
            study.set_user_attr("window_s", win_sec)
            study.set_user_attr("outer_fold", ofold + 1)

            callbacks = []
            if TensorBoardCallback is not None:
                callbacks.append(TensorBoardCallback(str(tb_dir), metric_name="objective"))

            try:
                study.optimize(
                    objective,
                    n_trials=tracking["n_trials"],
                    callbacks=callbacks if callbacks else None,
                    show_progress_bar=False,
                )
            finally:
                if mlflow is not None:
                    try:
                        active_mlflow_parent and mlflow.end_run()
                    except Exception:
                        pass

            # save all trials for this study (graph)
            try:
                df_trials = study.trials_dataframe(
                    attrs=("number", "value", "state", "params", "user_attrs", "system_attrs")
                )
            except TypeError:
                df_trials = study.trials_dataframe()
            df_trials.insert(0, "pipeline", "graph")
            df_trials.insert(1, "window_s", win_sec)
            df_trials.insert(2, "outer_fold", ofold + 1)
            df_trials.insert(3, "study_name", study.study_name)
            trial_path = SAVE_DIR / f"optuna_graph_win{win_sec}_fold{ofold + 1}.csv"
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
        xai_results_classical,
    ) = classical_nested_cv(
        raw_before=raw_before, raw_after=raw_after, raw_base=raw_base,
        outer_spec_by_win=outer_spec_by_win, n_pairs_by_win=n_pairs_by_win,
        info_mne=info_mne, ch_names=used_ch_names,
        optuna_best_rows=optuna_best_rows, optuna_all_trials_rows=optuna_all_trials_rows,
    )

    if xai_results_classical:
        all_occ_c = np.concatenate([r["occlusion"] for r in xai_results_classical], axis=0)
        importance_dict_c = {"Occlusion": all_occ_c}
        write_topomap_values_csv(importance_dict_c, used_ch_names, "classical", SAVE_DIR)
        write_topomap_consensus_csv(importance_dict_c, used_ch_names, "classical", SAVE_DIR)

    # ---- per-fold artifacts for classical ----
    for i, pf in enumerate(per_fold_preds_classical, 1):
        yv = pf.get("ytrue"); pv = pf.get("probs")
        if yv is None or pv is None:
            continue
        if len(np.unique(yv)) > 1 and len(pv) > 0:
            fpr, tpr, _ = roc_curve(yv, pv)
        else:
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
        write_fold_artifacts(prefix="classical", fold=i, y_true=yv, y_prob=pv, outdir=SAVE_DIR, fpr=fpr, tpr=tpr)

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

    # Macro PR
    build_macro_pr_csv("gwt_gat", SAVE_DIR)
    build_macro_pr_csv("classical", SAVE_DIR)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Outer-fold validation)")
    print("=" * 80)
    print(f"{'Metric':<12} {'Graph (GWT+GAT)':<28} {'Classical (DSP+ML)':<28}")
    for k, label in [
        ("acc",  "Accuracy"), ("bacc", "BalancedAcc"), ("f1", "F1"),
        ("prec", "Precision"), ("rec", "Recall"), ("auc", "ROC AUC"), ("ap", "AP"),
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
        sum_graph, sum_class, float("nan"), float("nan"),
        best_params_graph, best_params_class,
    )

    finalize_prefix("gwt_gat", SAVE_DIR)
    finalize_prefix("classical", SAVE_DIR)

    plot_all()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {SAVE_DIR.absolute()}")

if __name__ == "__main__":
    main()
