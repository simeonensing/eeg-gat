from __future__ import annotations

from typing import Dict, Tuple, Callable

import mne
import numpy as np
from scipy.signal import welch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CFG
from utils.shared_preprocessing import windows_for_length



def compute_bandpowers_epochwise(data_uV: np.ndarray, sfreq: float, bands: Dict[str, Tuple[float, float]]):
    nperseg = min(1024, int(2.0 * sfreq))
    noverlap = nperseg // 2
    band_names = list(bands.keys())
    bp = []
    for ch in range(data_uV.shape[0]):
        freqs, psd = welch(data_uV[ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap, scaling='density')
        vals = []
        for name in band_names:
            fmin, fmax = bands[name]
            mask = (freqs >= fmin) & (freqs <= fmax)
            vals.append(0.0 if not np.any(mask) else np.trapezoid(psd[mask], freqs[mask]))
        bp.append(vals)
    return np.array(bp), np.array(freqs)


def build_classical_feature_table_baseline_norm(
        raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, raw_base: mne.io.BaseRaw,
        win_sec: int, n_pairs_reference: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    sfreq = float(raw_before.info['sfreq'])
    wins_b = windows_for_length(raw_before.n_times, sfreq, win_sec, step_sec=0, include_tail=True)
    wins_a = windows_for_length(raw_after.n_times, sfreq, win_sec, step_sec=0, include_tail=True)
    n_pairs = min(len(wins_b), len(wins_a))
    if n_pairs != n_pairs_reference:
        print(f"  [Classical WARNING] Pair count mismatch: graph={n_pairs_reference}, classical={n_pairs}. Using min.")
        n_pairs = min(n_pairs, n_pairs_reference)
    if n_pairs < 2: raise RuntimeError(f"[Classical] Too few window pairs for win={win_sec}s.")
    base_uV = (raw_base.get_data() * 1e6).astype(np.float64, copy=False)
    bp_base, _ = compute_bandpowers_epochwise(base_uV, sfreq, CFG.bands)
    Xs, ys = [], []
    eps = 1e-8
    for i in range(n_pairs):
        s_b, e_b = wins_b[i]
        s_a, e_a = wins_a[i]
        seg_b = (raw_before.get_data(start=s_b, stop=e_b) * 1e6).astype(np.float64, copy=False)
        seg_a = (raw_after.get_data(start=s_a, stop=e_a) * 1e6).astype(np.float64, copy=False)
        bp_b, _ = compute_bandpowers_epochwise(seg_b, sfreq, CFG.bands)
        bp_a, _ = compute_bandpowers_epochwise(seg_a, sfreq, CFG.bands)
        nb = 10.0 * np.log10((bp_b + eps) / (bp_base + eps))
        na = 10.0 * np.log10((bp_a + eps) / (bp_base + eps))
        nb = np.log(np.clip((nb - nb.min(axis=1, keepdims=True)) /
                            (nb.max(axis=1, keepdims=True) - nb.min(axis=1, keepdims=True) + eps), 1e-6, 1.0))
        na = np.log(np.clip((na - na.min(axis=1, keepdims=True)) /
                            (na.max(axis=1, keepdims=True) - na.min(axis=1, keepdims=True) + eps), 1e-6, 1.0))
        Xs.append(nb.flatten()); ys.append(0)
        Xs.append(na.flatten()); ys.append(1)
    X = np.vstack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=int)
    return X, y, n_pairs


def _calibrated(est, cv=3, method="sigmoid"):
    try:
        return CalibratedClassifierCV(estimator=est, cv=cv, method=method)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=est, cv=cv, method=method)


def make_logreg(C=1.0, max_pca=15, seed=CFG.random_seed):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=min(max_pca, 15), svd_solver='full', random_state=seed)),
        ("clf", _calibrated(LogisticRegression(C=C, class_weight="balanced",
                                               max_iter=300, solver="lbfgs",
                                               random_state=seed), cv=3, method="sigmoid"))
    ])


def make_svc(C=1.0, gamma="scale", max_pca=15, seed=CFG.random_seed):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=min(max_pca, 15), svd_solver='full', random_state=seed)),
        ("clf", _calibrated(SVC(C=C, kernel="rbf", gamma=gamma, probability=False,
                                class_weight="balanced", random_state=seed), cv=3, method="sigmoid"))
    ])


def eval_sklearn_model(model, Xva, yva):
    p = model.predict_proba(Xva)[:, 1]
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(yva, yhat)
    bacc = balanced_accuracy_score(yva, yhat)
    f1 = f1_score(yva, yhat)
    prec = precision_score(yva, yhat, zero_division=0)
    rec = recall_score(yva, yhat, zero_division=0)
    aucv = roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else float('nan')
    ap = average_precision_score(yva, p) if len(np.unique(yva)) > 1 else float('nan')
    return dict(acc=acc, bacc=bacc, f1=f1, prec=prec, rec=rec, auc=aucv, ap=ap, probs=p, ytrue=yva)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE with uniform bins over [0,1]."""
    if y_prob.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(sel):
            continue
        bin_conf = y_prob[sel].mean()
        bin_acc  = y_true[sel].mean()
        w = sel.mean()  # weight by bin frequency
        ece += w * abs(bin_acc - bin_conf)
    return float(ece)


def sanity_shuffle_ap(model_factory: Callable[[], object], Xtr, ytr, Xva, yva, n=3):
    rng = np.random.default_rng(CFG.random_seed)
    aps = []
    for _ in range(n):
        ytr_shuf = ytr.copy()
        rng.shuffle(ytr_shuf)
        m = model_factory()
        m.fit(Xtr, ytr_shuf)
        p = m.predict_proba(Xva)[:, 1]
        if len(np.unique(yva)) > 1:
            aps.append(average_precision_score(yva, p))
    return float(np.mean(aps)) if aps else float('nan')
