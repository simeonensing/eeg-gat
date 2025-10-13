from __future__ import annotations

from typing import Dict

import numpy as np

from config import CFG
from utils.shared_preprocessing import windows_for_length



def window_gwt_maps(power_full, band_tuple, wstart, wend, baseline_vec, G, meyer, degree_centrality=None,
                    all_freqs=None):
    fmin, fmax = band_tuple
    freqs = CFG.all_freqs if all_freqs is None else all_freqs
    fmask = (freqs >= fmin) & (freqs <= fmax)
    band_pw_win = power_full[:, fmask, wstart:wend].mean(axis=(1, 2))
    x = 10.0 * np.log10((band_pw_win + 1e-12) / (baseline_vec + 1e-12))
    if degree_centrality is not None: x = x * degree_centrality
    x = (x - x.mean()) / (x.std() + 1e-10)
    x_hat = G.U.T @ x
    kernels = meyer._kernels
    wavelet_inds = list(range(1, len(kernels)))
    maps = [G.U @ (kernels[k](G.e) * x_hat) for k in wavelet_inds]
    return np.stack(maps, axis=0)


def build_gwt_feature_table(
        power_bef: np.ndarray, power_aft: np.ndarray,
        baseline_by_band: Dict[str, np.ndarray],
        G, meyer, degree_centrality, sfreq: float,
        used_band: str, s_index: int, win_sec: int
):
    band_tuple = CFG.bands[used_band]
    wins_b = windows_for_length(power_bef.shape[2], sfreq, win_sec, step_sec=0, include_tail=True)
    wins_a = windows_for_length(power_aft.shape[2], sfreq, win_sec, step_sec=0, include_tail=True)
    n_pairs = min(len(wins_b), len(wins_a))
    if n_pairs < 2: raise RuntimeError(f"Too few window pairs for win={win_sec}s.")
    Xs, ys = [], []
    base_vec = baseline_by_band[used_band]
    for i in range(n_pairs):
        s_b, e_b = wins_b[i]
        s_a, e_a = wins_a[i]
        maps_b = window_gwt_maps(power_bef, band_tuple, s_b, e_b, base_vec, G, meyer, degree_centrality)
        maps_a = window_gwt_maps(power_aft, band_tuple, s_a, e_a, base_vec, G, meyer, degree_centrality)
        Xs.append(maps_b[s_index]); ys.append(0)
        Xs.append(maps_a[s_index]); ys.append(1)
    X = np.vstack(Xs)
    y = np.asarray(ys, int)
    A = np.asarray(G.W.todense(), float)
    return X, y, A, n_pairs
