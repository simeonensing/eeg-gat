from __future__ import annotations
try:
    from pygsp import graphs, filters as gsp_filters
except Exception:
    graphs = None
    gsp_filters = None
from pathlib import Path
from config import CFG

import mne
import numpy as np
from mne import create_info
from mne.time_frequency import tfr_array_morlet

try:
    from pygsp import graphs, filters as gsp_filters
except Exception:
    graphs = None
    gsp_filters = None


def full_power(data_uV, sfreq, all_freqs, n_cycles):
    X = data_uV - data_uV.mean(axis=1, keepdims=True)
    P = tfr_array_morlet(X[np.newaxis, :, :], sfreq=sfreq, freqs=all_freqs, n_cycles=n_cycles, output='power')[0]
    return P


def band_baseline_from_base(data_base_uV, sfreq, band, all_freqs, n_cycles):
    fmin, fmax = band
    fmask = (all_freqs >= fmin) & (all_freqs <= fmax)
    base_pw = tfr_array_morlet(data_base_uV[np.newaxis, :, :], sfreq=sfreq, freqs=all_freqs,
                               n_cycles=n_cycles, output='power')[0]
    return base_pw[:, fmask, :].mean(axis=(1, 2))


def _standard_positions(name: str):
    try:
        std = mne.channels.make_standard_montage(name)
        return std.get_positions()['ch_pos']
    except Exception:
        return {}


def build_graph_info(raw_for_positions, used_ch_names, montage_name, n_scales, s_max):
    # ---- gather multiple sources of positions ----
    mont = raw_for_positions.get_montage()
    pos_mont = mont.get_positions()['ch_pos'] if mont is not None else {}
    pos_main = _standard_positions(montage_name)          # requested montage
    pos_1005 = _standard_positions("standard_1005")       # dense fallback
    pos_1020 = _standard_positions("standard_1020")       # coarse fallback

    # allow AF3/AF4 <-> Fp1/Fp2 aliasing when the exact key is missing
    alias = {"AF3": "Fp1", "AF4": "Fp2", "Fp1": "AF3", "Fp2": "AF4"}

    def _loc_from_info(ch):
        try:
            i = raw_for_positions.ch_names.index(ch)
            loc = np.asarray(raw_for_positions.info['chs'][i]['loc'][:3], float)
            if np.isfinite(loc).all() and not np.allclose(loc, 0):
                return loc
        except Exception:
            pass
        return None

    def _resolve_pos(ch):
        # 1) actual attached montage
        if ch in pos_mont and pos_mont[ch] is not None:
            return np.asarray(pos_mont[ch][:3], float)
        # 2) requested montage
        if ch in pos_main and pos_main[ch] is not None:
            return np.asarray(pos_main[ch][:3], float)
        # 3) dense & coarse fallbacks
        for posd in (pos_1005, pos_1020):
            if ch in posd and posd[ch] is not None:
                return np.asarray(posd[ch][:3], float)
        # 4) alias (AF3↔Fp1, AF4↔Fp2)
        if ch in alias:
            alt = alias[ch]
            for posd in (pos_mont, pos_main, pos_1005, pos_1020):
                if alt in posd and posd[alt] is not None:
                    return np.asarray(posd[alt][:3], float)
        # 5) last resort: channel loc[] in Info
        loc = _loc_from_info(ch)
        if loc is not None:
            return loc
        raise KeyError(f"No 3D position for channel '{ch}'.")

    # ---- positions -> adjacency/graph/wavelets ----
    ch_pos = np.vstack([_resolve_pos(ch) for ch in used_ch_names])
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(ch_pos)
    A = np.exp(-D**2 / np.mean(D**2)); np.fill_diagonal(A, 0)

    if graphs is None or gsp_filters is None:
        raise ImportError("pygsp not available. pip install pygsp")

    G = graphs.Graph(A)
    G.compute_laplacian(); G.compute_fourier_basis()

    deg = A.sum(axis=1); deg /= (deg.max() + 1e-12)
    num_wavelets = n_scales - 1
    s_vals = s_max / np.power(np.sqrt(2), np.arange(num_wavelets))
    meyer = gsp_filters.Meyer(G, Nf=n_scales, scales=s_vals)

    # build a montage from the resolved positions (used later by topomap)
    ch_pos_dict = {name: pos for name, pos in zip(used_ch_names, ch_pos)}
    custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos_dict, coord_frame="head")
    info_subset = create_info(ch_names=used_ch_names,
                              sfreq=raw_for_positions.info['sfreq'],
                              ch_types=['eeg'] * len(used_ch_names))
    info_subset.set_montage(custom_montage, on_missing="ignore")

    return G, meyer, s_vals, deg, info_subset
