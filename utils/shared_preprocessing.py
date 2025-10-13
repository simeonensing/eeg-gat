from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import mne
from mne.preprocessing import compute_current_source_density


def prep_raw(path: str, montage: str, keep_ch: List[str]) -> mne.io.BaseRaw:
    if not Path(path).exists(): raise FileNotFoundError(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    want = [ch for ch in keep_ch if ch in raw.ch_names]
    raw.pick(picks=want)
    raw.set_montage(montage, match_case=False)
    mne.set_eeg_reference(raw, 'average', projection=False, verbose="ERROR")
    raw.filter(l_freq=1.0, h_freq=45.0, method='fir', fir_window='hamming', phase='zero', verbose="ERROR")
    raw = compute_current_source_density(raw)
    return raw


def align_channels(raw_list: List[mne.io.BaseRaw], order_pref: List[str]) -> Tuple[List[mne.io.BaseRaw], List[str]]:
    common = set(raw_list[0].ch_names)
    for r in raw_list[1:]: common &= set(r.ch_names)
    common = [ch for ch in order_pref if ch in common]
    if len(common) < 4: raise RuntimeError(f"Too few common channels: {common}")
    out = []
    for r in raw_list:
        r2 = r.copy().pick(common).reorder_channels(common)
        out.append(r2)
    return out, common


def windows_for_length(n_samples, sfreq, win_sec, step_sec=0, include_tail=True):
    W = int(round(win_sec * sfreq))
    S = int(round(step_sec * sfreq))
    if W < 1: raise ValueError("win_sec too small.")
    if S <= 0: S = W
    stop = max(0, n_samples - W + 1)
    idx = [(s, s + W) for s in range(0, stop, S)]
    if include_tail and (n_samples >= W) and (not idx or idx[-1][1] < n_samples):
        s = n_samples - W
        if not idx or idx[-1][0] != s: idx.append((s, s + W))
    return idx
