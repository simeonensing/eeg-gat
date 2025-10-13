from __future__ import annotations
from typing import Callable, List, Tuple

import numpy as np

from config import CFG
from utils.utils import valid_split, pair_ids_from_rows



def contiguous_purged_folds(
        block_ids_rows: np.ndarray, y_rows: np.ndarray, n_splits: int, embargo_blocks: int,
        validator: Callable[[np.ndarray, np.ndarray], bool]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    blocks = np.unique(block_ids_rows)
    K = len(blocks)
    if n_splits > K: return []
    sizes = [K // n_splits + (1 if i < K % n_splits else 0) for i in range(n_splits)]
    starts = np.cumsum([0] + sizes[:-1])
    folds = []
    for st, sz in zip(starts, sizes):
        val_blocks = blocks[st:st + sz]
        left = max(0, st - embargo_blocks)
        right = min(K, st + sz + embargo_blocks)
        train_blocks = np.concatenate([blocks[:left], blocks[right:]]) if left < right else blocks[:0]
        va_idx = np.flatnonzero(np.isin(block_ids_rows, val_blocks))
        tr_idx = np.flatnonzero(np.isin(block_ids_rows, train_blocks))
        if validator(y_rows[tr_idx], y_rows[va_idx]): folds.append((tr_idx, va_idx))
    return folds


def plan_outer(y_rows: np.ndarray, n_pairs: int, block_sizes_pairs: List[int],
               embargoes: List[int], K_target: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int, int]:
    best = ([], 0, None, None)
    for bs in block_sizes_pairs:
        pair_blocks = np.repeat(np.arange((n_pairs + bs - 1) // bs), bs)[:n_pairs]
        blocks_rows = np.repeat(pair_blocks, 2)
        K = len(np.unique(blocks_rows))
        for E in embargoes:
            k_try = min(K_target, K)
            if k_try < 2: continue
            folds = contiguous_purged_folds(
                blocks_rows, y_rows, n_splits=k_try, embargo_blocks=E,
                validator=lambda ytr, yva: valid_split(
                    ytr, yva,
                    min_train=CFG.cv.min_train_rows, min_val=CFG.cv.min_val_rows,
                    min_pos=CFG.cv.min_pos_per_split, min_neg=CFG.cv.min_neg_per_split, ratio_lo=CFG.cv.ratio_lo
                )
            )
            if len(folds) == k_try: return folds, k_try, bs, E
            if len(folds) > best[1]: best = (folds, len(folds), bs, E)
    return best


def plan_inner(tr_idx_full: np.ndarray, y_all: np.ndarray, used_bs_pairs_outer: int) -> List[
    Tuple[np.ndarray, np.ndarray]]:
    pair_ids_train = pair_ids_from_rows(tr_idx_full)
    uniq_pairs = np.unique(pair_ids_train)
    n_pairs_tr = len(uniq_pairs)
    if n_pairs_tr < 4: return []
    bs_cands = sorted({used_bs_pairs_outer, max(4, used_bs_pairs_outer // 2), 6, 5, 4}, reverse=True)
    E_cands = [CFG.cv.embargo_blocks_inner, max(0, CFG.cv.embargo_blocks_inner - 1), 0]
    best_local: List[Tuple[np.ndarray, np.ndarray]] = []
    best_k = 0

    for bs in bs_cands:
        pair_blocks = np.repeat(np.arange((n_pairs_tr + bs - 1) // bs), bs)[:n_pairs_tr]
        map_pair = {pid: bid for pid, bid in zip(uniq_pairs, pair_blocks)}
        blocks_rows = np.array([map_pair[pid] for pid in pair_ids_train], int)
        K = len(np.unique(blocks_rows))
        for E in E_cands:
            for k_try in [CFG.cv.inner_folds_target, CFG.cv.inner_folds_target - 1, 2]:
                if k_try < 2 or k_try > K: continue
                folds = contiguous_purged_folds(
                    blocks_rows, y_all[tr_idx_full], n_splits=k_try, embargo_blocks=E,
                    validator=lambda ytr, yva: valid_split(
                        ytr, yva, min_train=CFG.cv.min_train_rows_inner, min_val=CFG.cv.min_val_rows_inner,
                        min_pos=CFG.cv.min_pos_per_split_inner, min_neg=CFG.cv.min_neg_per_split_inner,
                        ratio_lo=CFG.cv.ratio_lo_inner
                    )
                )
                if len(folds) == k_try:
                    return [(tr_idx_full[tr], tr_idx_full[va]) for tr, va in folds]
                if len(folds) > best_k:
                    best_k = len(folds)
                    best_local = [(tr_idx_full[tr], tr_idx_full[va]) for tr, va in folds]
    return best_local
