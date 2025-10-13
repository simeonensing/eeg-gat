# config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Any
import json
import os

# Optional YAML override (safe to remove if you don't want YAML)
try:
    import yaml
except Exception:
    yaml = None

import numpy as np


# =========================
# High-level: what to edit
# =========================
# - Paths under DataConfig (point to your EDFs)
# - Channels & montage
# - Window sizes (window_grid)
# - CV targets (outer/inner folds)
# - Training hyperparams
# - Split feasibility thresholds
# - Tracking URIs / flags


@dataclass
class DataConfig:
    # Absolute or relative paths to your EEG files
    pre_active_path: str = "dataset/baseline.edf"
    post_sham_path: str = "dataset/sham.edf"
    post_active_path: str = "dataset/active.edf"

    montage_name: str = "standard_1020"
    keep_channels: List[str] = field(default_factory=lambda: [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ])

    # Where all CSVs/plots land
    save_dir: str = "results"

    def ensure_dirs(self) -> None:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_dir, "figures").mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralConfig:
    # Canonical EEG bands
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "delta": (1, 3),
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta":  (13, 30),
        "gamma": (31, 45),
    })
    band_order: List[str] = field(default_factory=lambda: ["delta", "theta", "alpha", "beta", "gamma"])

    # Morlet TFR grid
    all_freqs: np.ndarray = field(default_factory=lambda: np.linspace(1, 45, 90))
    n_cycles: np.ndarray = field(default_factory=lambda: np.linspace(1, 45, 90) / 2.0)

    # Graph Wavelet Transform (Meyer)
    n_scales: int = 5
    s_max: float = 0.35

    # Sliding windows (seconds). Add more if you want multi-window evaluation.
    window_grid: list[int] = field(default_factory=lambda: [4])

    def to_dict(self) -> Dict[str, Any]:
        # Convert numpy arrays to lists for logging
        d = asdict(self)
        d["all_freqs"] = list(map(float, d["all_freqs"]))
        d["n_cycles"] = list(map(float, d["n_cycles"]))
        return d


@dataclass
class CVConfig:
    # Targets; planner will degrade gracefully if infeasible
    outer_folds_target: int = 5
    inner_folds_target: int = 3
    random_seed: int = 42

    # Shared split “style” parameters
    block_size_pairs_default: int = 8
    embargo_blocks_outer_default: int = 2
    purge_pairs: int = 2  # hard-purge around validation blocks

    # Feasibility thresholds (outer)
    min_pos_per_split: int = 12
    min_neg_per_split: int = 12
    min_train_rows: int = 100
    min_val_rows: int = 50
    ratio_lo: float = 0.75   # min class-balance ratio in each split

    # Feasibility thresholds (inner; slightly looser)
    embargo_blocks_inner: int = 2
    min_pos_per_split_inner: int = 10
    min_neg_per_split_inner: int = 10
    min_train_rows_inner: int = 80
    min_val_rows_inner: int = 40
    ratio_lo_inner: float = 0.70

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    # Global optimization hyperparams for GAT training
    max_epochs: int = 120
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 20
    # auto: "cuda" if available else "cpu"; you may also force "cpu"
    device: str = "auto"

    # Ensemble & selection behavior
    outer_seeds: int = 3            # number of models to ensemble per outer fold
    avg_two_seeds_inner: bool = True
    use_ap_selection: bool = True   # AP for inner selection; otherwise BCE

    # Label smoothing for stability
    label_smooth_eps: float = 0.05

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device"] = self.resolved_device()
        return d


@dataclass
class ToggleConfig:
    # Calibration, explainability, and classical-XAI behavior
    use_temp_scaling: bool = True
    explain_class: str = "active"  # or "sham"
    use_logit_deltas_classical: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackingConfig:
    # Turn on/off trackers
    enable_mlflow: bool = True
    enable_tensorboard: bool = True

    # MLflow
    mlflow_tracking_uri: str | None = None   # e.g., "http://localhost:5000" or leave None to use env
    mlflow_experiment: str = "eeg-gwtgat-vs-classical"
    mlflow_nested_runs: bool = True

    # TensorBoard
    tb_root: str = "tb"  # a directory under which runs will be timestamped

    # What to log as artifacts (relative to save_dir)
    log_artifact_dirnames: List[str] = field(default_factory=lambda: ["", "figures"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    toggles: ToggleConfig = field(default_factory=ToggleConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # --- Helpers ---
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. `pip install pyyaml` or remove YAML usage.")
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        # Allow partial trees; dataclasses handle defaults
        def merge(dc_cls, src):
            if src is None:
                return dc_cls()  # default
            if isinstance(src, dict):
                return dc_cls(**src)
            raise TypeError(f"Invalid section for {dc_cls.__name__}: {type(src)}")
        return cls(
            data=merge(DataConfig, raw.get("data")),
            spectral=merge(SpectralConfig, raw.get("spectral")),
            cv=merge(CVConfig, raw.get("cv")),
            train=merge(TrainConfig, raw.get("train")),
            toggles=merge(ToggleConfig, raw.get("toggles")),
            tracking=merge(TrackingConfig, raw.get("tracking")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data.to_dict(),
            "spectral": self.spectral.to_dict(),
            "cv": self.cv.to_dict(),
            "train": self.train.to_dict(),
            "toggles": self.toggles.to_dict(),
            "tracking": self.tracking.to_dict(),
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def ensure_dirs(self) -> None:
        self.data.ensure_dirs()

    # Convenience: numpy-ready attributes your code expects
    @property
    def bands(self) -> Dict[str, Tuple[float, float]]:
        return self.spectral.bands

    @property
    def band_order(self) -> List[str]:
        return self.spectral.band_order

    @property
    def all_freqs(self) -> np.ndarray:
        return np.asarray(self.spectral.all_freqs, dtype=float)

    @property
    def n_cycles(self) -> np.ndarray:
        return np.asarray(self.spectral.n_cycles, dtype=float)


# -------------
# Build CFG
# -------------
def load_cfg(yaml_path: str | None = None) -> ExperimentConfig:
    """
    1) If yaml_path is provided and exists -> load YAML.
    2) Else if ENV var XAI_CONFIG_YAML is set and exists -> load it.
    3) Else -> defaults.
    """
    if yaml_path and Path(yaml_path).exists():
        cfg = ExperimentConfig.from_yaml(yaml_path)
    else:
        env_yaml = os.environ.get("XAI_CONFIG_YAML")
        if env_yaml and Path(env_yaml).exists():
            cfg = ExperimentConfig.from_yaml(env_yaml)
        else:
            cfg = ExperimentConfig()

    # realize directories and normalize device
    cfg.ensure_dirs()
    # force resolution of device now (so downstream logging sees the concrete value)
    _ = cfg.train.resolved_device()
    return cfg


# Public singleton you can import everywhere:
CFG: ExperimentConfig = load_cfg()
