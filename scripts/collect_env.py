#!/usr/bin/env python3
"""
Collect environment + config snapshot for reproducibility.
Outputs: reproducibility/environment.json
"""
import json, os, platform, subprocess, sys
from pathlib import Path

# Try to import your config to embed CFG + paths
try:
    from config import CFG, PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CFG = None

out_dir = PROJECT_ROOT / "reproducibility"
out_dir.mkdir(parents=True, exist_ok=True)

def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return ""

info = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "git_commit": sh("git rev-parse HEAD"),
    "git_status": sh("git status --porcelain"),
    "pip_freeze": sh("pip freeze"),
}

if CFG:
    info["mlflow_tracking_uri"] = getattr(CFG.mlflow, "tracking_uri", None)
    info["optuna_storage_url"] = getattr(CFG.optuna, "storage_url", None)
    info["tb_log_dir"] = getattr(CFG.tensorboard, "log_dir", None)
    info["cfg_snapshot"] = CFG.to_dict()

try:
    import torch
    info["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name_0": torch.cuda.get_device_name(0) if torch.cuda.device_count() else "",
    }
except Exception:
    pass

out_file = out_dir / "environment.json"
out_file.write_text(json.dumps(info, indent=2))
print(f"[✓] Saved reproducibility snapshot → {out_file}")
