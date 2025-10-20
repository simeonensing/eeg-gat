#!/usr/bin/env python3
"""Generate SHA256 checksums for dataset files."""
import hashlib
from pathlib import Path

root = Path("dataset")
out = Path("reproducibility/checksums.txt")
out.parent.mkdir(exist_ok=True)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

lines = []
for p in sorted(root.glob("**/*")):
    if p.is_file():
        lines.append(f"{sha256(p)}  {p.relative_to(root)}")
out.write_text("\n".join(lines) + "\n")
print(f"[✓] Checksums written → {out}")
