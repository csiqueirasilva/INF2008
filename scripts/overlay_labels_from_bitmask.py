#!/usr/bin/env python3
"""
Overlay labels onto circular-synth.png (and clahe2.png if present) using label_bitmask.png.

Expects DeepDRR output folder with:
- deepdrr/circular-synth.png
- deepdrr/clahe2.png (optional; if missing, only circular is processed)
- deepdrr/label_bitmask.png (uint16 with bits per label index)

Outputs overlays to <run>/deepdrr/label_overlay_circular.png and label_overlay_clahe2.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spine_segmentation.core.label_colors import label_to_color


def overlay_from_bitmask(base: np.ndarray, bitmask: np.ndarray, max_bits: int = 16, alpha: float = 0.45) -> np.ndarray:
    if base.ndim == 2:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base
    color = np.zeros_like(base_bgr)
    for bit in range(max_bits):
        mask = (bitmask & (1 << bit)) != 0
        if not np.any(mask):
            continue
        color[mask] = label_to_color(bit + 1)  # label ids are 1-based in our palette
    return cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0.0)


def process_run(run_dir: Path, out_root: Path | None = None):
    deepdrr_dir = run_dir / "deepdrr"
    bitmask_circ_path = deepdrr_dir / "label_bitmask_circular_synth.png"
    bitmask_cropped_path = deepdrr_dir / "label_bitmask_cropped_letterboxed.png"
    circ_path = deepdrr_dir / "circular-synth.png"
    clahe_path = deepdrr_dir / "clahe2.png"

    if not circ_path.exists():
        return False

    if bitmask_circ_path.exists():
        bitmask_circ = cv2.imread(str(bitmask_circ_path), cv2.IMREAD_UNCHANGED)
        circ = cv2.imread(str(circ_path), cv2.IMREAD_GRAYSCALE)
        if bitmask_circ is not None and circ is not None:
            circ_overlay = overlay_from_bitmask(circ, bitmask_circ)
            target = (out_root if out_root else deepdrr_dir) / "label_overlay_circular.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), circ_overlay)

    if clahe_path.exists() and bitmask_cropped_path.exists():
        bitmask_crop = cv2.imread(str(bitmask_cropped_path), cv2.IMREAD_UNCHANGED)
        clahe = cv2.imread(str(clahe_path), cv2.IMREAD_GRAYSCALE)
        if bitmask_crop is not None and clahe is not None:
            clahe_overlay = overlay_from_bitmask(clahe, bitmask_crop)
            target = (out_root if out_root else deepdrr_dir) / "label_overlay_clahe2.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), clahe_overlay)
    return True


def main():
    ap = argparse.ArgumentParser(description="Overlay label_bitmask onto circular-synth.png and clahe2.png.")
    ap.add_argument("--runs", nargs="+", required=True, help="DeepDRR run directories (each contains deepdrr/...)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Optional root to write overlays (defaults to each run's deepdrr).")
    args = ap.parse_args()

    ok = 0
    for run in args.runs:
        if process_run(Path(run), args.out_dir):
            ok += 1
    print(f"Processed {ok} runs")


if __name__ == "__main__":
    main()
