#!/usr/bin/env python3
"""
Quick check: overlay bbox_labels.json onto circular-synth.png and save cropped view.

This reads DeepDRR output folders (each containing deepdrr/circular-synth.png and bbox_labels.json),
draws the bbox, and writes a crop and overlay for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def load_bbox(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bbox = data.get("bbox")
    if not bbox:
        return None
    return int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"])


def main():
    ap = argparse.ArgumentParser(description="Overlay and crop circular-synth.png using bbox_labels.json.")
    ap.add_argument("--runs", nargs="+", required=True, help="Paths to DeepDRR run directories (contain deepdrr/circular-synth.png and bbox_labels.json).")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/circular_crop_preview"), help="Where to write overlays and crops.")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for run in args.runs:
        run_dir = Path(run)
        circ_path = run_dir / "deepdrr" / "circular-synth.png"
        bbox_path = run_dir / "bbox_labels.json"
        if not circ_path.exists() or not bbox_path.exists():
            continue
        bbox = load_bbox(bbox_path)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        img = cv2.imread(str(circ_path))
        if img is None:
            continue

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        crop = img[y0 : y1 + 1, x0 : x1 + 1]

        stem = run_dir.name  # e.g., off_p0p0
        prefix = run_dir.parent.name  # e.g., HN_P001
        base = f"{prefix}_{stem}"
        cv2.imwrite(str(out_dir / f"{base}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{base}_crop.png"), crop)
        processed += 1

    print(f"Wrote {processed} overlays/crops to {out_dir}")


if __name__ == "__main__":
    main()
