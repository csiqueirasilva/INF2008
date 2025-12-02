#!/usr/bin/env python3
"""
Draw YOLO txt labels onto images for quick inspection.
Assumes labels directory mirrors images by stem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def load_bbox(lbl_path: Path, w: int, h: int):
    if not lbl_path.exists():
        return None
    line = lbl_path.read_text().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh = map(float, parts[:5])
    x0 = int((cx - bw / 2) * w)
    y0 = int((cy - bh / 2) * h)
    x1 = int((cx + bw / 2) * w)
    y1 = int((cy + bh / 2) * h)
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser(description="Overlay YOLO labels on images.")
    ap.add_argument("--image-dir", type=Path, required=True, help="Directory with images.")
    ap.add_argument("--label-dir", type=Path, required=True, help="Directory with YOLO txt labels.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to save overlays.")
    ap.add_argument("--pattern", default="*.png", help="Glob pattern for images.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.image_dir.glob(args.pattern))
    count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = args.label_dir / f"{img_path.stem}.txt"
        bbox = load_bbox(lbl_path, w, h)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(overlay, "neck", (x0, max(10, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"{img_path.stem}_overlay.png"), overlay)
        count += 1
    print(f"Wrote {count} overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
