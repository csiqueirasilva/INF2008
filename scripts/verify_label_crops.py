#!/usr/bin/env python3
"""
Quick sanity check: apply YOLO bboxes back onto the mask images and save crops
to a new directory for visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_bbox(lbl_path: Path) -> tuple[int, int, int, int] | None:
    if not lbl_path.exists():
        return None
    line = lbl_path.read_text().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 5:
        return None
    _, cx, cy, w, h = map(float, parts)
    return cx, cy, w, h  # type: ignore


def to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[int, int, int, int]:
    x0 = int((cx - w / 2) * W)
    y0 = int((cy - h / 2) * H)
    x1 = int((cx + w / 2) * W)
    y1 = int((cy + h / 2) * H)
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser(description="Crop mask images using saved YOLO labels to verify bbox correctness.")
    ap.add_argument("--image-dir", type=Path, default=Path("prepared/neck_bbox_yolo/train/images"))
    ap.add_argument("--label-dir", type=Path, default=Path("prepared/neck_bbox_yolo/train/labels"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/label_crop_check"))
    ap.add_argument("--source-dir", type=Path, default=None, help="Optional original mask dir to crop from instead of --image-dir.")
    ap.add_argument("--strip-prefix", default="", help="Prefix to strip from label stems when looking up source files (e.g., real_).")
    ap.add_argument("--source-suffix", default="", help="Suffix to append before extension when loading source images (e.g., _mask).")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of samples (0 = all).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(list(args.label_dir.glob("*.txt")))
    if args.limit > 0:
        labels = labels[: args.limit]

    count = 0
    for lbl_path in labels:
        stem = lbl_path.stem
        source_stem = stem[len(args.strip_prefix) :] if args.strip_prefix and stem.startswith(args.strip_prefix) else stem

        img_path = None
        search_dir = args.source_dir if args.source_dir else args.image_dir
        for ext in (".png", ".jpg"):
            candidate = search_dir / f"{source_stem}{args.source_suffix}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        lbl = load_bbox(lbl_path)
        if lbl is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        cx, cy, w, h = lbl
        x0, y0, x1, y1 = to_xyxy(cx, cy, w, h, img.shape[1], img.shape[0])
        crop = img[y0 : y1 + 1, x0 : x1 + 1]
        cv2.imwrite(str(args.out_dir / f"{stem}_crop.png"), crop)

        # Also save a quick overlay for spot checks.
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.imwrite(str(args.out_dir / f"{stem}_overlay.png"), overlay)
        count += 1

    print(f"Saved {count} crops/overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
