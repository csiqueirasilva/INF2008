#!/usr/bin/env python3
"""
Export full-frame mask images to a YOLO-style detection dataset (single class: neck).

This assumes you already preprocessed frames to masks (e.g., CLAHE2+blur+Otsu)
and want to train on the full view, not cropped panels. We simply read each mask,
compute the bounding box of nonzero pixels, and write images + YOLO txt labels.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def norm_yolo(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[float, float, float, float]:
    cx = (x0 + x1 + 1) / 2.0 / w
    cy = (y0 + y1 + 1) / 2.0 / h
    bw = (x1 - x0 + 1) / w
    bh = (y1 - y0 + 1) / h
    return cx, cy, bw, bh


def list_images(dir_path: Path) -> list[Path]:
    return sorted(list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.jpeg")))


def main():
    ap = argparse.ArgumentParser(description="Export full-frame mask images to YOLO format (single class).")
    ap.add_argument("--mask-dir", type=Path, required=True, help="Directory with full-frame mask images (.png/.jpg).")
    ap.add_argument("--output-root", type=Path, default=Path("prepared/neck_bbox_yolo_full"), help="Output dataset root.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data for val split.")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling before split.")
    ap.add_argument("--prefix", default="realfull_", help="Prefix for output filenames to avoid collisions.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of masks (0 = all).")
    args = ap.parse_args()

    images = list_images(args.mask_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.mask_dir}")

    samples: list[tuple[Path, tuple[int, int, int, int]]] = []
    for img_path in images:
        mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue
        samples.append((img_path, bbox))

    if not samples:
        raise SystemExit("No masks with foreground pixels found.")

    random.seed(args.seed)
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - args.val_ratio))
    splits = [("train", samples[:split_idx]), ("val", samples[split_idx:])]

    for split_name, rows in splits:
        img_dir = args.output_root / split_name / "images"
        lbl_dir = args.output_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, bbox) in enumerate(rows):
            stem = f"{args.prefix}{img_path.stem}"
            dest_img = img_dir / f"{stem}.png"
            dest_lbl = lbl_dir / f"{stem}.txt"

            mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape[:2]
            x0, y0, x1, y1 = bbox
            cx, cy, bw, bh = norm_yolo(x0, y0, x1, y1, w, h)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(dest_img), mask_bgr)
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Wrote YOLO dataset to {args.output_root} (train={len(splits[0][1])}, val={len(splits[1][1])})")


if __name__ == "__main__":
    main()
