#!/usr/bin/env python3
"""
Preprocess VFSS frames to match the YOLO training style (CLAHE2 + blur + Otsu overlay).

This keeps the original resolution and saves both the CLAHE2 image and the Otsu overlay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def double_clahe(img: np.ndarray, clip1: float, tile1: int, clip2: float, tile2: int):
    clahe1 = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(tile1, tile1)).apply(img)
    clahe2 = cv2.createCLAHE(clipLimit=clip2, tileGridSize=(tile2, tile2)).apply(clahe1)
    return clahe1, clahe2


def otsu_mask(img: np.ndarray, blur_k: int) -> np.ndarray:
    if blur_k <= 0 or blur_k % 2 == 0:
        blur_k = 1
    blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def overlay_mask(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if base.ndim == 2:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base
    color_mask = np.zeros_like(base_bgr)
    color_mask[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(base_bgr, 0.7, color_mask, 0.3, 0)


def main():
    ap = argparse.ArgumentParser(description="Preprocess frames for YOLO neck bbox detection.")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory with raw frames (.png/.jpg)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for preprocessed images")
    ap.add_argument(
        "--output-kind",
        choices=["mask", "overlay", "clahe2", "all"],
        default="mask",
        help="What to save: binary mask, overlay, clahe2, or all three (with suffixes).",
    )
    ap.add_argument("--clip-limit1", type=float, default=2.0)
    ap.add_argument("--clip-limit2", type=float, default=2.0)
    ap.add_argument("--tile-size1", type=int, default=8)
    ap.add_argument("--tile-size2", type=int, default=8)
    ap.add_argument("--blur-kernel", type=int, default=5)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(list(args.input_dir.glob("*.png")) + list(args.input_dir.glob("*.jpg"))):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        clahe1, clahe2 = double_clahe(img, args.clip_limit1, args.tile_size1, args.clip_limit2, args.tile_size2)
        mask = otsu_mask(clahe2, args.blur_kernel)
        overlay = overlay_mask(clahe2, mask)

        stem = img_path.stem
        if args.output_kind in ("mask", "all"):
            cv2.imwrite(str(args.out_dir / f"{stem}_mask.png"), mask)
        if args.output_kind in ("overlay", "all"):
            cv2.imwrite(str(args.out_dir / f"{stem}_overlay.png"), overlay)
        if args.output_kind in ("clahe2", "all"):
            cv2.imwrite(str(args.out_dir / f"{stem}_clahe2.png"), clahe2)
        count += 1

    print(f"Processed {count} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
