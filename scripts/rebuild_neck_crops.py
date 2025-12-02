#!/usr/bin/env python3
"""
Rebuild neck-cropped, letterboxed images and bitmasks from circular views.

For each DeepDRR sample, this script:
- loads the circular image (default circular_synth_clahe2.png) and the circular vertebra bitmask
  (default label_bitmask_circular_synth.png),
- finds the tight bounding box of all vertebra labels (nonzero mask),
  applies a small padding,
  crops both image and mask,
  letterboxes to a target size (default 1024),
- writes the new cropped/letterboxed clahe2 + bitmask to a parallel output root,
  preserving the directory structure (HN_Pxxx/off_xxx/deepdrr/...).

Use this to regenerate clean neck crops after filtering bad vertebra masks/boxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def find_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def letterbox(img: np.ndarray, size: int, interp: int = cv2.INTER_AREA) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size), dtype=img.dtype)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild neck crops from circular images and masks.")
    ap.add_argument("--input-root", type=Path, required=True, help="Root with circular images/masks (e.g., outputs/dataset_synth_headneck_2).")
    ap.add_argument("--output-root", type=Path, required=True, help="Destination root (e.g., outputs/dataset_synth_headneck_3).")
    ap.add_argument("--image-name", default="circular-synth.png", help="Circular image filename to crop.")
    ap.add_argument("--mask-name", default="label_bitmask_circular_synth.png", help="Circular mask filename to crop.")
    ap.add_argument("--out-image-name", default="clahe2.png", help="Output cropped image filename.")
    ap.add_argument("--out-mask-name", default="label_bitmask_cropped_letterboxed.png", help="Output cropped mask filename.")
    ap.add_argument("--target-size", type=int, default=1024, help="Letterbox size.")
    ap.add_argument("--pad-frac", type=float, default=0.05, help="Padding fraction relative to bbox max side.")
    ap.add_argument("--apply-clahe2", action="store_true", help="Apply double CLAHE to the cropped image before saving.")
    ap.add_argument("--clip-limit1", type=float, default=2.0)
    ap.add_argument("--clip-limit2", type=float, default=2.0)
    ap.add_argument("--tile-size1", type=int, default=8)
    ap.add_argument("--tile-size2", type=int, default=8)
    args = ap.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    count = 0
    skipped = 0

    mask_paths = list(input_root.rglob(args.mask_name))
    if not mask_paths:
        print(f"No masks found under {input_root} matching {args.mask_name}")
        return
    for mask_path in mask_paths:
        img_path = mask_path.parent / args.image_name
        if not img_path.exists():
            skipped += 1
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or img is None:
            skipped += 1
            continue
        bbox = find_bbox(mask)
        if bbox is None:
            skipped += 1
            continue
        x0, y0, x1, y1 = bbox
        side = max(x1 - x0 + 1, y1 - y0 + 1)
        pad = int(round(side * args.pad_frac))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(mask.shape[1] - 1, x1 + pad)
        y1 = min(mask.shape[0] - 1, y1 + pad)

        crop_img = img[y0 : y1 + 1, x0 : x1 + 1]
        crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

        if args.apply_clahe2:
            clahe1 = cv2.createCLAHE(clipLimit=args.clip_limit1, tileGridSize=(args.tile_size1, args.tile_size1))
            clahe2 = cv2.createCLAHE(clipLimit=args.clip_limit2, tileGridSize=(args.tile_size2, args.tile_size2))
            crop_img = clahe2.apply(clahe1.apply(crop_img))

        lb_img, _, _, _ = letterbox(crop_img, args.target_size, interp=cv2.INTER_AREA)
        lb_mask, _, _, _ = letterbox(crop_mask, args.target_size, interp=cv2.INTER_NEAREST)

        # destination path mirrors input relative to input_root
        rel = mask_path.parent.relative_to(input_root)
        out_dir = output_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / args.out_image_name), lb_img)
        cv2.imwrite(str(out_dir / args.out_mask_name), lb_mask)
        count += 1

    print(f"Done. Wrote {count} samples. Skipped {skipped}. Output: {output_root}")


if __name__ == "__main__":
    main()
