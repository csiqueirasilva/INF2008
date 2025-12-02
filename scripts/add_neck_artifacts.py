#!/usr/bin/env python3
"""
Post-process DeepDRR outputs to add convenience artifacts:
- GT bbox overlay (C2–C4) on the clahe2 image
- Neck crop without letterbox (from circular-synth.png)
- Per-vertebra crops (C2–C4) without letterbox

Usage:
  poetry run python scripts/add_neck_artifacts.py \
    --root outputs/dataset_synth_headneck_4 \
    --pad-frac 0.05 --vertebra-pad-frac 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

CLS = [1, 2, 4]  # C2, C3, C4 bitmask ids
CLS_NAME = {1: "C2", 2: "C3", 4: "C4"}
COLORS = {
    1: (0, 255, 0),
    2: (0, 200, 255),
    4: (255, 128, 0),
}


def find_bbox(mask: np.ndarray, lid: int | None = None) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask if lid is None else (mask == lid))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pad_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int, pad_frac: float) -> tuple[int, int, int, int]:
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    pad = int(max(bw, bh) * pad_frac)
    return max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad)


def square_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int, pad_frac: float) -> tuple[int, int, int, int]:
    """Make bbox square by expanding the shorter side, then pad."""
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    side = max(bw, bh)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = side // 2
    sx0 = max(0, cx - half)
    sy0 = max(0, cy - half)
    sx1 = min(w - 1, sx0 + side - 1)
    sy1 = min(h - 1, sy0 + side - 1)
    # adjust if we clipped at borders
    sx0 = max(0, sx1 - side + 1)
    sy0 = max(0, sy1 - side + 1)
    return pad_bbox(sx0, sy0, sx1, sy1, w, h, pad_frac)


def iter_masks(root: Path, name: str) -> Iterable[Path]:
    yield from root.rglob(name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Add bbox overlays and crops to DeepDRR outputs.")
    ap.add_argument("--root", type=Path, required=True, help="Dataset root, e.g., outputs/dataset_synth_headneck_4")
    ap.add_argument("--mask-name", default="label_bitmask_circular_synth.png", help="Mask filename to read.")
    ap.add_argument("--img-name", default="circular-synth.png", help="Image to crop (non-letterboxed).")
    ap.add_argument("--clahe-name", default="clahe2.png", help="Image to draw overlay on (letterboxed OK).")
    ap.add_argument("--pad-frac", type=float, default=0.05, help="Padding for neck crop bbox.")
    ap.add_argument("--vertebra-pad-frac", type=float, default=0.1, help="Padding for per-vertebra crops.")
    ap.add_argument("--save-hires-overlay", action="store_true", help="Also save a high-res overlay on the circular image.")
    ap.add_argument("--hires-alpha", type=float, default=0.45, help="Alpha for high-res overlay blend.")
    ap.add_argument("--square-bbox", action="store_true", help="Make neck and per-vertebra bboxes square (expand shorter side).")
    ap.add_argument("--save-bbox-mask-overlay", action="store_true", help="Save overlay of mask+bounding boxes on clahe image.")
    ap.add_argument("--bbox-mask-alpha", type=float, default=0.35, help="Alpha for mask in bbox+mask overlay.")
    ap.add_argument("--save-per-class", action="store_true", help="Save per-class mask+box overlays separately.")
    args = ap.parse_args()

    count = 0
    for mask_path in iter_masks(args.root, args.mask_name):
        deepdrr_dir = mask_path.parent
        img_path = deepdrr_dir / args.img_name
        clahe_path = deepdrr_dir / args.clahe_name
        if not img_path.exists() or not clahe_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        circ = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        clahe = cv2.imread(str(clahe_path), cv2.IMREAD_GRAYSCALE)
        if circ is None or clahe is None:
            continue
        h, w = circ.shape[:2]

        # Neck crop without letterbox
        neck_box = find_bbox(mask)
        if neck_box:
            if args.square_bbox:
                nx0, ny0, nx1, ny1 = square_bbox(*neck_box, w=w, h=h, pad_frac=args.pad_frac)
            else:
                nx0, ny0, nx1, ny1 = pad_bbox(*neck_box, w=w, h=h, pad_frac=args.pad_frac)
            neck_crop = circ[ny0:ny1 + 1, nx0:nx1 + 1]
            cv2.imwrite(str(deepdrr_dir / "neck_crop_nolb.png"), neck_crop)

        # GT bbox overlay on clahe2
        ov = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
        for lid in CLS:
            box = find_bbox(mask, lid)
            if not box:
                continue
            if args.square_bbox:
                x0, y0, x1, y1 = square_bbox(*box, w=w, h=h, pad_frac=0.0)
            else:
                x0, y0, x1, y1 = box
            cv2.rectangle(ov, (x0, y0), (x1, y1), COLORS[lid], 2)
            cv2.putText(ov, CLS_NAME[lid], (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[lid], 2, cv2.LINE_AA)
        cv2.imwrite(str(deepdrr_dir / "bbox_overlay_gt.png"), ov)

        # Bbox + mask overlay on clahe (optional)
        if args.save_bbox_mask_overlay:
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for lid in CLS:
                color_mask[mask == lid] = COLORS[lid]
            clahe_rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
            mix = cv2.addWeighted(clahe_rgb, 1 - args.bbox_mask_alpha, color_mask, args.bbox_mask_alpha, 0)
            # redraw boxes on top
            for lid in CLS:
                box = find_bbox(mask, lid)
                if not box:
                    continue
                if args.square_bbox:
                    x0, y0, x1, y1 = square_bbox(*box, w=w, h=h, pad_frac=0.0)
                else:
                    x0, y0, x1, y1 = box
                cv2.rectangle(mix, (x0, y0), (x1, y1), COLORS[lid], 2)
                cv2.putText(mix, CLS_NAME[lid], (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[lid], 2, cv2.LINE_AA)
            cv2.imwrite(str(deepdrr_dir / "bbox_mask_overlay.png"), mix)

            # Per-class overlays
            for lid in CLS:
                box = find_bbox(mask, lid)
                if not box:
                    continue
                if args.square_bbox:
                    x0, y0, x1, y1 = square_bbox(*box, w=w, h=h, pad_frac=0.0)
                else:
                    x0, y0, x1, y1 = box
                single = np.zeros_like(clahe_rgb)
                single[mask == lid] = COLORS[lid]
                ov_single = cv2.addWeighted(clahe_rgb, 1 - args.bbox_mask_alpha, single, args.bbox_mask_alpha, 0)
                cv2.rectangle(ov_single, (x0, y0), (x1, y1), COLORS[lid], 2)
                cv2.putText(ov_single, CLS_NAME[lid], (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[lid], 2, cv2.LINE_AA)
                cv2.imwrite(str(deepdrr_dir / f"bbox_mask_overlay_{CLS_NAME[lid]}.png"), ov_single)

        # High-res overlay on circular image (no letterbox) if requested
        if args.save_hires_overlay:
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for lid in CLS:
                color = COLORS[lid]
                color_mask[mask == lid] = color
            circ_rgb = cv2.cvtColor(circ, cv2.COLOR_GRAY2BGR)
            hires = cv2.addWeighted(circ_rgb, 1 - args.hires_alpha, color_mask, args.hires_alpha, 0)
            cv2.imwrite(str(deepdrr_dir / "overlay_hires.png"), hires)

        # Per-vertebra crops without letterbox
        crops_dir = deepdrr_dir / "vertebra_crops"
        crops_dir.mkdir(exist_ok=True)
        for lid in CLS:
            box = find_bbox(mask, lid)
            if not box:
                continue
            if args.square_bbox:
                x0, y0, x1, y1 = square_bbox(*box, w=w, h=h, pad_frac=args.vertebra_pad_frac)
            else:
                x0, y0, x1, y1 = pad_bbox(*box, w=w, h=h, pad_frac=args.vertebra_pad_frac)
            crop_img = circ[y0:y1 + 1, x0:x1 + 1]
            cv2.imwrite(str(crops_dir / f"{CLS_NAME[lid]}_crop.png"), crop_img)

        count += 1

    print(f"Processed {count} samples under {args.root}")


if __name__ == "__main__":
    main()
