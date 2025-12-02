#!/usr/bin/env python3
"""
Extract per-vertebra crops (with padding) from DeepDRR outputs and save
letterboxed images/masks plus quick overlays for sanity checking.

Intended to build per-vertebra UNet datasets (e.g., C2–C4) while keeping
the same letterbox/size the current UNet expects (default 384x384).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Supported DeepDRR image keys
IMAGE_CHOICES = {
    "clahe2": Path("clahe2.png"),
    "circular_synth": Path("circular-synth.png"),
    "circular_clahe2": Path("circular_synth_clahe2.png"),
    "orig_gray": Path("orig_gray.png"),
    "otsu_mask": Path("otsu_mask.png"),
    "precrop": Path("deepdrr_precrop.png"),
}

# Class ids (bitmask values) and human names
CLASS_LABELS = [1, 2, 4, 8, 16, 32, 64]  # C1..C7
CLASS_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CLASS_COLORS = [
    (50, 80, 200),
    (70, 170, 240),
    (120, 210, 240),
    (120, 200, 80),
    (100, 160, 70),
    (40, 120, 180),
    (150, 90, 40),
]


def _bbox_from_class(mask: np.ndarray, lid: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask == lid)
    if not xs.size or not ys.size:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _pad_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int, pad_frac: float) -> tuple[int, int, int, int]:
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    pad = int(max(bw, bh) * pad_frac)
    return max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad)


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if img.ndim == 2 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, *(() if img.ndim == 2 else (3,))), dtype=img.dtype)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w, ...] = resized
    return canvas, scale, x_off, y_off


def _overlay(img_gray: np.ndarray, mask: np.ndarray, lid: int, alpha: float = 0.45) -> np.ndarray:
    color = CLASS_COLORS[CLASS_LABELS.index(lid)]
    color_arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_arr[mask == lid] = color
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) if img_gray.ndim == 2 else img_gray.copy()
    return cv2.addWeighted(base, 1 - alpha, color_arr, alpha, 0)


def iter_bitmasks(root: Path, bitmask_name: str, limit: int = 0) -> Iterable[Path]:
    masks = list(root.rglob(bitmask_name))
    return masks if limit <= 0 else masks[:limit]


def parse_classes(selection: str | None) -> list[int]:
    if not selection:
        return CLASS_LABELS
    parts = selection.split(",")
    mapping = {name: lid for name, lid in zip(CLASS_NAMES, CLASS_LABELS)}
    lids: list[int] = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            lid = int(p)
        else:
            lid = mapping.get(p.upper(), None)
        if lid is None or lid not in CLASS_LABELS:
            raise ValueError(f"Unknown class spec '{p}'. Use C1..C7 or 1,2,4,...")
        lids.append(lid)
    return lids


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract per-vertebra crops with padding and letterbox for UNet training.")
    ap.add_argument("--input-root", type=Path, required=True, help="Root with DeepDRR outputs (e.g., outputs/dataset_synth_headneck_2).")
    ap.add_argument("--out-dir", type=Path, default=Path("prepared/vertebra_crops_clahe2"), help="Destination root.")
    ap.add_argument("--image-key", choices=list(IMAGE_CHOICES.keys()), default="precrop", help="Which DeepDRR image to crop.")
    ap.add_argument("--bitmask-name", default="label_bitmask_circular_synth.png", help="Bitmask filename to use for boxes (e.g., label_bitmask_circular_synth.png or label_bitmask_cropped_letterboxed.png).")
    ap.add_argument("--classes", default=None, help="Comma-separated classes to keep (e.g., C2,C3,C4 or 1,2,4). Default: all C1..C7.")
    ap.add_argument("--padding-frac", type=float, default=0.0, help="Padding as fraction of max(box_w, box_h). Use 0 for no padding.")
    ap.add_argument("--out-size", type=int, default=384, help="Letterbox size.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on samples processed.")
    ap.add_argument("--apply-clahe2", action="store_true", help="Apply double CLAHE to the source image before cropping.")
    ap.add_argument("--clip-limit1", type=float, default=2.0, help="First CLAHE clip limit.")
    ap.add_argument("--clip-limit2", type=float, default=2.0, help="Second CLAHE clip limit.")
    ap.add_argument("--tile-size1", type=int, default=8, help="First CLAHE tile size.")
    ap.add_argument("--tile-size2", type=int, default=8, help="Second CLAHE tile size.")
    args = ap.parse_args()

    img_name = IMAGE_CHOICES[args.image_key]
    target_classes = parse_classes(args.classes)

    out_raw_img = args.out_dir / "crops" / "images"
    out_raw_mask = args.out_dir / "crops" / "masks"
    out_lb_img = args.out_dir / "letterbox" / "images"
    out_lb_mask = args.out_dir / "letterbox" / "masks"
    out_overlay = args.out_dir / "letterbox" / "overlays"
    for d in (out_raw_img, out_raw_mask, out_lb_img, out_lb_mask, out_overlay):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.csv"
    rows = []

    for bm_path in iter_bitmasks(args.input_root, args.bitmask_name, args.limit):
        deepdrr_dir = bm_path.parent
        img_path = deepdrr_dir / img_name
        if not img_path.exists():
            continue
        case = deepdrr_dir.parent.parent.name  # HN_Pxxx
        angle = deepdrr_dir.parent.name        # off_m0p2, etc.

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if args.apply_clahe2:
            clahe1 = cv2.createCLAHE(clipLimit=args.clip_limit1, tileGridSize=(args.tile_size1, args.tile_size1))
            clahe2 = cv2.createCLAHE(clipLimit=args.clip_limit2, tileGridSize=(args.tile_size2, args.tile_size2))
            img = clahe2.apply(clahe1.apply(img))
        mask_full = cv2.imread(str(bm_path), cv2.IMREAD_UNCHANGED)
        if mask_full is None:
            continue
        h, w = img.shape[:2]

        for lid in target_classes:
            bbox = _bbox_from_class(mask_full, lid)
            if bbox is None:
                continue
            x0, y0, x1, y1 = _pad_bbox(*bbox, w=w, h=h, pad_frac=args.padding_frac)
            crop_img = img[y0:y1 + 1, x0:x1 + 1]
            crop_mask = mask_full[y0:y1 + 1, x0:x1 + 1]

            lb_img, scale, x_off, y_off = _letterbox(crop_img, args.out_size)
            lb_mask, _, _, _ = _letterbox(crop_mask, args.out_size)

            cls_name = CLASS_NAMES[CLASS_LABELS.index(lid)]
            stem = f"{case}_{angle}_{cls_name}"

            raw_img_path = out_raw_img / f"{stem}.png"
            raw_mask_path = out_raw_mask / f"{stem}.png"
            lb_img_path = out_lb_img / f"{stem}.png"
            lb_mask_path = out_lb_mask / f"{stem}.png"
            overlay_path = out_overlay / f"{stem}.png"

            cv2.imwrite(str(raw_img_path), crop_img)
            # save masks as label ids (0 background, 1 foreground)
            cv2.imwrite(str(raw_mask_path), (crop_mask == lid).astype(np.uint8))
            cv2.imwrite(str(lb_img_path), lb_img)
            cv2.imwrite(str(lb_mask_path), (lb_mask == lid).astype(np.uint8))
            overlay = _overlay(lb_img, lb_mask, lid)
            cv2.imwrite(str(overlay_path), overlay)

            rows.append({
                "case": case,
                "angle": angle,
                "class_name": cls_name,
                "class_id": lid,
                "src_image": img_path.as_posix(),
                "src_mask": bm_path.as_posix(),
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "padding_frac": args.padding_frac,
                "crop_image": raw_img_path.as_posix(),
                "crop_mask": raw_mask_path.as_posix(),
                "letterbox_image": lb_img_path.as_posix(),
                "letterbox_mask": lb_mask_path.as_posix(),
                "overlay_image": overlay_path.as_posix(),
                "scale": scale,
                "offset_x": x_off,
                "offset_y": y_off,
            })

    if rows:
        with manifest_path.open("w", newline="") as f:
            fieldnames = [
                "case", "angle", "class_name", "class_id",
                "src_image", "src_mask",
                "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
                "padding_frac",
                "crop_image", "crop_mask",
                "letterbox_image", "letterbox_mask",
                "overlay_image",
                "scale", "offset_x", "offset_y",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} crops to {manifest_path}")
    else:
        print("No crops written; check inputs and class selection.")


if __name__ == "__main__":
    main()
