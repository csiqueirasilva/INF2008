#!/usr/bin/env python3
"""
Post-process DeepDRR label bitmasks in-place or to a copy:
- Morphological opening (erode->dilate) per class.
- Priority law (C1 > C2 > ... > C7) to enforce single class per pixel.
- Keep only the largest connected component; optional min-area drop.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

BIT_MAP = {"C1": 1, "C2": 2, "C3": 4, "C4": 8, "C5": 16, "C6": 32, "C7": 64}
DEFAULT_CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def apply_opening(binary: np.ndarray, ksize: int, iters: int) -> np.ndarray:
    if ksize <= 0 or binary.max() == 0:
        return binary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=iters)


def enforce_priority(mask: np.ndarray, class_order: Sequence[str]) -> np.ndarray:
    out = np.zeros_like(mask, dtype=mask.dtype)
    for cname in class_order:
        bit = BIT_MAP[cname]
        sel = (mask & bit) != 0
        out[(out == 0) & sel] = bit
    return out


def clean_mask(mask: np.ndarray, class_order: Sequence[str], ksize: int, iters: int, min_area: int) -> np.ndarray:
    # Priority first
    mask = enforce_priority(mask, class_order)
    out = np.zeros_like(mask, dtype=mask.dtype)
    for cname in class_order:
        bit = BIT_MAP[cname]
        binary = (mask & bit).astype("uint8")
        binary = apply_opening(binary, ksize, iters)
        if binary.max() == 0:
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = areas.argmax() + 1
        if min_area > 0 and areas[max_idx - 1] < min_area:
            continue
        keep = (labels == max_idx)
        out[keep] = bit
    return out


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    colors = {
        1: (0, 255, 0), 2: (0, 200, 255), 4: (255, 128, 0),
        8: (255, 0, 0), 16: (128, 0, 255), 32: (255, 0, 255), 64: (255, 255, 0)
    }
    if len(gray.shape) == 2:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        base = gray.copy()
    overlay = base.copy()
    for bit, color in colors.items():
        if (mask & bit).max() == 0:
            continue
        channel = (mask & bit) > 0
        overlay[channel] = overlay[channel] * 0.5 + np.array(color) * 0.5
    return overlay.astype(np.uint8)


def process_file(mask_path: Path, ksize: int, iters: int, classes: Sequence[str], min_area: int) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return
    clean = enforce_priority(mask, classes)
    clean = clean_mask(clean, classes, ksize, iters, min_area)
    cv2.imwrite(str(mask_path), clean)
    # overlay
    # Choose the matching image for overlay
    if "circular_synth" in mask_path.name:
        img_path = mask_path.parent / "circular-synth.png"
        if not img_path.exists():
            img_path = mask_path.parent / "clahe2.png"
    else:
        img_path = mask_path.parent / "clahe2.png"
        if not img_path.exists():
            img_path = mask_path.parent / "circular-synth.png"
    if img_path.exists():
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            ov = overlay_mask(gray, clean)
            cv2.imwrite(str(mask_path.parent / f"{mask_path.stem}_overlay.png"), ov)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean DeepDRR bitmasks with opening + priority.")
    ap.add_argument("--root", type=Path, required=True, help="Root containing deepdrr folders.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Optional output root. If set, input is copied here before cleaning.")
    ap.add_argument("--kernel", type=int, default=3, help="Opening kernel size.")
    ap.add_argument("--iters", type=int, default=1, help="Opening iterations.")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="Comma-separated classes to keep, priority order.")
    ap.add_argument("--min-component-area", type=int, default=0,
                    help="Drop label if largest CC below this area (0 disables).")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    work_root = args.root
    if args.out_root:
        if args.out_root == args.root:
            raise SystemExit("out-root must differ from root.")
        if args.out_root.exists() and any(args.out_root.iterdir()):
            raise SystemExit(f"out-root {args.out_root} exists and is not empty.")
        args.out_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.root, args.out_root, dirs_exist_ok=True)
        work_root = args.out_root
        print(f"Copied {args.root} -> {work_root}")

    targets = ["label_bitmask_cropped_letterboxed.png", "label_bitmask_circular_synth.png"]
    mask_files = []
    for t in targets:
        mask_files.extend(work_root.rglob(t))
    if not mask_files:
        raise SystemExit("No bitmask files found under root.")

    for mp in mask_files:
        process_file(mp, ksize=args.kernel, iters=args.iters, classes=classes, min_area=args.min_component_area)

    print(f"Processed {len(mask_files)} bitmasks under {work_root}")


if __name__ == "__main__":
    main()
