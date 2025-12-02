#!/usr/bin/env python3
"""
Compute per-class IoU/Dice for UNet predictions vs. ground-truth masks.

Defaults match our cropped UNet setup:
- GT masks: prepared/unet_cropped_flat/masks
- Pred masks: outputs/unet_cropped_flat_train_preds
- Classes: power-of-two labels for C1–C7: [1,2,4,8,16,32,64]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np


def compute_metrics(gt_path: Path, pred_path: Path, classes: list[int]):
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    pr = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    if gt is None or pr is None or gt.shape != pr.shape:
        return None
    per_class = {}
    for c in classes:
        g = gt == c
        p = pr == c
        inter = int((g & p).sum())
        union = int((g | p).sum())
        dice_denom = g.sum() + p.sum()
        iou = inter / union if union else None
        dice = (2 * inter) / dice_denom if dice_denom else None
        per_class[c] = {"iou": iou, "dice": dice}
    return per_class


def main():
    ap = argparse.ArgumentParser(description="Evaluate IoU/Dice between UNet predictions and ground truth.")
    ap.add_argument("--gt-dir", type=Path, default=Path("prepared/unet_cropped_flat/masks"))
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/unet_cropped_flat_train_preds"))
    ap.add_argument("--pattern", default="*.png", help="Glob pattern for GT masks.")
    ap.add_argument("--classes", default="1,2,4,8,16,32,64", help="Comma-separated class IDs.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of files.")
    ap.add_argument("--pred-suffix", default="", help="Suffix to append to GT stem when finding preds (e.g., '_mask').")
    args = ap.parse_args()

    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    gt_files = sorted(args.gt_dir.glob(args.pattern))
    if args.limit > 0:
        gt_files = gt_files[: args.limit]

    agg = {c: [] for c in classes}
    count = 0
    for gt_path in gt_files:
        pred_path = args.pred_dir / f"{gt_path.stem}{args.pred_suffix}.png"
        metrics = compute_metrics(gt_path, pred_path, classes)
        if metrics is None:
            continue
        count += 1
        for c, vals in metrics.items():
            if vals["iou"] is not None:
                agg[c].append(vals["iou"])

    if count == 0:
        print("No matching GT/pred pairs found.")
        return

    for c in classes:
        vals = agg[c]
        if vals:
            print(f"Class {c} IoU: {np.mean(vals):.3f} over {len(vals)} samples")
    all_vals = [v for vs in agg.values() for v in vs]
    if all_vals:
        print(f"Mean IoU (all classes): {np.mean(all_vals):.3f}")
    print(f"Samples evaluated: {count}")


if __name__ == "__main__":
    main()
