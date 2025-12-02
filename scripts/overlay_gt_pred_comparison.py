#!/usr/bin/env python3
"""
Create comparison panels for GT vs predicted masks:
- Left: GT overlay
- Middle: Pred overlay
- Right: Diff overlay (TP green, FP red, FN blue) with mean IoU text.

Assumes:
- GT masks in --gt-dir (uint8/uint16 labels)
- Pred masks in --pred-dir (same stem, optional --pred-suffix)
- Optional images in --image-dir to blend; otherwise uses GT mask as base.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from spine_segmentation.core.label_colors import label_to_color

LABEL_NAMES = {1: "C1", 2: "C2", 4: "C3", 8: "C4", 16: "C5", 32: "C6", 64: "C7"}

def to_color_mask(label_map: np.ndarray, max_bits: int = 16) -> np.ndarray:
    h, w = label_map.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    unique = [int(v) for v in np.unique(label_map) if v > 0]
    for lid in unique:
        color[label_map == lid] = label_to_color(lid)
    return color


def overlay(base: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else base
    return cv2.addWeighted(base_bgr, 1.0 - alpha, color_mask, alpha, 0.0)


def compute_iou(gt: np.ndarray, pr: np.ndarray, classes: list[int]):
    per_class = {}
    vals = []
    for c in classes:
        g = gt == c
        p = pr == c
        inter = int((g & p).sum())
        union = int((g | p).sum())
        iou = inter / union if union else None
        per_class[c] = iou
        if iou is not None:
            vals.append(iou)
    mean_iou = float(np.mean(vals)) if vals else 0.0
    return mean_iou, per_class


def diff_overlay(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    h, w = gt.shape[:2]
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    # True positive: both equal and >0 (green)
    tp = (gt == pr) & (gt > 0)
    # False positive: pred >0, gt ==0 (red)
    fp = (pr > 0) & (gt == 0)
    # False negative: gt >0, pred ==0 (blue)
    fn = (gt > 0) & (pr == 0)
    diff[tp] = (0, 200, 0)
    diff[fp] = (0, 0, 200)
    diff[fn] = (200, 0, 0)
    return diff


def main():
    ap = argparse.ArgumentParser(description="Overlay GT vs Pred masks side by side.")
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, default=None, help="Optional images to blend; if absent, uses GT mask as base.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--pred-suffix", default="", help="Suffix to append to GT stem for preds (e.g., '_mask').")
    ap.add_argument("--classes", default="1,2,4,8,16,32,64", help="Comma-separated class IDs for IoU text.")
    args = ap.parse_args()

    classes = [int(x) for x in args.classes.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(args.gt_dir.glob(args.pattern))
    count = 0
    best = {c: (None, -1) for c in classes}  # (filename, iou)
    worst = {c: (None, 2) for c in classes}
    best_mean = (None, -1)
    worst_mean = (None, 2)
    for gt_path in gt_files:
        pred_path = args.pred_dir / f"{gt_path.stem}{args.pred_suffix}.png"
        if not pred_path.exists():
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        if gt is None or pred is None or gt.shape != pred.shape:
            continue
        if args.image_dir:
            img_path = args.image_dir / gt_path.name
            base = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if base is None:
                base = gt
        else:
            base = gt

        gt_color = to_color_mask(gt)
        pr_color = to_color_mask(pred)
        ov_gt = overlay(base, gt_color, alpha=0.45)
        ov_pr = overlay(base, pr_color, alpha=0.45)
        diff = diff_overlay(gt, pred)
        mean_iou, per_class_iou = compute_iou(gt, pred, classes)
        cv2.putText(ov_pr, f"IoU={mean_iou:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        combined = np.hstack([ov_gt, ov_pr])
        diff_panel = diff.copy()
        y0 = 20
        for c in classes:
            iou_c = per_class_iou.get(c)
            name = LABEL_NAMES.get(c, str(c))
            txt = f"{name}: {iou_c:.3f}" if iou_c is not None else f"{name}: n/a"
            cv2.putText(diff_panel, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            y0 += 15
        cv2.putText(diff_panel, f"Mean IoU: {mean_iou:.3f}", (10, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        combined = np.hstack([combined, diff_panel])
        out_path = args.out_dir / f"{gt_path.stem}_gt_pred_diff.png"
        cv2.imwrite(str(out_path), combined)
        # track best/worst
        if mean_iou > best_mean[1]:
            best_mean = (out_path.name, mean_iou)
        if mean_iou < worst_mean[1] or worst_mean[1] < 0:
            worst_mean = (out_path.name, mean_iou)
        for c in classes:
            iou_c = per_class_iou.get(c)
            if iou_c is None:
                continue
            if iou_c > best[c][1]:
                best[c] = (out_path.name, iou_c)
            if iou_c < worst[c][1] or worst[c][1] < 0:
                worst[c] = (out_path.name, iou_c)
        count += 1
    print(f"Wrote {count} comparisons to {args.out_dir}")
    print("Best per class:")
    for c in classes:
        name = LABEL_NAMES.get(c, str(c))
        fname, val = best[c]
        if fname:
            print(f"  {name}: {val:.3f} ({fname})")
    print("Worst per class:")
    for c in classes:
        name = LABEL_NAMES.get(c, str(c))
        fname, val = worst[c]
        if fname and val >= 0:
            print(f"  {name}: {val:.3f} ({fname})")
    if best_mean[0]:
        print(f"Best mean IoU: {best_mean[1]:.3f} ({best_mean[0]})")
    if worst_mean[0]:
        print(f"Worst mean IoU: {worst_mean[1]:.3f} ({worst_mean[0]})")


if __name__ == "__main__":
    main()
