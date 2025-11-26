#!/usr/bin/env python3
"""
Crop frames using YOLO txt bboxes and save letterboxed crops and side-by-side comparisons.

Expected YOLO prediction layout (from `yolo detect predict ... save_txt=True save_conf=True`):
  pred_root/images/*.png
  pred_root/labels/*.txt  (class cx cy w h [conf])

We load the ORIGINAL frame from --frame-root (matching stem), crop by the bbox,
letterbox back to the original resolution (or a fixed --target-size), and
optionally write a side-by-side panel (orig | crop).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def letterbox(img, target_w: int, target_h: int, color: int = 0):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if img.ndim == 2:
        letter = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=int(color)
        )
    else:
        letter = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(int(color),) * 3
        )
    return letter


def parse_label(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, cx, cy, bw, bh, *rest = parts
    cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
    x0 = int((cx - bw / 2) * w)
    y0 = int((cy - bh / 2) * h)
    x1 = int((cx + bw / 2) * w)
    y1 = int((cy + bh / 2) * h)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w - 1, x1)
    y1 = min(h - 1, y1)
    return x0, y0, x1, y1


def stack_horizontal(left, right):
    h_left = left.shape[0]
    h_right = right.shape[0]
    target_h = max(h_left, h_right)

    def pad_to_h(img, target):
        h, w = img.shape[:2]
        if h == target:
            return img
        top = (target - h) // 2
        bottom = target - h - top
        if img.ndim == 2:
            return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return cv2.hconcat([pad_to_h(left, target_h), pad_to_h(right, target_h)])


def main():
    ap = argparse.ArgumentParser(description="Crop frames from YOLO predictions and letterbox.")
    ap.add_argument("--pred-root", type=Path, required=True, help="YOLO predict directory (with images/ and labels/)")
    ap.add_argument("--frame-root", type=Path, required=True, help="Directory with original frames")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/frames_cropped_yolo"), help="Output root directory")
    ap.add_argument("--letterbox-color", type=int, default=0, help="Padding color (0-255)")
    ap.add_argument("--target-size", type=int, default=None, help="Optional fixed square size (e.g., 384). Defaults to original frame size.")
    ap.add_argument("--side-by-side", action="store_true", help="Also save orig|crop panels for quick inspection")
    args = ap.parse_args()

    images_dir = args.pred_root / "images"
    if images_dir.exists():
        pred_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    else:
        # Fallback to flat layout (images directly under pred_root)
        pred_images = list(args.pred_root.glob("*.png")) + list(args.pred_root.glob("*.jpg"))
    out_crop_dir = args.out_dir / "crops"
    out_side_dir = args.out_dir / "side_by_side"
    out_crop_dir.mkdir(parents=True, exist_ok=True)
    if args.side_by_side:
        out_side_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_pred_path in pred_images:
        stem = img_pred_path.stem
        lbl_path = args.pred_root / "labels" / f"{stem}.txt"
        frame_path = args.frame_root / f"{stem}.png"
        if not lbl_path.exists() or not frame_path.exists():
            continue
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            continue
        # If multiple detections, pick the first; could sort by conf if present.
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        bbox = parse_label(lines[0], w, h)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        crop = frame[y0 : y1 + 1, x0 : x1 + 1]

        target_w, target_h = (args.target_size, args.target_size) if args.target_size else (w, h)
        boxed = letterbox(crop, target_w, target_h, color=args.letterbox_color)

        cv2.imwrite(str(out_crop_dir / f"{stem}_crop.png"), boxed)
        if args.side_by_side:
            panel = stack_horizontal(frame, boxed)
            cv2.imwrite(str(out_side_dir / f"{stem}_side.png"), panel)
        count += 1

    print(f"Wrote {count} crops to {out_crop_dir}")
    if args.side_by_side:
        print(f"Wrote side-by-side panels to {out_side_dir}")


if __name__ == "__main__":
    main()
