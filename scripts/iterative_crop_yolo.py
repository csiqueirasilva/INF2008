#!/usr/bin/env python3
"""
Iterative crop-and-detect helper for neck bbox refinement.

Workflow per image:
1) Run YOLO on the (preprocessed) mask image.
2) Pick a box (highest confidence or smallest area).
3) Crop to that box and repeat up to --max-iter times, stopping early if the
   bbox area does not shrink enough (--min-improve).
4) Apply the final bbox to the ORIGINAL frame, letterbox back to the original size,
   and optionally save a side-by-side panel.

Assumptions:
- Detection runs on preprocessed masks (e.g., CLAHE2+Otsu) located in --mask-dir.
- Original frames (PNG/JPG) with the same stem live in --frame-dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def letterbox(img: np.ndarray, target_w: int, target_h: int, color: int = 0) -> np.ndarray:
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
        return cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=int(color)
        )
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(int(color),) * 3
    )


def stack_horizontal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])

    def pad_to_h(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        top = (h - img.shape[0]) // 2
        bottom = h - img.shape[0] - top
        if img.ndim == 2:
            return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return cv2.hconcat([pad_to_h(left), pad_to_h(right)])


def choose_box(boxes, strategy: str) -> Tuple[np.ndarray, float]:
    """
    boxes: list of ultralytics Boxes
    returns: (xyxy np.array shape (4,), conf)
    """
    if not boxes:
        return None, 0.0
    if strategy == "smallest":
        scored = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            scored.append((area, xyxy, float(b.conf[0])))
        scored.sort(key=lambda t: t[0])
        area, xyxy, conf = scored[0]
        return xyxy, conf
    else:  # best_conf
        best = max(boxes, key=lambda b: float(b.conf[0]))
        return best.xyxy[0].cpu().numpy(), float(best.conf[0])


def map_to_original(xyxy: np.ndarray, offset: tuple[int, int]) -> tuple[int, int, int, int]:
    x_off, y_off = offset
    x0 = int(round(xyxy[0] + x_off))
    y0 = int(round(xyxy[1] + y_off))
    x1 = int(round(xyxy[2] + x_off))
    y1 = int(round(xyxy[3] + y_off))
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser(description="Iterative crop using YOLO bboxes.")
    ap.add_argument("--model", type=Path, required=True, help="Path to YOLO weights (best.pt).")
    ap.add_argument("--mask-dir", type=Path, required=True, help="Directory with preprocessed mask images.")
    ap.add_argument("--frame-dir", type=Path, required=True, help="Directory with original frames (PNG/JPG).")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/frames_cropped_iter"), help="Output root.")
    ap.add_argument("--max-iter", type=int, default=2, help="Max refinement iterations.")
    ap.add_argument("--min-improve", type=float, default=0.05, help="Min relative area shrink to keep iterating.")
    ap.add_argument("--strategy", choices=["best_conf", "smallest"], default="best_conf", help="Box selection rule.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO predict.")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    ap.add_argument(
        "--min-box-frac",
        type=float,
        default=0.0,
        help="Minimum allowed final box area as a fraction of the original image area. 0 disables.",
    )
    ap.add_argument(
        "--final-pad-frac",
        type=float,
        default=0.0,
        help="Optional padding added around the final box as a fraction of the final box size.",
    )
    ap.add_argument("--letterbox-color", type=int, default=0, help="Padding color for final letterbox.")
    ap.add_argument("--side-by-side", action="store_true", help="Save orig|crop panels.")
    args = ap.parse_args()

    model = YOLO(str(args.model))

    out_crop_dir = args.out_dir / "crops"
    out_side_dir = args.out_dir / "side_by_side"
    out_crop_dir.mkdir(parents=True, exist_ok=True)
    if args.side_by_side:
        out_side_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(list(args.mask_dir.glob("*.png")) + list(args.mask_dir.glob("*.jpg")))
    processed = 0
    for mask_path in mask_paths:
        stem = mask_path.stem.replace("_mask", "")  # allow *_mask naming
        frame_path = None
        for ext in (".png", ".jpg"):
            candidate = args.frame_dir / f"{stem}{ext}"
            if candidate.exists():
                frame_path = candidate
                break
        if frame_path is None:
            continue

        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        # YOLO expects 3-channel input; duplicate grayscale mask into 3 channels.
        mask_img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        h0, w0 = mask_img.shape[:2]
        cur_mask = mask_img_rgb
        offset_x, offset_y = 0, 0
        last_area = None
        final_box = None

        for it in range(max(1, args.max_iter)):
            results = model.predict(cur_mask, conf=args.conf, iou=args.iou, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                break
            xyxy, conf = choose_box(results[0].boxes, args.strategy)
            if xyxy is None:
                break

            # Map to original coordinates
            mapped = map_to_original(xyxy, (offset_x, offset_y))
            x0, y0, x1, y1 = mapped
            # Clamp to original image bounds
            x0 = max(0, min(w0 - 1, x0))
            y0 = max(0, min(h0 - 1, y0))
            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            area = (x1 - x0 + 1) * (y1 - y0 + 1)

            if last_area is not None:
                improve = (last_area - area) / max(last_area, 1e-6)
                if improve < args.min_improve:
                    final_box = (x0, y0, x1, y1)
                    break
            final_box = (x0, y0, x1, y1)
            last_area = area

            # Prepare next iteration crop
            offset_x, offset_y = x0, y0
            cur_mask = mask_img_rgb[y0 : y1 + 1, x0 : x1 + 1]

        if final_box is None:
            continue

        x0, y0, x1, y1 = final_box
        # Expand overly small boxes to a minimum fraction of the original area.
        if args.min_box_frac > 0:
            min_area = args.min_box_frac * (h0 * w0)
            cur_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            if cur_area < min_area:
                scale = (min_area / max(cur_area, 1e-6)) ** 0.5
                cx = 0.5 * (x0 + x1)
                cy = 0.5 * (y0 + y1)
                half_w = 0.5 * (x1 - x0 + 1) * scale
                half_h = 0.5 * (y1 - y0 + 1) * scale
                x0 = int(round(cx - half_w))
                x1 = int(round(cx + half_w))
                y0 = int(round(cy - half_h))
                y1 = int(round(cy + half_h))
                x0 = max(0, min(w0 - 1, x0))
                x1 = max(0, min(w0 - 1, x1))
                y0 = max(0, min(h0 - 1, y0))
                y1 = max(0, min(h0 - 1, y1))
        # Optional padding around the final box.
        if args.final_pad_frac > 0:
            pad_x = int(round((x1 - x0 + 1) * args.final_pad_frac))
            pad_y = int(round((y1 - y0 + 1) * args.final_pad_frac))
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(w0 - 1, x1 + pad_x)
            y1 = min(h0 - 1, y1 + pad_y)

        crop = frame[y0 : y1 + 1, x0 : x1 + 1]
        boxed = letterbox(crop, frame.shape[1], frame.shape[0], color=args.letterbox_color)
        cv2.imwrite(str(out_crop_dir / f"{stem}_crop.png"), boxed)
        if args.side_by_side:
            panel = stack_horizontal(frame, boxed)
            cv2.imwrite(str(out_side_dir / f"{stem}_side.png"), panel)
        processed += 1

    print(f"Processed {processed} frames → {out_crop_dir}")
    if args.side_by_side:
        print(f"Side-by-side panels → {out_side_dir}")


if __name__ == "__main__":
    main()
