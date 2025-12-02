#!/usr/bin/env python3
"""
Run YOLO (full-frame bbox) + UNet (cropped mask bitmask) in one pass:
1) detect neck bbox on full-frame Otsu masks
2) crop + letterbox to original size
3) run UNet on the cropped mask
4) save overlays and predicted masks
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from spine_segmentation.commands.predict_unet import _load_model, _predict_image, _resolve_device

def double_clahe(img: np.ndarray, clip1: float = 2.0, tile1: int = 8, clip2: float = 2.0, tile2: int = 8) -> np.ndarray:
    clahe1 = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(tile1, tile1)).apply(img)
    clahe2 = cv2.createCLAHE(clipLimit=clip2, tileGridSize=(tile2, tile2)).apply(clahe1)
    return clahe2


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
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=int(color))
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(int(color),) * 3)


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)


def main():
    ap = argparse.ArgumentParser(description="YOLO crop + UNet segment on full-frame masks.")
    ap.add_argument("--yolo-model", type=Path, required=True, help="YOLO weights path (e.g., runs/neck_bbox/.../best.pt)")
    ap.add_argument("--unet-model", type=Path, required=True, help="UNet checkpoint (unet_best.pt)")
    ap.add_argument("--mask-dir", type=Path, required=True, help="Directory with full-frame Otsu masks (e.g., prepared/framesXX_otsu)")
    ap.add_argument("--frame-dir", type=Path, required=False, help="Optional directory with original frames to crop alongside masks")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/infer_unet_pipeline"), help="Output root")
    ap.add_argument("--pattern", default="*_mask.png", help="Glob pattern for masks")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--max-det", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay blend factor")
    ap.add_argument("--unet-clahe", action="store_true", help="Apply double CLAHE to crop before UNet (for CLAHE-only models).")
    args = ap.parse_args()

    masks = sorted(args.mask_dir.glob(args.pattern))
    if not masks:
        raise SystemExit(f"No masks matched {args.pattern} under {args.mask_dir}")

    out_crops = args.out_dir / "crops"
    out_overlays = args.out_dir / "overlays"
    out_preds = args.out_dir / "pred_masks"
    out_frame_crops = args.out_dir / "frame_crops"
    out_crops.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    out_preds.mkdir(parents=True, exist_ok=True)
    if args.frame_dir:
        out_frame_crops.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(None)
    model_unet, _ = _load_model(args.unet_model, device)
    yolo = YOLO(str(args.yolo_model))

    processed = 0
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        H, W = mask.shape[:2]
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        yolo_res = yolo.predict(
            source=mask_bgr,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
        )
        if not yolo_res or yolo_res[0].boxes is None or len(yolo_res[0].boxes) == 0:
            continue
        best = max(yolo_res[0].boxes, key=lambda b: float(b.conf[0]))
        xyxy = best.xyxy[0].cpu().numpy()
        x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
        x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, W, H)

        crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        crop_mask_lb = letterbox(crop_mask, W, H, color=0)

        stem = mask_path.stem.replace("_mask", "")
        crop_path = out_crops / f"{stem}_crop.png"
        cv2.imwrite(str(crop_path), crop_mask_lb)

        if args.frame_dir:
            frame_path = args.frame_dir / f"{stem}.png"
            if not frame_path.exists():
                frame_path = args.frame_dir / f"{stem}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                crop_frame = frame[y0 : y1 + 1, x0 : x1 + 1]
                crop_frame_lb = letterbox(crop_frame, W, H, color=0)
                cv2.imwrite(str(out_frame_crops / f"{stem}_frame_crop.png"), crop_frame_lb)

        overlay_path = out_overlays / f"{stem}_overlay.png"
        pred_path = out_preds / f"{stem}_pred.png"

        # Run UNet on the cropped mask (optionally apply CLAHE2).
        unet_input = crop_mask_lb
        if args.unet_clahe:
            unet_input = double_clahe(unet_input)
            cv2.imwrite(str(out_crops / f"{stem}_crop_clahe2.png"), unet_input)
            unet_image_path = out_crops / f"{stem}_crop_clahe2.png"
        else:
            unet_image_path = crop_path

        _predict_image(
            model=model_unet,
            device=device,
            image_path=unet_image_path,
            overlay_out=overlay_path,
            mask_out=pred_path,
            alpha=args.alpha,
            clahe=False,
        )
        processed += 1

    print(f"Processed {processed} masks → {args.out_dir}")


if __name__ == "__main__":
    main()
