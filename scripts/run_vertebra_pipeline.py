#!/usr/bin/env python3
"""
End-to-end vertebra (C2–C4) pipeline:
1) Run YOLO on CLAHE2 frames to detect C2/C3/C4 (classes 1,2,3).
2) Pick top-1 box per class per image (by confidence).
3) Crop each box, letterbox to UNet size, run the class-specific UNet.
4) Paste masks back to full frame and save mask + overlay; also save YOLO bbox overlays.

Assumptions:
- Input frames are already CLAHE2-processed single-channel images (e.g., *_clahe2.png).
- UNets are binary models (background vs target vertebra) trained on 384×384 letterboxed inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Reuse UNet/checkpoint loading from the training module
from spine_segmentation.commands.train_unet import UNet, _load_checkpoint


# Class ids and display
CLASS_IDS = [1, 2, 3]  # C2, C3, C4 in the per-vertebra YOLO
CLASS_NAMES = {1: "C2", 2: "C3", 3: "C4"}
PALETTE = {
    1: (255, 0, 0),   # Blue-ish for C2
    2: (0, 255, 0),   # Green for C3
    3: (0, 0, 255),   # Red for C4
}
NECK_COLOR = (0, 255, 255)


@dataclass
class Detection:
    cls: int
    conf: float
    xyxy: Tuple[int, int, int, int]


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=img.dtype)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def unletterbox(mask_lb: np.ndarray, scale: float, x0: int, y0: int, target_shape: Tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    # Extract the region that corresponds to the resized content
    h_lb, w_lb = mask_lb.shape[:2]
    new_w = min(w_lb, int(round(tw * scale)))
    new_h = min(h_lb, int(round(th * scale)))
    x1 = x0 + new_w
    y1 = y0 + new_h
    content = mask_lb[y0:y1, x0:x1]
    if content.size == 0:
        return np.zeros(target_shape, dtype=mask_lb.dtype)
    restored = cv2.resize(content, (tw, th), interpolation=cv2.INTER_NEAREST)
    return restored


def load_unet(path: Path) -> tuple[torch.nn.Module, int]:
    ckpt = _load_checkpoint(path, map_location=torch.device("cpu"))
    state_dict = ckpt.get("model_state_dict")
    num_classes = int(ckpt.get("num_classes", 1))
    if state_dict is None:
        raise RuntimeError(f"Checkpoint at {path} missing model_state_dict")
    model = UNet(n_channels=1, n_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes


def run_unet(model: torch.nn.Module, num_classes: int, img_384: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(img_384).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if num_classes <= 1:
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return (prob > 0.5).astype(np.uint8)
    else:
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return (pred == 1).astype(np.uint8)  # foreground channel


def draw_bbox_overlay(gray: np.ndarray, dets: List[Detection]) -> np.ndarray:
    ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for d in dets:
        color = PALETTE.get(d.cls, (0, 255, 255))
        x0, y0, x1, y1 = d.xyxy
        cv2.rectangle(ov, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            ov,
            f"{CLASS_NAMES[d.cls]} {d.conf:.2f}",
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return ov


def main() -> None:
    ap = argparse.ArgumentParser(description="Run vertebra (C2–C4) YOLO+UNet pipeline on CLAHE2 frames.")
    ap.add_argument("--frame-dir", type=Path, required=True, help="Directory with CLAHE2 frames (e.g., prepared/frames7_clahe2).")
    ap.add_argument("--pattern", default="*_clahe2.png", help="Glob pattern for frames.")
    ap.add_argument("--neck-yolo-model", type=Path, required=True, help="Path to coarse neck YOLO model.")
    ap.add_argument("--yolo-model", type=Path, required=True, help="Path to per-vertebra YOLO model.")
    ap.add_argument("--unet-c2", type=Path, required=True, help="Path to C2 UNet model.")
    ap.add_argument("--unet-c3", type=Path, required=True, help="Path to C3 UNet model.")
    ap.add_argument("--unet-c4", type=Path, required=True, help="Path to C4 UNet model.")
    ap.add_argument("--imgsz", type=int, default=384, help="Image size for YOLO and UNet letterbox.")
    ap.add_argument("--conf", type=float, default=0.2, help="Vertebra YOLO confidence.")
    ap.add_argument("--iou", type=float, default=0.6, help="Vertebra YOLO IoU.")
    ap.add_argument("--neck-conf", type=float, default=0.2, help="Neck YOLO confidence.")
    ap.add_argument("--neck-iou", type=float, default=0.6, help="Neck YOLO IoU.")
    ap.add_argument("--neck-pad-frac", type=float, default=0.1, help="Padding fraction to expand the neck bbox.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    args = ap.parse_args()

    out_bbox = args.out_dir / "bbox_overlays"
    out_masks = args.out_dir / "masks"
    out_overlays = args.out_dir / "overlays"
    for d in (out_bbox, out_masks, out_overlays):
        d.mkdir(parents=True, exist_ok=True)

    # Load models
    neck_yolo = YOLO(str(args.neck_yolo_model))
    yolo = YOLO(str(args.yolo_model))
    unets = {1: load_unet(args.unet_c2), 2: load_unet(args.unet_c3), 3: load_unet(args.unet_c4)}

    img_paths = sorted(args.frame_dir.glob(args.pattern))
    total = len(img_paths)
    if not img_paths:
        raise SystemExit(f"No images found in {args.frame_dir} with pattern {args.pattern}")
    print(f"Found {total} images in {args.frame_dir} (pattern={args.pattern})")

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Neck YOLO inference
        neck_res = neck_yolo.predict(
            source=[img_bgr],
            imgsz=args.imgsz,
            conf=args.neck_conf,
            iou=args.neck_iou,
            verbose=False,
        )[0]
        neck_box: Detection | None = None
        if neck_res.boxes is not None and len(neck_res.boxes) > 0:
            boxes = neck_res.boxes.xyxy.cpu().numpy()
            confs = neck_res.boxes.conf.cpu().numpy()
            for conf, (x0, y0, x1, y1) in zip(confs, boxes):
                d = Detection(cls=0, conf=float(conf), xyxy=(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
                if neck_box is None or d.conf > neck_box.conf:
                    neck_box = d

        if neck_box is None:
            # fallback: use full image as crop
            nx0, ny0, nx1, ny1 = 0, 0, w - 1, h - 1
        else:
            nx0, ny0, nx1, ny1 = neck_box.xyxy
            pad = int(max(nx1 - nx0 + 1, ny1 - ny0 + 1) * args.neck_pad_frac)
            nx0 = max(0, nx0 - pad)
            ny0 = max(0, ny0 - pad)
            nx1 = min(w - 1, nx1 + pad)
            ny1 = min(h - 1, ny1 + pad)

        crop_bgr = img_bgr[ny0 : ny1 + 1, nx0 : nx1 + 1]
        crop_gray = img[ny0 : ny1 + 1, nx0 : nx1 + 1]
        cw, ch = crop_bgr.shape[1], crop_bgr.shape[0]

        # Vertebra YOLO on cropped region
        res = yolo.predict(
            source=[crop_bgr],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=CLASS_IDS,
            verbose=False,
        )[0]

        dets_all: List[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for cls, conf, (x0, y0, x1, y1) in zip(clss, confs, boxes):
                dets_all.append(
                    Detection(
                        cls=int(cls),
                        conf=float(conf),
                        xyxy=(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
                    )
                )

        # Save bbox overlay (draw neck + vertebra on full frame)
        bbox_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if neck_box is not None:
            cv2.rectangle(bbox_img, (nx0, ny0), (nx1, ny1), NECK_COLOR, 2)
            cv2.putText(
                bbox_img,
                f"neck {neck_box.conf:.2f}",
                (nx0, max(0, ny0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                NECK_COLOR,
                2,
                cv2.LINE_AA,
            )
        for d in dets_all:
            color = PALETTE.get(d.cls, (0, 255, 255))
            x0, y0, x1, y1 = d.xyxy
            # map to full frame coords
            x0 += nx0; x1 += nx0; y0 += ny0; y1 += ny0
            cv2.rectangle(bbox_img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                bbox_img,
                f"{CLASS_NAMES[d.cls]} {d.conf:.2f}",
                (x0, max(0, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(out_bbox / img_path.name), bbox_img)

        # Select best per class (in crop coords)
        best: Dict[int, Detection] = {}
        for d in dets_all:
            if d.cls not in CLASS_IDS:
                continue
            if d.cls not in best or d.conf > best[d.cls].conf:
                best[d.cls] = d

        full_mask = np.zeros_like(img, dtype=np.uint8)
        for cls_id, det in best.items():
            x0, y0, x1, y1 = det.xyxy
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(cw - 1, x1), min(ch - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = crop_gray[y0 : y1 + 1, x0 : x1 + 1]
            lb_img, scale, off_x, off_y = letterbox(crop, args.imgsz)
            model, num_classes = unets[cls_id]
            pred_mask = run_unet(model, num_classes, lb_img)
            mask_crop = unletterbox(pred_mask, scale, off_x, off_y, (y1 - y0 + 1, x1 - x0 + 1))
            # map to full frame
            full_mask[ny0 + y0 : ny0 + y1 + 1, nx0 + x0 : nx0 + x1 + 1][mask_crop > 0] = cls_id

        # Save mask and overlay
        cv2.imwrite(str(out_masks / img_path.name), full_mask)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cls_id, color in PALETTE.items():
            overlay[full_mask == cls_id] = (
                0.55 * overlay[full_mask == cls_id] + 0.45 * np.array(color)
            ).astype(np.uint8)
        cv2.imwrite(str(out_overlays / img_path.name), overlay)
        if idx % 20 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    print(f"Done. BBox overlays: {out_bbox}, masks: {out_masks}, overlays: {out_overlays}")


if __name__ == "__main__":
    main()
