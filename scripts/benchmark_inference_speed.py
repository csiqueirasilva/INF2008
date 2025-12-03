#!/usr/bin/env python3
"""
Benchmark end-to-end inference speed (preprocess -> neck YOLO -> crop/letterbox -> UNet).

Outputs mean/median timings per stage across N runs after warmup, and total per-frame latency.
Defaults point to the current neck YOLO and positional UNet checkpoints.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from spine_segmentation.commands.train_unet import UNet, _load_checkpoint


def letterbox(img: np.ndarray, target: int, color: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target - new_w
    pad_h = target - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    if img.ndim == 2:
        return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(color, color, color)
    )


def build_unet_input(
    base_gray: np.ndarray,
    use_otsu_channel: bool,
    use_coord_channels: bool,
    otsu_blur_kernel: int,
) -> np.ndarray:
    channels = [base_gray.astype(np.float32) / 255.0]
    if use_otsu_channel:
        k = otsu_blur_kernel
        if k <= 0:
            k = 1
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(base_gray, (k, k), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        channels.append(otsu.astype(np.float32) / 255.0)
    if use_coord_channels:
        h, w = base_gray.shape
        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        channels.extend([yy, xx])
    return np.stack(channels, axis=0)  # (C, H, W)


def load_unet(path: Path, device: torch.device) -> tuple[UNet, dict]:
    ckpt = _load_checkpoint(path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    num_classes = int(ckpt["num_classes"])
    cfg = ckpt.get("config", {}) or {}
    n_channels = int(cfg.get("num_input_channels", 1))
    model = UNet(n_channels=n_channels, n_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def time_ms(fn, runs: int, warmup: int = 5) -> tuple[list[float], float]:
    times = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        if i >= warmup:
            times.append(elapsed)
    mean = float(np.mean(times)) if times else 0.0
    return times, mean


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark end-to-end neck pipeline latency.")
    ap.add_argument("--frame", type=Path, required=True, help="Path to a clahe2 frame (PNG).")
    ap.add_argument("--neck-yolo", type=Path, default=Path("runs/neck_bbox/yolov8n_circ62/weights/best.pt"))
    ap.add_argument("--unet", type=Path, default=Path("runs/unet_cropped6_clahe_pos/unet_best.pt"))
    ap.add_argument("--device", default="cuda", help="Device for YOLO and UNet (cuda or cpu).")
    ap.add_argument("--imgsz", type=int, default=384, help="YOLO input size.")
    ap.add_argument("--crop-size", type=int, default=1024, help="Letterbox size for UNet input.")
    ap.add_argument("--runs", type=int, default=30, help="Number of timed runs (after warmup).")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not timed).")
    ap.add_argument("--conf", type=float, default=0.25, help="Neck YOLO confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.6, help="Neck YOLO IoU threshold.")
    args = ap.parse_args()

    frame = cv2.imread(str(args.frame), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise SystemExit(f"Failed to read frame: {args.frame}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load models
    neck = YOLO(str(args.neck_yolo))
    unet_model, unet_cfg = load_unet(args.unet, device)
    use_otsu = bool(unet_cfg.get("use_otsu_channel", False))
    use_coord = bool(unet_cfg.get("use_coord_channels", False))
    blur_k = int(unet_cfg.get("otsu_blur_kernel", 5))

    def run_once():
        # YOLO expects 3-channel BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        res = neck.predict(
            source=[bgr],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=1,
            verbose=False,
            device=device,
        )[0]
        if res.boxes is not None and len(res.boxes) > 0:
            box = res.boxes.xyxy[0].cpu().numpy().astype(int)
            x0, y0, x1, y1 = box.tolist()
        else:
            # fallback to full frame
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = 0, 0, w - 1, h - 1
        crop = frame[y0 : y1 + 1, x0 : x1 + 1]
        crop_lb = letterbox(crop, args.crop_size, color=0)

        inp = build_unet_input(
            crop_lb,
            use_otsu_channel=use_otsu,
            use_coord_channels=use_coord,
            otsu_blur_kernel=blur_k,
        )
        tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = unet_model(tensor)

    # Warmup + timing
    times, mean_ms = time_ms(run_once, runs=args.runs, warmup=args.warmup)
    print(f"Runs: {args.runs}, warmup: {args.warmup}, device: {device}")
    print(f"Mean total latency: {mean_ms:.2f} ms")
    print(f"Median: {np.median(times):.2f} ms, P90: {np.percentile(times, 90):.2f} ms")


if __name__ == "__main__":
    main()
