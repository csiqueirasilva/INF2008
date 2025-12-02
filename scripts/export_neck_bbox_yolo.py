#!/usr/bin/env python3
"""
Export synthetic neck crops + bboxes to a YOLO-style detection dataset.

This walks DeepDRR generation outputs (e.g., outputs/dataset_synth_headneck/**/bbox_labels.json)
and writes train/val splits with matching images and YOLO txt labels (single class: neck).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_CHOICES = {
    "otsu_overlay": Path("deepdrr/otsu_overlay.png"),
    "otsu_mask": Path("deepdrr/otsu_mask.png"),
    "clahe2": Path("deepdrr/clahe2.png"),
    "deepdrr": Path("deepdrr.png"),
    "circular_synth": Path("deepdrr/circular-synth.png"),
    "circular_synth_otsu": Path("deepdrr/circular-synth.png"),
    "circular_synth_clahe2": Path("deepdrr/circular-synth.png"),
}


def _load_bbox(bbox_json: Path) -> tuple[int, int, int, int] | None:
    with open(bbox_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    bbox = payload.get("bbox")
    if bbox is None:
        return None
    return int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"])


def _norm_yolo(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[float, float, float, float]:
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0 + 1) / w
    bh = (y1 - y0 + 1) / h
    return cx, cy, bw, bh


def collect_samples(input_root: Path, image_key: str, use_circular_bitmask: bool) -> list[tuple[Path, tuple[int, int, int, int]]]:
    samples: list[tuple[Path, tuple[int, int, int, int]]] = []
    if use_circular_bitmask and image_key.startswith("circular_"):
        for bm_path in input_root.rglob("label_bitmask_circular_synth.png"):
            # bm_path.parent is deepdrr/, so grab the filename only
            img_path = bm_path.parent / IMAGE_CHOICES[image_key].name
            if not img_path.exists():
                continue
            bbox = _bbox_from_bitmask(bm_path)
            if bbox is None:
                continue
            samples.append((img_path, bbox))
    else:
        for bbox_path in input_root.rglob("bbox_labels.json"):
            img_path = bbox_path.parent / IMAGE_CHOICES[image_key]
            if not img_path.exists():
                continue
            bbox = _load_bbox(bbox_path)
            if bbox is None:
                continue
            samples.append((img_path, bbox))
    return samples


def _double_clahe(img: np.ndarray, clip1: float, tile1: int, clip2: float, tile2: int) -> np.ndarray:
    clahe1 = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(tile1, tile1)).apply(img)
    clahe2 = cv2.createCLAHE(clipLimit=clip2, tileGridSize=(tile2, tile2)).apply(clahe1)
    return clahe2


def _otsu_mask(img: np.ndarray, blur_k: int) -> np.ndarray:
    if blur_k <= 0 or blur_k % 2 == 0:
        blur_k = 1
    blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _bbox_from_bitmask(mask_path: Path) -> tuple[int, int, int, int] | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if not xs.size or not ys.size:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def export_yolo(
    samples: list[tuple[Path, tuple[int, int, int, int]]],
    out_root: Path,
    val_ratio: float,
    seed: int,
    image_key: str,
    clip1: float,
    clip2: float,
    tile1: int,
    tile2: int,
    blur_k: int,
    preview_count: int,
) -> None:
    random.seed(seed)
    random.shuffle(samples)
    split = int(len(samples) * (1 - val_ratio))
    splits = [("train", samples[:split]), ("val", samples[split:])]
    preview = samples[:preview_count] if preview_count > 0 else []

    for split_name, rows in splits:
        img_dir = out_root / split_name / "images"
        lbl_dir = out_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, bbox) in enumerate(rows):
            stem = img_path.parent.parent.name + "_" + img_path.parent.name + f"_{idx}"
            target_img = img_dir / f"{stem}.png"
            target_lbl = lbl_dir / f"{stem}.txt"

            if image_key == "circular_synth_otsu":
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                clahe2 = _double_clahe(gray, clip1, tile1, clip2, tile2)
                proc = _otsu_mask(clahe2, blur_k)
                img = proc
                cv2.imwrite(str(target_img), cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
            elif image_key == "circular_synth_clahe2":
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                clahe2 = _double_clahe(gray, clip1, tile1, clip2, tile2)
                img = clahe2
                cv2.imwrite(str(target_img), cv2.cvtColor(clahe2, cv2.COLOR_GRAY2BGR))
            else:
                shutil.copy2(img_path, target_img)
                img = cv2.imread(str(target_img), cv2.IMREAD_GRAYSCALE)
            if img is None:
                target_img.unlink(missing_ok=True)
                continue
            h, w = img.shape[:2]
            cx, cy, bw, bh = _norm_yolo(*bbox, w, h)
            with open(target_lbl, "w", encoding="utf-8") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    if preview:
        img_dir = out_root / "preview" / "images"
        lbl_dir = out_root / "preview" / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx, (img_path, bbox) in enumerate(preview):
            stem = img_path.parent.parent.name + "_" + img_path.parent.name + f"_{idx}"
            target_img = img_dir / f"{stem}.png"
            target_lbl = lbl_dir / f"{stem}.txt"

            if image_key == "circular_synth_otsu":
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                clahe2 = _double_clahe(gray, clip1, tile1, clip2, tile2)
                proc = _otsu_mask(clahe2, blur_k)
                img = proc
                cv2.imwrite(str(target_img), cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
            elif image_key == "circular_synth_clahe2":
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                clahe2 = _double_clahe(gray, clip1, tile1, clip2, tile2)
                img = clahe2
                cv2.imwrite(str(target_img), cv2.cvtColor(clahe2, cv2.COLOR_GRAY2BGR))
            else:
                shutil.copy2(img_path, target_img)
                img = cv2.imread(str(target_img), cv2.IMREAD_GRAYSCALE)
            if img is None:
                target_img.unlink(missing_ok=True)
                continue
            h, w = img.shape[:2]
            cx, cy, bw, bh = _norm_yolo(*bbox, w, h)
            with open(target_lbl, "w", encoding="utf-8") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export neck bbox dataset to YOLO format.")
    parser.add_argument("--input-root", type=Path, default=Path("outputs/dataset_synth_headneck"),
                        help="Root directory containing DeepDRR outputs with bbox_labels.json")
    parser.add_argument("--output-root", type=Path, default=Path("prepared/neck_bbox_yolo"),
                        help="Destination directory for YOLO train/val splits")
    parser.add_argument("--image-key", choices=list(IMAGE_CHOICES.keys()), default="otsu_overlay",
                        help="Which image to use as detector input. Use circular_synth_otsu to apply CLAHE2+blur+Otsu.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data for validation split")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling before split")
    parser.add_argument("--clip-limit1", type=float, default=2.0, help="CLAHE clipLimit stage 1 (for circular_synth_otsu).")
    parser.add_argument("--clip-limit2", type=float, default=2.0, help="CLAHE clipLimit stage 2 (for circular_synth_otsu).")
    parser.add_argument("--tile-size1", type=int, default=8, help="CLAHE tile size stage 1 (for circular_synth_otsu).")
    parser.add_argument("--tile-size2", type=int, default=8, help="CLAHE tile size stage 2 (for circular_synth_otsu).")
    parser.add_argument("--blur-kernel", type=int, default=5, help="Gaussian blur kernel for Otsu (for circular_synth_otsu).")
    parser.add_argument("--preview-count", type=int, default=0, help="Optional number of samples to copy into preview/ split.")
    parser.add_argument(
        "--use-circular-bitmask",
        action="store_true",
        help="When exporting circular_* images, derive bbox from deepdrr/label_bitmask_circular_synth.png instead of bbox_labels.json.",
    )
    args = parser.parse_args()

    if args.image_key not in IMAGE_CHOICES:
        parser.error(f"Unknown image-key {args.image_key}")

    samples = collect_samples(args.input_root, args.image_key, use_circular_bitmask=args.use_circular_bitmask)
    if not samples:
        raise SystemExit("No samples found under input-root; ensure bbox_labels.json exists.")
    export_yolo(
        samples,
        args.output_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_key=args.image_key,
        clip1=args.clip_limit1,
        clip2=args.clip_limit2,
        tile1=args.tile_size1,
        tile2=args.tile_size2,
        blur_k=args.blur_kernel,
        preview_count=args.preview_count,
    )
    print(f"Wrote YOLO dataset to {args.output_root} (classes: ['neck'])")


if __name__ == "__main__":
    main()
