#!/usr/bin/env python3
"""
Render YOLO labels as overlays for a dataset split.

Usage:
  poetry run python scripts/render_yolo_overlays.py \
    --root prepared/neck_bbox_yolo_full_preview_5/preview \
    --out-dir prepared/neck_bbox_yolo_full_preview_5/preview_overlays
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

COLOR = (255, 0, 0)  # BGR blue by default


def render_split(root: Path, out_dir: Path, color: tuple[int, int, int]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    img_dir = root / "images"
    lbl_dir = root / "labels"
    for img_path in img_dir.glob("*.png"):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None or not lbl_path.exists():
            continue
        h, w = img.shape[:2]
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x0 = int((cx - bw / 2) * w)
            y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w)
            y1 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.imwrite(str(out_dir / img_path.name), img)
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Render YOLO bbox overlays for a split.")
    ap.add_argument("--root", type=Path, required=True, help="Split root containing images/ and labels/")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to save overlays.")
    ap.add_argument("--color", type=str, default="255,0,0", help="BGR color, e.g., 255,0,0 for blue.")
    args = ap.parse_args()

    color = tuple(int(c) for c in args.color.split(","))
    n = render_split(args.root, args.out_dir, color)
    print(f"Wrote {n} overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
