#!/usr/bin/env python3
"""
CLI curator: review cropped neck panels and add good ones to the YOLO neck bbox
dataset using CLAHE2 + Otsu preprocessing. Shows an ASCII preview (no GUI).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def double_clahe(img: np.ndarray, clip1: float, tile1: int, clip2: float, tile2: int) -> tuple[np.ndarray, np.ndarray]:
    clahe1 = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(tile1, tile1)).apply(img)
    clahe2 = cv2.createCLAHE(clipLimit=clip2, tileGridSize=(tile2, tile2)).apply(clahe1)
    return clahe1, clahe2


def otsu_mask(img: np.ndarray, blur_k: int) -> np.ndarray:
    if blur_k <= 0 or blur_k % 2 == 0:
        blur_k = 1
    blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def save_sample(
    crop_path: Path,
    dest_root: Path,
    split: str,
    prefix: str,
    clip1: float,
    tile1: int,
    clip2: float,
    tile2: int,
    blur_k: int,
) -> tuple[bool, str]:
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        return False, f"Could not read {crop_path}"
    _, clahe2 = double_clahe(crop, clip1, tile1, clip2, tile2)
    mask = otsu_mask(clahe2, blur_k)
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return False, f"No foreground pixels in {crop_path}"

    h, w = mask.shape[:2]
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1 + 1) / 2) / w
    cy = ((y0 + y1 + 1) / 2) / h
    bw = (x1 - x0 + 1) / w
    bh = (y1 - y0 + 1) / h

    dest_img_dir = dest_root / split / "images"
    dest_lbl_dir = dest_root / split / "labels"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)

    dest_stem = f"{prefix}{crop_path.stem.replace('_crop', '')}"
    dest_img_path = dest_img_dir / f"{dest_stem}.png"
    dest_lbl_path = dest_lbl_dir / f"{dest_stem}.txt"

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dest_img_path), mask_bgr)
    with open(dest_lbl_path, "w", encoding="utf-8") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return True, f"Added {dest_img_path} and {dest_lbl_path}"


def ascii_preview(img: np.ndarray, width: int, charset: str = " .:-=+*#%@") -> str:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    if h == 0 or w == 0 or width <= 0:
        return ""
    scale = width / float(w)
    new_h = max(1, int(h * scale * 0.55))  # compensate for character aspect ratio
    small = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    normalized = np.clip(small.astype(np.float32) / 255.0, 0.0, 1.0)
    idxs = (normalized * (len(charset) - 1)).round().astype(int)
    lines = ["".join(charset[idx] for idx in row) for row in idxs]
    return "\n".join(lines)


def natural_key(path: Path) -> list[int | str]:
    """Sort path stems in human/numeric order (e.g., f2 before f10)."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p for p in parts]


def read_key(prompt: str = "") -> str:
    """Read a single keypress without requiring Enter."""
    print(prompt, end="", flush=True)
    try:
        import msvcrt  # type: ignore

        ch = msvcrt.getch()
        if isinstance(ch, bytes):
            ch = ch.decode(errors="ignore")
    except ImportError:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print()
    return ch


def main():
    ap = argparse.ArgumentParser(description="Review cropped panels and add good ones to neck_bbox_yolo.")
    ap.add_argument("--side-dir", type=Path, default=Path("outputs/frames_cropped_iter/side_by_side"))
    ap.add_argument("--crop-dir", type=Path, default=Path("outputs/frames_cropped_iter/crops"))
    ap.add_argument("--dest-root", type=Path, default=Path("prepared/neck_bbox_yolo"))
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--prefix", default="real_")
    ap.add_argument("--clip-limit1", type=float, default=2.0)
    ap.add_argument("--clip-limit2", type=float, default=2.0)
    ap.add_argument("--tile-size1", type=int, default=8)
    ap.add_argument("--tile-size2", type=int, default=8)
    ap.add_argument("--blur-kernel", type=int, default=5)
    ap.add_argument(
        "--preview-width",
        type=int,
        default=96,
        help="ASCII preview width in characters (0 disables preview).",
    )
    args = ap.parse_args()

    side_images = sorted(list(args.side_dir.glob("*.png")), key=natural_key)
    if not side_images:
        raise SystemExit(f"No side-by-side panels found in {args.side_dir}")

    total = len(side_images)
    for idx, side_path in enumerate(side_images, start=1):
        stem = side_path.stem.replace("_side", "")
        crop_path = args.crop_dir / f"{stem}_crop.png"
        if not crop_path.exists():
            print(f"Missing crop for {side_path.name}, expected {crop_path}")
            continue

        panel = cv2.imread(str(side_path))
        if panel is None:
            print(f"Could not read {side_path}")
            continue

        print(f"\n[{idx}/{total}] {side_path.name} -> {crop_path.name}")
        preview = ascii_preview(panel, width=args.preview_width)
        if preview:
            print(preview)
        else:
            print("(preview disabled; set --preview-width to see ASCII art)")

        while True:
            action = read_key("Action [a=add, s=skip, q=quit, h=help]: ").strip().lower()
            if action in {"q", "quit", "exit"}:
                return
            if action in {"a", "add", "y", "yes"}:
                ok, msg = save_sample(
                    crop_path,
                    dest_root=args.dest_root,
                    split=args.split,
                    prefix=args.prefix,
                    clip1=args.clip_limit1,
                    tile1=args.tile_size1,
                    clip2=args.clip_limit2,
                    tile2=args.tile_size2,
                    blur_k=args.blur_kernel,
                )
                print(msg)
                break
            if action in {"s", "skip", "n", "no", ""}:
                print(f"Skipped {stem}")
                break
            if action in {"h", "?", "help"}:
                print("a/add: accept and write mask+label | s/skip/Enter: ignore | q: quit")
            else:
                print("Unknown option, use 'a', 's', or 'q'.")


if __name__ == "__main__":
    main()
