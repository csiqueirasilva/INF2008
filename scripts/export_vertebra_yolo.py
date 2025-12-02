#!/usr/bin/env python3
"""
Export per-vertebra YOLO boxes from DeepDRR bitmasks (C1–C7).

Reads label_bitmask_cropped_letterboxed.png and/or label_bitmask_circular_synth.png
and writes a 7-class YOLO dataset (C1..C7) with train/val splits.

Defaults expect:
- input_root: outputs/dataset_synth_headneck_2
- image_key: choose which image to pair (e.g., clahe2.png, circular-synth.png)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_CHOICES = {
    "clahe2": Path("clahe2.png"),
    "otsu_overlay": Path("otsu_overlay.png"),
    "circular_synth": Path("circular-synth.png"),
}

DEFAULT_CLASS_LABELS = [1, 2, 4]  # C2, C3, C4 in the thin-slab neck set
DEFAULT_CLASS_NAMES = ["C2", "C3", "C4"]


def bbox_from_class(mask: np.ndarray, lid: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask == lid)
    if not xs.size or not ys.size:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def norm_yolo(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[float, float, float, float]:
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0 + 1) / w
    bh = (y1 - y0 + 1) / h
    return cx, cy, bw, bh


def clean_binary_mask(binary: np.ndarray, min_area: int, open_ksize: int, open_iters: int) -> np.ndarray:
    """
    Denoise a single-label binary mask:
    1) optional morphological opening to erase small blobs/bridges
    2) keep only the largest connected component
    """
    import cv2  # local import to avoid global dependency

    if binary.max() == 0:
        return binary

    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=open_iters)
        if binary.max() == 0:
            return binary

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = areas.argmax() + 1
    if min_area > 0 and areas[max_idx - 1] < min_area:
        return np.zeros_like(binary, dtype="uint8")
    return (labels == max_idx).astype("uint8")


def apply_priority(mask: np.ndarray, class_names: list[str], class_labels: list[int], use_priority: bool) -> np.ndarray:
    """
    Ensure only one class per pixel following a fixed priority C1 > C2 > ... > C7.
    Works with the provided class_names/class_labels mapping.
    """
    if not use_priority or len(class_names) <= 1:
        return mask
    name_to_bit = {n: b for n, b in zip(class_names, class_labels)}
    priority = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    out = np.zeros_like(mask, dtype=mask.dtype)
    for pname in priority:
        lid = name_to_bit.get(pname)
        if lid is None:
            continue
        sel = (mask & lid) != 0
        out[(out == 0) & sel] = lid
    return out


def collect_samples(input_root: Path, image_key: str, bitmask_name: str) -> list[tuple[Path, Path]]:
    samples: list[tuple[Path, Path]] = []
    bitmask_name = Path(bitmask_name).name
    for bm_path in input_root.rglob(bitmask_name):
        img_path = bm_path.parent / IMAGE_CHOICES[image_key]
        if not img_path.exists():
            continue
        samples.append((img_path, bm_path))
    return samples


def export(
    samples: list[tuple[Path, Path]],
    out_root: Path,
    val_ratio: float,
    seed: int,
    preview_count: int,
    class_labels: list[int],
    class_names: list[str],
    max_frac: float,
    min_component_area: int,
    open_ksize: int,
    open_iters: int,
    use_priority: bool,
) -> None:
    random.seed(seed)
    random.shuffle(samples)

    preview_samples = samples[:preview_count] if preview_count > 0 else []
    remaining = samples[preview_count:] if preview_count > 0 else samples
    if not remaining:
        raise SystemExit("Not enough samples after taking preview split.")

    split = int(len(remaining) * (1 - val_ratio))
    splits = [("train", remaining[:split]), ("val", remaining[split:])]
    if preview_samples:
        splits.append(("preview", preview_samples))

    skipped_too_big = 0
    kept = 0

    for split_name, rows in splits:
        img_dir = out_root / split_name / "images"
        lbl_dir = out_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir = out_root / "preview_overlays"
        if split_name == "preview":
            overlay_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, bm_path) in enumerate(rows):
            stem = img_path.parent.parent.name + "_" + img_path.parent.name + f"_{idx}"
            dest_img = img_dir / f"{stem}.png"
            dest_lbl = lbl_dir / f"{stem}.txt"

            # copy image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            cv2.imwrite(str(dest_img), img)
            h, w = img.shape[:2]

            bm = cv2.imread(str(bm_path), cv2.IMREAD_UNCHANGED)
            if bm is None:
                continue
            bm = apply_priority(bm, class_names=class_names, class_labels=class_labels, use_priority=use_priority)

            lines = []
            boxes_for_preview: list[tuple[int, int, int, int, int]] = []
            skip_sample = False
            for cls_idx, lid in enumerate(class_labels):
                # work on a per-class binary mask to avoid interfering with other labels
                binary = (bm == lid).astype("uint8")
                if open_ksize > 0 or min_component_area >= 0:
                    binary = clean_binary_mask(
                        binary,
                        min_area=min_component_area if min_component_area >= 0 else 0,
                        open_ksize=open_ksize,
                        open_iters=open_iters,
                    )
                if binary.max() == 0:
                    continue
                ys, xs = np.nonzero(binary)
                if not xs.size or not ys.size:
                    continue
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bbox = (x0, y0, x1, y1)
                bw_frac = (x1 - x0 + 1) / w
                bh_frac = (y1 - y0 + 1) / h
                if max_frac > 0 and (bw_frac > max_frac or bh_frac > max_frac):
                    skip_sample = True
                    break
                boxes_for_preview.append((*bbox, cls_idx))
                cx, cy, bw, bh = norm_yolo(*bbox, w=w, h=h)
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if skip_sample or not lines:
                if skip_sample:
                    skipped_too_big += 1
                continue
            kept += 1
            dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

            if split_name == "preview" and boxes_for_preview:
                overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for x0, y0, x1, y1, cls_idx in boxes_for_preview:
                    rng = np.random.RandomState(cls_idx + 42)
                    color = tuple(int(c) for c in rng.randint(0, 255, size=3))
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(
                        overlay,
                        class_names[cls_idx],
                        (x0, max(0, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imwrite(str(overlay_dir / dest_img.name), overlay)

    if max_frac > 0:
        print(f"Skipped {skipped_too_big} samples with boxes larger than max-frac {max_frac}. Kept {kept}.")


def main():
    ap = argparse.ArgumentParser(description="Export per-vertebra YOLO boxes from bitmasks (C1–C7).")
    ap.add_argument("--input-root", type=Path, default=Path("outputs/dataset_synth_headneck_2"))
    ap.add_argument("--output-root", type=Path, default=Path("prepared/vertebra_yolo"))
    ap.add_argument("--image-key", choices=list(IMAGE_CHOICES.keys()), default="clahe2",
                    help="Which DeepDRR image to use as detector input.")
    ap.add_argument("--bitmask-name", default="label_bitmask_cropped_letterboxed.png",
                    help="Bitmask filename to derive boxes from (e.g., label_bitmask_cropped_letterboxed.png or label_bitmask_circular_synth.png).")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--preview-count", type=int, default=0, help="Optional number of samples to copy into preview split.")
    ap.add_argument("--labels", default="C2,C3,C4",
                    help="Comma-separated class names (subset of C1..C7) matching bitmask bits 1,2,4,8,16,32,64.")
    ap.add_argument("--max-frac", type=float, default=0.35,
                    help="Optional max bbox fraction (width or height) to keep; >0 drops oversized boxes (default: 0.35).")
    ap.add_argument("--min-component-area", type=int, default=50,
                    help="Keep only the largest connected component per class; drop label if area below this (set <0 to disable).")
    ap.add_argument("--open-ksize", type=int, default=3,
                    help="Apply morphological opening (ellipse kernel) of this size before component filtering (set 0 to disable).")
    ap.add_argument("--open-iters", type=int, default=1,
                    help="Iterations for morphological opening.")
    ap.add_argument(
        "--sequential-bits",
        action="store_true",
        help="Map classes to bits in order (1,2,4,...) ignoring absolute label IDs. "
             "Use this when bitmasks were written with guide-labels (e.g., 2-4) so bit 1 == first class.",
    )
    ap.add_argument(
        "--priority-law",
        action="store_true",
        help="Enforce priority C1>C2>...>C7 so only one class remains per pixel; uses the provided class list.",
    )
    args = ap.parse_args()

    samples = collect_samples(args.input_root, args.image_key, args.bitmask_name)
    if not samples:
        raise SystemExit("No samples found; check input root and bitmask/image names.")

    class_names = [c.strip() for c in args.labels.split(",") if c.strip()]
    if args.sequential_bits:
        class_labels = [1 << i for i in range(len(class_names))]
    else:
        name_to_bit = {"C1": 1, "C2": 2, "C3": 4, "C4": 8, "C5": 16, "C6": 32, "C7": 64}
        class_labels = [name_to_bit[n] for n in class_names]

    export(
        samples,
        args.output_root,
        args.val_ratio,
        args.seed,
        args.preview_count,
        class_labels=class_labels,
        class_names=class_names,
        max_frac=args.max_frac,
        min_component_area=args.min_component_area,
        open_ksize=args.open_ksize,
        open_iters=args.open_iters,
        use_priority=args.priority_law,
    )

    data_yaml = args.output_root / "data.yaml"
    data_yaml.write_text("\n".join([
        f"path: {args.output_root}",
        "train: train/images",
        "val: val/images",
        "preview: preview/images",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
    ]))
    print(f"Wrote YOLO dataset to {args.output_root} (classes: {class_names})")


if __name__ == "__main__":
    main()
