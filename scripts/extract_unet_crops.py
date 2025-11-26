#!/usr/bin/env python3
"""
Collect cropped/letterboxed DeepDRR panels and their label bitmasks into a flat
directory for UNet training/inspection.

By default grabs:
- image: deepdrr/otsu_mask.png  (CLAHE2 + blur + Otsu on the cropped DRR)
- mask : deepdrr/label_bitmask_cropped_letterboxed.png
Outputs a flat structure with images/, masks/, and a manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract cropped DeepDRR panels + bitmasks for UNet.")
    ap.add_argument("--input-root", type=Path, required=True, help="Root containing DeepDRR runs (e.g., outputs/dataset_synth_headneck_2).")
    ap.add_argument("--out-dir", type=Path, default=Path("prepared/unet_cropped_flat"), help="Destination root.")
    ap.add_argument(
        "--image-kind",
        choices=["otsu_mask", "orig_gray", "clahe2", "otsu_overlay"],
        default="otsu_mask",
        help="Which DeepDRR image to pair with the bitmask.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on samples (0 = all).")
    args = ap.parse_args()

    img_sub = {
        "otsu_mask": "otsu_mask.png",
        "orig_gray": "orig_gray.png",
        "clahe2": "clahe2.png",
        "otsu_overlay": "otsu_overlay.png",
    }[args.image_kind]

    out_img = args.out_dir / "images"
    out_mask = args.out_dir / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"

    rows = []
    bitmasks = list(args.input_root.rglob("label_bitmask_cropped_letterboxed.png"))
    if args.limit > 0:
        bitmasks = bitmasks[: args.limit]

    for bm in bitmasks:
        deepdrr_dir = bm.parent
        run_dir = deepdrr_dir.parent  # e.g., off_m0p2
        case_dir = run_dir.parent     # e.g., HN_P001
        img_path = deepdrr_dir / img_sub
        if not img_path.exists():
            continue

        stem = f"{case_dir.name}_{run_dir.name}"
        dest_img = out_img / f"{stem}.png"
        dest_mask = out_mask / f"{stem}.png"
        shutil.copy2(img_path, dest_img)
        shutil.copy2(bm, dest_mask)
        rows.append((dest_img.as_posix(), dest_mask.as_posix(), case_dir.name, run_dir.name))

    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask_labels", "case", "angle"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} pairs to {manifest_path}")
    print(f"Images: {out_img}")
    print(f"Masks:  {out_mask}")


if __name__ == "__main__":
    main()
