#!/usr/bin/env python3
"""
Overlay bbox_labels.json onto clahe2.png (or circular-synth CLAHE2) to verify alignment.

For each DeepDRR run dir containing bbox_labels.json, reads deepdrr/clahe2.png,
draws the bbox, and writes overlay + crop to an output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_bbox(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bbox = data.get("bbox")
    if not bbox:
        return None
    return int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"])


def main():
    ap = argparse.ArgumentParser(description="Overlay bbox_labels.json onto clahe2.png.")
    ap.add_argument("--runs", nargs="+", required=True, help="Paths to DeepDRR run directories (contain deepdrr/clahe2.png and bbox_labels.json).")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/clahe2_bbox_preview"), help="Where to write overlays and crops.")
    ap.add_argument("--image-name", default="clahe2.png", help="Image file under deepdrr/ to annotate (default: clahe2.png).")
    ap.add_argument("--bitmask-circular", default="label_bitmask_circular_synth.png",
                    help="If present, derive bbox from this bitmask instead of bbox_labels.json.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of samples (0 = all).")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [Path(r) for r in args.runs]
    if args.limit > 0:
        runs = runs[: args.limit]

    count = 0
    for run_dir in runs:
        img_path = run_dir / "deepdrr" / args.image_name
        bbox_path = run_dir / "bbox_labels.json"
        bm_circ = run_dir / "deepdrr" / args.bitmask_circular
        if not img_path.exists():
            continue

        bbox = None
        if bm_circ.exists():
            bm = cv2.imread(str(bm_circ), cv2.IMREAD_UNCHANGED)
            if bm is not None:
                ys, xs = np.nonzero(bm)
                if xs.size and ys.size:
                    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if bbox is None and bbox_path.exists():
            bbox = load_bbox(bbox_path)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        crop = img[y0 : y1 + 1, x0 : x1 + 1]

        stem = f"{run_dir.parent.name}_{run_dir.name}"
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{stem}_crop.png"), crop)
        count += 1

    print(f"Saved {count} overlays/crops to {out_dir}")


if __name__ == "__main__":
    main()
