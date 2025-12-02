#!/usr/bin/env python3
"""
Prepare class-specific manifests for per-vertebra UNet training (e.g., C2, C3, C4)
from the crops produced by extract_vertebra_crops.py.

Reads the master manifest (containing letterbox_image/mask paths) and writes one
manifest per requested class. You can then train a binary UNet per class using
`spine train-unet --csv <class_manifest> ...`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


DEF_CLASSES = ["C2", "C3", "C4"]


def read_rows(manifest: Path) -> list[dict]:
    with manifest.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def filter_by_class(rows: Iterable[dict], cls: str) -> list[dict]:
    cls = cls.upper()
    out = []
    for row in rows:
        if row.get("class_name", "").upper() != cls:
            continue
        out.append(row)
    return out


def write_manifest(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "mask_labels", "case", "angle", "class_name", "class_id"]
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "image": row["letterbox_image"],
                    "mask_labels": row["letterbox_mask"],
                    "case": row.get("case", ""),
                    "angle": row.get("angle", ""),
                    "class_name": row.get("class_name", ""),
                    "class_id": row.get("class_id", ""),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare per-vertebra UNet manifests from crop manifest.")
    ap.add_argument("--manifest", type=Path, required=True, help="Master manifest from extract_vertebra_crops.py")
    ap.add_argument("--out-dir", type=Path, default=Path("prepared/vertebra_unet_manifests"), help="Where to write class manifests.")
    ap.add_argument("--classes", default=",".join(DEF_CLASSES), help="Comma-separated classes to include, e.g., C2,C3,C4")
    args = ap.parse_args()

    rows = read_rows(args.manifest)
    if not rows:
        raise SystemExit("No rows found in manifest.")

    classes = [c.strip().upper() for c in args.classes.split(",") if c.strip()]
    for cls in classes:
        cls_rows = filter_by_class(rows, cls)
        if not cls_rows:
            print(f"Skipping {cls}: no samples found.")
            continue
        dest = args.out_dir / cls / "manifest.csv"
        write_manifest(cls_rows, dest)
        print(f"Wrote {len(cls_rows)} rows to {dest}")


if __name__ == "__main__":
    main()
