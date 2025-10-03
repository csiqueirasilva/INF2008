from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import click
import cv2
import numpy as np
import pandas as pd

from .root import cli
from ..core.label_colors import label_to_color


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image", "mask_labels"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Manifest {path} missing columns: {missing}")
    return df


def _derive_json_path(image_path: Path) -> Path:
    stem = image_path.stem
    angle_dir = image_path.parent.parent
    json_dir = angle_dir / "labels-json"
    return json_dir / f"{stem}.json"


def _crop_center(img: np.ndarray, center: Tuple[float, float], size: int) -> np.ndarray:
    half = size // 2
    pad_y = pad_x = half + 2
    img_pad = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0)
    cy = int(round(center[0])) + pad_y
    cx = int(round(center[1])) + pad_x
    y0 = cy - half
    y1 = y0 + size
    x0 = cx - half
    x1 = x0 + size
    return img_pad[y0:y1, x0:x1]


def _apply_circular_mask(img: np.ndarray, radius: int) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    out = img.copy()
    if img.ndim == 2:
        out[~circle] = 0
    else:
        out[~circle] = 0
    return out


def _recolor(mask_labels: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask_labels.shape, 3), dtype=np.uint8)
    unique = [int(v) for v in np.unique(mask_labels) if v > 0]
    for lid in unique:
        color[mask_labels == lid] = label_to_color(lid)
    return color


@cli.command("prepare-unet-tiles")
@click.option("--csv", "csv_path", type=click.Path(path_type=Path), required=False,
              help="Manifest produced by build_pseudo_dataset.sh")
@click.option("--image", "images", type=click.Path(path_type=Path), multiple=True,
              help="Direct path(s) to projection image(s) (bypasses CSV)")
@click.option("--mask", "masks", type=click.Path(path_type=Path), multiple=True,
              help="Matching mask_label image(s) for --image")
@click.option("--annotation", "annotations", type=click.Path(path_type=Path), multiple=True,
              help="Optional labels.json path(s); derived automatically when omitted")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True,
              help="Directory where 384x384 tiles will be stored")
@click.option("--size", type=int, default=384, show_default=True,
              help="Output tile size (square)")
@click.option("--label-id", type=click.INT, default=None, show_default=True,
              help="Label ID whose centroid anchors the crop; if omitted, use average of available labels")
@click.option("--radius", type=int, default=192, show_default=True,
              help="Radius (pixels) for circular mask (defaults to size/2)")
@click.option("--zoom", type=float, default=1.0, show_default=True,
              help="Zoom factor (>1 crops tighter around the centroid before resizing back to output size)")
@click.option("--write-overlay/--no-write-overlay", default=True, show_default=True,
              help="Optionally write recoloured overlay for QA")
def prepare_unet_tiles(csv_path: Path | None, images: tuple[Path, ...], masks: tuple[Path, ...],
                       annotations: tuple[Path, ...], out_dir: Path, size: int, label_id: int,
                       radius: int, zoom: float, write_overlay: bool) -> None:
    """Crop 384x384 tiles centred on a vertebra centroid and apply circular vignette."""

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[Path, Path, Path]] = []

    if csv_path is not None:
        csv_path = csv_path.expanduser().resolve()
        df = _load_manifest(csv_path)
        for image_path_str, mask_path_str in zip(df["image"], df["mask_labels"]):
            image_path = Path(image_path_str).expanduser().resolve()
            mask_path = Path(mask_path_str).expanduser().resolve()
            json_path = _derive_json_path(image_path)
            entries.append((image_path, mask_path, json_path))

    if images or masks or annotations:
        if len(images) != len(masks):
            raise click.ClickException("--image and --mask must be provided the same number of times")
        if annotations and len(annotations) not in (0, len(images)):
            raise click.ClickException("--annotation count must match --image when supplied")
        for idx, (img_path_raw, mask_path_raw) in enumerate(zip(images, masks)):
            image_path = img_path_raw.expanduser().resolve()
            mask_path = mask_path_raw.expanduser().resolve()
            if annotations:
                json_path = annotations[idx].expanduser().resolve()
            else:
                json_path = _derive_json_path(image_path)
            entries.append((image_path, mask_path, json_path))

    if not entries:
        raise click.ClickException("Provide either --csv or at least one --image/--mask pair")

    images_dir = out_dir / "images"
    masks_dir = out_dir / "mask_labels"
    overlays_dir = out_dir / "overlays"
    for d in (images_dir, masks_dir, overlays_dir):
        d.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    skipped = 0

    for image_path, mask_path, json_path in entries:

        if not image_path.exists() or not mask_path.exists() or not json_path.exists():
            click.echo(f"⚠️ Skipping {image_path} (missing files)")
            skipped += 1
            continue

        with open(json_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        labels_meta = meta.get("labels", {})
        if label_id is not None:
            label_info = labels_meta.get(str(label_id))
            if not label_info or not label_info.get("present"):
                click.echo(f"⚠️ Skipping {image_path} (label {label_id} absent)")
                skipped += 1
                continue
            centroid = label_info.get("centroid_2d_index")
            if centroid is None:
                click.echo(f"⚠️ Skipping {image_path} (missing centroid for label {label_id})")
                skipped += 1
                continue
        else:
            centroids = [info.get("centroid_2d_index") for info in labels_meta.values()
                         if info.get("present") and info.get("centroid_2d_index")]
            if not centroids:
                click.echo(f"⚠️ Skipping {image_path} (no centroids available)")
                skipped += 1
                continue
            cy = sum(c[0] for c in centroids) / len(centroids)
            cx = sum(c[1] for c in centroids) / len(centroids)
            centroid = (cy, cx)

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask_labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask_labels is None:
            click.echo(f"⚠️ Skipping {image_path} (read error)")
            skipped += 1
            continue

        crop_size = size
        if zoom > 1.0:
            crop_size = max(8, int(round(size / zoom)))

        tile_img = _crop_center(img, (centroid[0], centroid[1]), crop_size)
        tile_mask = _crop_center(mask_labels, (centroid[0], centroid[1]), crop_size)

        if crop_size != size:
            tile_img = cv2.resize(tile_img, (size, size), interpolation=cv2.INTER_LINEAR)
            tile_mask = cv2.resize(tile_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        effective_radius = radius if radius > 0 else size // 2
        tile_img = _apply_circular_mask(tile_img, effective_radius)
        tile_mask = _apply_circular_mask(tile_mask, effective_radius)

        stem = image_path.stem
        out_img_path = images_dir / f"{stem}.png"
        out_mask_path = masks_dir / f"{stem}.png"

        cv2.imwrite(str(out_img_path), tile_img)
        cv2.imwrite(str(out_mask_path), tile_mask)

        if write_overlay:
            color = _recolor(tile_mask)
            overlay = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(overlay, 0.55, color, 0.45, 0.0)
            cv2.imwrite(str(overlays_dir / f"{stem}.png"), overlay)

        records.append({
            "image": str(out_img_path),
            "mask_labels": str(out_mask_path),
        })

    if not records:
        raise click.ClickException("No tiles were generated. Check that centroids exist for the requested label.")

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    click.echo(f"✅ Generated {len(records)} tiles (skipped {skipped}). Manifest written to {manifest_path}")
