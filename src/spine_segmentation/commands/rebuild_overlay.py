from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import click
import cv2
import numpy as np

from .root import cli
from ..core.label_colors import label_to_color


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return boolean mask containing the largest connected component."""
    if mask.dtype != np.uint8:
        work = mask.astype(np.uint8)
    else:
        work = mask
    num, labels = cv2.connectedComponents(work, connectivity=4)
    if num <= 1:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    max_idx = int(counts.argmax())
    return labels == max_idx


@cli.command("rebuild-overlay")
@click.argument("sample_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--out-path", type=click.Path(path_type=Path), default=None,
              help="Where to save the blended overlay (defaults to overlay_rebuilt.png inside the sample directory)")
@click.option("--mask-out", type=click.Path(path_type=Path), default=None,
              help="Optional path to save the per-label colour mask")
@click.option("--alpha", type=float, default=0.45, show_default=True,
              help="Blend factor for colour mask vs grayscale image")
@click.option("--ignore-label-map/--use-label-map", default=False, show_default=True,
              help="Force JSON-based reconstruction even if mask_labels.png exists")
def rebuild_overlay(sample_dir: Path, out_path: Path | None, mask_out: Path | None, alpha: float, ignore_label_map: bool) -> None:
    """Reconstruct a colour overlay using only image.png, mask.png, and labels.json."""

    sample_dir = sample_dir.resolve()
    img_path = sample_dir / "image.png"
    mask_path = sample_dir / "mask.png"
    labels_path = sample_dir / "labels.json"

    if not img_path.exists() or not mask_path.exists() or not labels_path.exists():
        raise click.ClickException("Sample directory must contain image.png, mask.png, and labels.json")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise click.ClickException(f"Failed to read {img_path}")

    import json
    with open(labels_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    labels_meta: Dict[str, Dict[str, object]] = meta.get("labels", {})

    h, w = img.shape
    label_map_path = sample_dir / "mask_labels.png"
    use_label_map = (not ignore_label_map) and label_map_path.exists()

    if use_label_map:
        label_map = cv2.imread(str(label_map_path), cv2.IMREAD_UNCHANGED)
        if label_map is None:
            raise click.ClickException(f"Failed to read {label_map_path}")
        if label_map.shape[:2] != (h, w):
            raise click.ClickException("mask_labels.png shape does not match image.png")
        if label_map.dtype != np.uint8:
            label_map = label_map.astype(np.uint8)
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        unique = [int(v) for v in np.unique(label_map) if v > 0]
        for lid in unique:
            color_mask[label_map == lid] = label_to_color(lid)
    else:
        union = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if union is None:
            raise click.ClickException(f"Failed to read {mask_path}")
        union_bool = union > 0

        label_map = np.zeros((h, w), dtype=np.uint8)
        assigned = np.zeros((h, w), dtype=bool)
        centroids: list[Tuple[float, float, int]] = []

        for lid_str, info in labels_meta.items():
            try:
                lid = int(lid_str)
            except ValueError:
                continue
            if not info.get("present"):
                continue
            bbox = info.get("bbox_2d_index")
            centroid = info.get("centroid_2d_index")
            if not bbox:
                continue
            y0, x0, y1, x1 = [int(v) for v in bbox]
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(h - 1, y1)
            x1 = min(w - 1, x1)
            roi_union = union_bool[y0:y1 + 1, x0:x1 + 1]
            if not roi_union.any():
                continue

            roi_unassigned = ~assigned[y0:y1 + 1, x0:x1 + 1]
            candidate = np.logical_and(roi_union, roi_unassigned)

            if candidate.any():
                candidate = _largest_component(candidate.astype(np.uint8)).astype(bool)
                label_map[y0:y1 + 1, x0:x1 + 1][candidate] = np.uint8(lid)
                assigned[y0:y1 + 1, x0:x1 + 1][candidate] = True
            if centroid:
                cy, cx = float(centroid[0]), float(centroid[1])
                centroids.append((cy, cx, lid))

        remaining = np.logical_and(union_bool, ~assigned)
        if centroids and remaining.any():
            centroid_arr = np.array([[c[0], c[1]] for c in centroids], dtype=np.float32)
            lids_arr = np.array([c[2] for c in centroids], dtype=np.uint8)
            ys, xs = np.where(remaining)
            pts = np.stack([ys, xs], axis=1).astype(np.float32)
            diff = pts[:, None, :] - centroid_arr[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            nearest = lids_arr[np.argmin(dist2, axis=1)]
            label_map[ys, xs] = nearest

        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        unique = [int(v) for v in np.unique(label_map) if v > 0]
        for lid in unique:
            color_mask[label_map == lid] = label_to_color(lid)

    if mask_out is not None:
        mask_out = Path(mask_out).resolve()
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_out), color_mask):
            raise click.ClickException(f"Failed to write colour mask to {mask_out}")

    if out_path is None:
        out_path = sample_dir / "overlay_rebuilt.png"
    else:
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(color_mask, alpha, img_rgb, 1.0 - alpha, 0.0)

    if not cv2.imwrite(str(out_path), overlay):
        raise click.ClickException(f"Failed to write overlay to {out_path}")

    click.echo(f"✅ Rebuilt overlay at {out_path}")
    if mask_out is not None:
        click.echo(f"   🎨 Colour mask saved to {mask_out}")
