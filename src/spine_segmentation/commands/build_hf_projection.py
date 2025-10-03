from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import json

import click
import cv2
import numpy as np
import scipy.ndimage as ndi

from .root import cli
from ..core.io import find_pairs, load_nii


def _labels_from_spec(spec: str) -> List[int]:
    out: list[int] = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def _rotate_volume(arr: np.ndarray, yaw: float | None, pitch: float | None, roll: float | None, order: int) -> np.ndarray:
    out = arr
    if yaw and abs(float(yaw)) > 1e-3:
        out = ndi.rotate(out, angle=float(yaw), axes=(1, 2), reshape=True, order=order, mode="nearest")
    if pitch and abs(float(pitch)) > 1e-3:
        out = ndi.rotate(out, angle=float(pitch), axes=(0, 2), reshape=True, order=order, mode="nearest")
    if roll and abs(float(roll)) > 1e-3:
        out = ndi.rotate(out, angle=float(roll), axes=(0, 1), reshape=True, order=order, mode="nearest")
    return out


@cli.command("build-hf-projection")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=0, show_default=True, help="0 = all")
@click.option("--labels", default="1-7", show_default=True, help="Label IDs to include")
@click.option("--plane", type=click.Choice(["sag", "cor", "ax"], case_sensitive=False), default="sag", show_default=True)
@click.option("--yaw", type=float, default=0.0, show_default=True)
@click.option("--pitch", type=float, default=90.0, show_default=True)
@click.option("--roll", type=float, default=0.0, show_default=True)
@click.option("--height", type=int, default=512, show_default=True, help="Output image height in pixels")
@click.option("--preserve-aspect/--no-preserve-aspect", default=False, show_default=True,
              help="Scale width to match the CT aspect ratio instead of forcing a square output")
@click.option("--native-resolution/--no-native-resolution", default=False, show_default=True,
              help="Skip resizing so the projection keeps the CT's native pixel grid")
@click.option("--ct-window-lo", type=float, default=-1000.0, show_default=True)
@click.option("--ct-window-hi", type=float, default=1000.0, show_default=True)
@click.option("--slab-mm", type=float, default=12.0, show_default=True, help="Thickness along projection axis. 0 = full volume")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--override-existing/--no-override-existing", default=False, show_default=True)
def build_hf_projection(data_root: Path, subset: str, limit_cases: int, labels: str, plane: str,
                        yaw: float, pitch: float, roll: float, height: int,
                        preserve_aspect: bool, native_resolution: bool,
                        slab_mm: float, ct_window_lo: float, ct_window_hi: float,
                        out_dir: Path, override_existing: bool) -> None:
    """Generate high-fidelity DRR-style projections that preserve CT aspect ratios."""

    lids = _labels_from_spec(labels)
    pairs = find_pairs(data_root, subset, limit=(None if limit_cases == 0 else limit_cases))
    if not pairs:
        raise click.ClickException("No CT volumes found for the given subset.")

    out_dir = out_dir.resolve()
    images_dir = out_dir / "images"
    labels_dir = out_dir / "mask_labels"
    json_dir = out_dir / "labels-json"
    for d in (images_dir, labels_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    rows: list[tuple[str, str]] = []

    axis_map = {"sag": 2, "cor": 1, "ax": 0}
    proj_axis = axis_map[plane.lower()]

    for ct_path, seg_path in pairs:
        if seg_path is None:
            click.echo(f"Skipping {ct_path.name}: no segmentation")
            continue

        base_name = ct_path.stem
        click.echo(f"Processing {base_name}")

        vol_hu, spacing, _ = load_nii(ct_path)
        seg_raw, _, _ = load_nii(seg_path)

        # Window and normalize HU to [0,1]
        vol = np.clip(vol_hu, ct_window_lo, ct_window_hi)
        vol = (vol - ct_window_lo) / (ct_window_hi - ct_window_lo + 1e-6)

        vol = _rotate_volume(vol, yaw, pitch, roll, order=1)
        seg_rot = _rotate_volume(seg_raw, yaw, pitch, roll, order=0)

        L = vol.shape[proj_axis]
        den = np.clip(vol, 0.0, 1.0)

        if slab_mm > 0:
            spacing_axis_mm = float(spacing[proj_axis])
            slab_vox = max(1, int(round(slab_mm / max(spacing_axis_mm, 1e-6))))
            half = slab_vox // 2
            center = L // 2
            i0 = max(0, center - half)
            i1 = min(L, i0 + slab_vox)
        else:
            i0, i1 = 0, L

        slicer = [slice(None)] * 3
        slicer[proj_axis] = slice(i0, i1)
        part = den[tuple(slicer)]
        img = part.sum(axis=proj_axis)
        img = img / (img.max() + 1e-6)

        silhouettes: dict[int, np.ndarray] = {}
        mask = np.zeros_like(img, dtype=np.uint8)
        for lid in lids:
            seg_slicer = [slice(None)] * 3
            seg_slicer[proj_axis] = slice(i0, i1)
            slab = (seg_rot[tuple(seg_slicer)] == lid)
            silhouette = slab.any(axis=proj_axis)
            silhouettes[lid] = silhouette
            mask[silhouette] = np.uint8(lid)

        orig_h, orig_w = img.shape
        axes2d = tuple(sorted(set([0, 1, 2]) - {proj_axis}))
        row_spacing = float(spacing[axes2d[0]])
        col_spacing = float(spacing[axes2d[1]])
        row_mm = orig_h * row_spacing
        col_mm = orig_w * col_spacing

        if native_resolution:
            target_h = orig_h
            target_w = orig_w
            new_row_spacing = row_spacing
            new_col_spacing = col_spacing
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            resized_masks = {lid: sil.astype(np.uint8) for lid, sil in silhouettes.items()}
        else:
            if height <= 0:
                raise click.ClickException("--height must be > 0 unless --native-resolution is enabled")
            target_h = int(height)
            if preserve_aspect:
                aspect = orig_w / max(orig_h, 1)
                target_w = int(max(1, round(target_h * aspect)))
            else:
                target_w = target_h
            new_row_spacing = row_mm / max(target_h, 1)
            new_col_spacing = col_mm / max(target_w, 1)
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            if img_u8.shape[0] != target_h or img_u8.shape[1] != target_w:
                img_u8 = cv2.resize(img_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)

            resized_masks = {}
            for lid, sil in silhouettes.items():
                resized = cv2.resize(sil.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                resized_masks[lid] = resized

        label_map = np.zeros((target_h, target_w), dtype=np.uint8)
        for lid, resized in resized_masks.items():
            label_map = np.where(resized > 0, np.uint8(lid), label_map)
        mask_u8 = label_map

        img_path = images_dir / f"{base_name}.png"
        mask_path = labels_dir / f"{base_name}.png"

        if not override_existing and img_path.exists() and mask_path.exists():
            click.echo(f"  Skipping existing {base_name}")
            rows.append((str(img_path), str(mask_path)))
            continue

        if not cv2.imwrite(str(img_path), img_u8):
            raise click.ClickException(f"Failed to save image {img_path}")
        if not cv2.imwrite(str(mask_path), mask_u8):
            raise click.ClickException(f"Failed to save mask {mask_path}")
        json_out_path = json_dir / f"{base_name}.json"
        labels_meta: dict[str, dict[str, float | int | list]] = {}
        for lid, resized in resized_masks.items():
            present = bool(resized.max())
            info: dict[str, float | int | list | bool] = {"present": present}
            if present:
                ys, xs = np.where(resized > 0)
                cy = float(ys.mean())
                cx = float(xs.mean())
                info["centroid_2d_index"] = [cy, cx]
                info["centroid_2d_mm"] = [cy * new_row_spacing, cx * new_col_spacing]
                y0 = int(ys.min())
                y1 = int(ys.max()) + 1
                x0 = int(xs.min())
                x1 = int(xs.max()) + 1
                info["bbox_2d_index"] = [y0, x0, y1, x1]
                info["bbox_2d_hw_mm"] = [float(y1 - y0) * new_row_spacing,
                                            float(x1 - x0) * new_col_spacing]
                area_px = int(len(xs))
                info["area_px"] = area_px
                info["area_mm2"] = area_px * new_row_spacing * new_col_spacing
            labels_meta[str(lid)] = info

        meta = {
            "case": base_name,
            "subset": subset,
            "image": str(img_path),
            "mask_labels": str(mask_path),
            "labels": labels_meta,
            "image_size_index": [target_h, target_w],
            "pixel_spacing": [new_row_spacing, new_col_spacing],
            "slab_mm": slab_mm,
            "slab_range_index": [i0, i1],
            "slab_range_mm": [i0 * float(spacing[proj_axis]), i1 * float(spacing[proj_axis])],
            "proj_axis": proj_axis,
            "proj_axis_spacing_mm": float(spacing[proj_axis]),
        }

        with open(json_out_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        rows.append((str(img_path), str(mask_path)))

    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "mask_labels"])
        writer.writerows(rows)

    click.echo(f"✅ High-fidelity projections written to {out_dir}")
