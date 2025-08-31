from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Dict, Tuple
import click
import numpy as np
import cv2

from .root import cli
from ..core.io import find_pairs, load_nii
import scipy.ndimage as ndi


def _ensure_gray_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _ct_window_to_density(vol_hu: np.ndarray, lo=200, hi=2000) -> np.ndarray:
    v = np.clip(vol_hu, lo, hi)
    v = (v - lo) / (hi - lo + 1e-6)
    return v.astype(np.float32)


def _project_plane(ct_den: np.ndarray, plane: str, power=1.5) -> tuple[np.ndarray, tuple[int,int], tuple[float,float]]:
    """
    Project CT along the appropriate axis for the given plane.
    Returns (img_u8, (rows, cols), (row_spacing_mm, col_spacing_mm)) where spacings correspond
    to the axes that remain after projection (before any resizing/rotation).
    plane:
      - 'sag': drop X (axis=2) => (Z,Y), spacing (s0, s1)
      - 'cor': drop Y (axis=1) => (Z,X), spacing (s0, s2)
      - 'ax' : drop Z (axis=0) => (Y,X), spacing (s1, s2)
    Note: spacing vector must be provided separately by caller.
    """
    plane = plane.lower()
    den = np.maximum(ct_den, 0.0) ** float(power)
    if plane == 'sag':
        att = den.sum(axis=2)  # (Z,Y)
        shape2d_axes = (0, 1)
    elif plane == 'cor':
        att = den.sum(axis=1)  # (Z,X)
        shape2d_axes = (0, 2)
    elif plane == 'ax':
        att = den.sum(axis=0)  # (Y,X)
        shape2d_axes = (1, 2)
    else:
        raise ValueError(f"invalid plane: {plane}")
    att = att / (att.max() + 1e-6)
    img = (att * 255.0).astype(np.uint8)
    return img, shape2d_axes, (0.0, 0.0)  # spacings filled by caller


def _silhouette_labels_2d(lab3d: np.ndarray, lids: list[int], axis=2) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for lid in lids:
        m = (lab3d == int(lid)).astype(np.uint8)
        sil = m.max(axis=axis)  # axis=2 collapses X → (Z,Y)
        out[int(lid)] = (sil > 0).astype(np.uint8) * 255
    return out


def _centroid(mask_u8: np.ndarray) -> Tuple[float, float] | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def _bbox(mask_u8: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def _apply_c_arm_aperture(img: np.ndarray, center_xy: Tuple[float, float], radius: float,
                          edge_softness: float = 0.15, boost_inside: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = center_xy
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip((radius - rr) / (radius * max(edge_softness, 1e-3)), 0.0, 1.0)
    base = img.astype(np.float32)
    out = base * (0.2 + 0.8 * mask)
    out = out * boost_inside
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _augment_tone(img: np.ndarray, gamma: float = 0.9, blur_k: int = 3, noise_std: float = 4.0) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), gamma)
    x = (x * 255.0).astype(np.uint8)
    if blur_k > 0:
        x = cv2.GaussianBlur(x, (blur_k, blur_k), 0)
    if noise_std > 0:
        n = np.random.normal(0, noise_std, x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return x


def _labels_from_spec(spec: str) -> list[int]:
    lids: list[int] = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-')
            lids.extend(list(range(int(a), int(b) + 1)))
        else:
            lids.append(int(tok))
    return sorted(set(lids))


def _rotate_volume(arr: np.ndarray, yaw: float|None, pitch: float|None, roll: float|None, order: int) -> np.ndarray:
    """
    Apply Z (yaw), Y (pitch), X (roll) rotations in degrees to a (Z,Y,X) array.
    Uses scipy.ndimage.rotate with reshape=True to avoid cropping.
    order: 1 for CT (linear), 0 for labels (nearest).
    """
    out = arr
    if yaw and abs(float(yaw)) > 1e-3:
        out = ndi.rotate(out, angle=float(yaw), axes=(1, 2), reshape=True, order=order, mode='nearest')
    if pitch and abs(float(pitch)) > 1e-3:
        out = ndi.rotate(out, angle=float(pitch), axes=(0, 2), reshape=True, order=order, mode='nearest')
    if roll and abs(float(roll)) > 1e-3:
        out = ndi.rotate(out, angle=float(roll), axes=(0, 1), reshape=True, order=order, mode='nearest')
    return out


@cli.command("build-pseudo-lateral")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=0, show_default=True, help="0 = all")
@click.option("--labels", default="1-7", show_default=True, help="Label IDs to include (e.g., '1-7')")
@click.option("--height", type=int, default=512, show_default=True)
@click.option("--plane", type=click.Choice(["sag","cor","ax"], case_sensitive=False), default="sag", show_default=True)
@click.option("--rot90", type=int, default=1, show_default=True, help="Rotate output by k*90 degrees (0..3)")
@click.option("--flip-h/--no-flip-h", default=False, show_default=True)
@click.option("--flip-v/--no-flip-v", default=False, show_default=True)
@click.option("--aperture/--no-aperture", default=True, show_default=True)
@click.option("--yaw", type=float, default=None, help="Rotate around Z axis (deg)")
@click.option("--pitch", type=float, default=None, help="Rotate around Y axis (deg)")
@click.option("--roll", type=float, default=None, help="Rotate around X axis (deg)")
@click.option("--slab-vox", type=int, default=0, show_default=True, help="If >0, generate multiple slabs along the projection axis with this thickness (voxels)")
@click.option("--slab-mm", type=float, default=0.0, show_default=True, help="If >0, slab thickness in millimeters (takes precedence over --slab-vox)")
@click.option("--slab-count", type=int, default=3, show_default=True, help="How many slabs to sample (centered around mid-depth)")
@click.option("--slab-step-vox", type=int, default=0, show_default=True, help="Step between slabs (voxels). Default: slab-vox (non-overlapping)")
@click.option("--slab-step-mm", type=float, default=0.0, show_default=True, help="Step between slabs in millimeters (takes precedence over --slab-step-vox)")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("data/pseudo_lateral"), show_default=True)
def build_pseudo_lateral(data_root, subset, limit_cases, labels, height, plane, rot90, flip_h, flip_v, aperture,
                         yaw, pitch, roll, slab_vox, slab_mm, slab_count, slab_step_vox, slab_step_mm, out_dir):
    """
    Generate lateral-like pseudo X-ray images directly from CT by projecting along X (sagittal MIP-like),
    add C-arm aperture, blur and noise, and write aligned 2D masks for chosen labels (default C1..C7).
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    lids = _labels_from_spec(labels)
    pairs = find_pairs(Path(data_root), subset, limit=(None if limit_cases == 0 else limit_cases))
    if not pairs:
        raise click.ClickException("No cases found.")

    for i, (ct, seg) in enumerate(pairs, 1):
        if seg is None:
            click.echo(f"[{i}] {ct.name}: no seg; skipping")
            continue
        click.echo(f"[{i}] {ct.name}")
        vol_hu, sp, _ = load_nii(ct)
        lab, _, _ = load_nii(seg)
        lab = lab.astype(np.int32)
        # CT -> density in [0,1]
        ct_den = _ct_window_to_density(vol_hu, lo=200, hi=2000)
        # Optional 3D rotations to simulate camera orientation
        if any(v is not None and abs(float(v)) > 1e-3 for v in (yaw, pitch, roll)):
            ct_den = _rotate_volume(ct_den, yaw=yaw, pitch=pitch, roll=roll, order=1)
            lab = _rotate_volume(lab, yaw=yaw, pitch=pitch, roll=roll, order=0).astype(np.int32)
            click.echo(f"   ↻ rotated (yaw={yaw or 0:.1f}, pitch={pitch or 0:.1f}, roll={roll or 0:.1f})")
        # Determine projection axis and slab plan
        axis_map = {'sag': 2, 'cor': 1, 'ax': 0}
        proj_axis = axis_map[plane.lower()]
        L = ct_den.shape[proj_axis]
        # spacing in mm along projection axis (approximate; rotation not rescaling spacing)
        axis_spacing_mm = float(sp[proj_axis])
        # convert mm → vox when requested
        if slab_mm and slab_mm > 0:
            slab_vox = int(max(1, round(float(slab_mm) / max(axis_spacing_mm, 1e-6))))
        else:
            slab_vox = int(max(0, slab_vox))
        slab_count = int(max(1, slab_count))
        if slab_step_mm and slab_step_mm > 0:
            step = int(max(1, round(float(slab_step_mm) / max(axis_spacing_mm, 1e-6))))
        else:
            step = int(slab_step_vox) if slab_step_vox > 0 else (slab_vox if slab_vox > 0 else max(1, L // (slab_count + 1)))

        # Helper to extract a slab [i0,i1) along proj_axis
        def proj_slab_sum(arr_den: np.ndarray, i0: int, i1: int) -> np.ndarray:
            s = [slice(None)] * 3
            s[proj_axis] = slice(max(0, i0), min(L, i1))
            part = arr_den[tuple(s)]
            if part.shape[proj_axis] == 0:
                return np.zeros(part.shape[:proj_axis] + part.shape[proj_axis+1:], dtype=np.uint8)
            den = np.maximum(part, 0.0) ** 1.6
            att = den.sum(axis=proj_axis)
            att = att / (att.max() + 1e-6)
            return (att * 255.0).astype(np.uint8)

        def sil_slab(lab3d: np.ndarray, i0: int, i1: int, lid: int) -> np.ndarray:
            s = [slice(None)] * 3
            s[proj_axis] = slice(max(0, i0), min(L, i1))
            part = (lab3d[tuple(s)] == int(lid)).astype(np.uint8)
            if part.shape[proj_axis] == 0:
                return np.zeros(part.shape[:proj_axis] + part.shape[proj_axis+1:], dtype=np.uint8)
            return (part.max(axis=proj_axis) > 0).astype(np.uint8) * 255

        # Spacing for remaining axes (rows, cols) before resize
        # sp is (s0,s1,s2). axes2d maps to rows/cols kept after summation.
        axes2d = tuple(sorted(set([0,1,2]) - {proj_axis}))
        row_spacing = float(sp[axes2d[0]])
        col_spacing = float(sp[axes2d[1]])

        # Slab sampling plan (centered around mid)
        starts = []
        if slab_vox <= 0:
            # single full projection
            starts = [(0, L)]
        else:
            mid = L // 2
            half = slab_vox // 2
            # center slab
            starts.append((mid - half, mid - half + slab_vox))
            # additional slabs before/after
            k_left = (slab_count - 1) // 2
            k_right = slab_count - 1 - k_left
            for j in range(1, k_left + 1):
                i0 = mid - half - j * step
                starts.append((i0, i0 + slab_vox))
            for j in range(1, k_right + 1):
                i0 = mid - half + j * step
                starts.append((i0, i0 + slab_vox))
            # sort by i0
            starts.sort(key=lambda t: t[0])

        stem_base = f"{subset}_{ct.stem}"

        for si, (i0, i1) in enumerate(starts):
            # Project CT slab
            base = proj_slab_sum(ct_den, i0, i1)
            h0, w0 = base.shape[:2]
            new_h = int(height)
            scale = new_h / float(h0)
            new_w = int(round(w0 * scale))
            base = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Tone and fluoroscopy-like aperture vignette
            gamma = random.uniform(0.85, 1.15)
            blur_k = random.choice([3, 5])
            noise_std = random.uniform(2.0, 6.0)
            img = _augment_tone(base, gamma=gamma, blur_k=blur_k, noise_std=noise_std)

            # 2D label silhouettes for this slab and union mask
            masks2d = {lid: sil_slab(lab, i0, i1, lid) for lid in lids}
            union = np.zeros_like(next(iter(masks2d.values())))
            for m in masks2d.values():
                union = np.maximum(union, m)
            # resize masks to match image
            union = cv2.resize(union, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks2d = {k: cv2.resize(v, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) for k, v in masks2d.items()}

            # optional 90-degree rotations and flips to match acquisition orientation
            k = int(rot90) % 4
            if k:
                img = np.rot90(img, k=k)
                union = np.rot90(union, k=k)
                masks2d = {kk: np.rot90(vv, k=k) for kk, vv in masks2d.items()}
            if flip_h:
                img = np.fliplr(img)
                union = np.fliplr(union)
                masks2d = {kk: np.fliplr(vv) for kk, vv in masks2d.items()}
            if flip_v:
                img = np.flipud(img)
                union = np.flipud(union)
                masks2d = {kk: np.flipud(vv) for kk, vv in masks2d.items()}

            # Apply aperture after orientation transforms
            if aperture:
                hA, wA = img.shape[:2]
                R = 0.48 * min(hA, wA)
                cx = 0.52 * wA
                cy = 0.50 * hA
                img = _apply_c_arm_aperture(img, (cx, cy), radius=R, edge_softness=0.22, boost_inside=1.12)

            # overlay for quick inspection
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            colors = {1:(255,0,0),2:(255,128,0),3:(255,255,0),4:(0,255,0),5:(0,255,255),6:(0,128,255),7:(0,0,255)}
            for lid, m in masks2d.items():
                if m.max() == 0:
                    continue
                overlay[m.astype(bool)] = colors.get(int(lid), (255,255,255))
            overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1.0, overlay, 0.35, 0.0)

            # save
            stem = f"{stem_base}_slab{si:02d}"
            out_img = out_dir / "images" / f"{stem}.png"
            out_msk = out_dir / "masks" / f"{stem}.png"
            out_ovr = out_dir / "overlays" / f"{stem}.png"
            cv2.imwrite(str(out_img), _ensure_gray_u8(img))
            cv2.imwrite(str(out_msk), _ensure_gray_u8(union))
            cv2.imwrite(str(out_ovr), overlay)

            # pixel spacing after resize/rot: divide by scale; swap if rot90 is odd
            row_mm = row_spacing / max(scale, 1e-6)
            col_mm = col_spacing / max(scale, 1e-6)
            if (k % 2) == 1:
                row_mm, col_mm = col_mm, row_mm

            # stats JSON with mm metrics
            stats = {}
            for lid, m in masks2d.items():
                present = bool(m.max() > 0)
                c = _centroid(m) if present else None
                b = _bbox(m) if present else None
                area_px = int(m.astype(bool).sum()) if present else 0
                area_mm2 = float(area_px * row_mm * col_mm) if present else 0.0
                bbox_mm = None
                if b is not None:
                    h_px = (b[2] - b[0] + 1)
                    w_px = (b[3] - b[1] + 1)
                    bbox_mm = [float(h_px * row_mm), float(w_px * col_mm)]
                c_mm = None
                if c is not None:
                    c_mm = [float(c[0] * row_mm), float(c[1] * col_mm)]
                stats[str(lid)] = {
                    "present": present,
                    "centroid_2d_index": (None if c is None else [float(c[0]), float(c[1])]),
                    "centroid_2d_mm": c_mm,
                    "bbox_2d_index": (None if b is None else [int(b[0]), int(b[1]), int(b[2]), int(b[3])]),
                    "bbox_2d_hw_mm": bbox_mm,
                    "area_px": area_px,
                    "area_mm2": area_mm2,
                }

            with open(out_dir / f"{stem}.json", "w") as f:
                json.dump({
                    "case": ct.stem,
                    "subset": subset,
                    "image": str(out_img),
                    "mask": str(out_msk),
                    "overlay": str(out_ovr),
                    "labels": stats,
                    "plane": plane,
                    "rot90": int(k),
                    "flip_h": bool(flip_h),
                    "flip_v": bool(flip_v),
                    "pixel_spacing": [float(row_mm), float(col_mm)],
                    "slab_index": int(si),
                    "slab_vox": int(slab_vox if slab_vox>0 else L),
                    "slab_mm": float((slab_vox if slab_vox>0 else L) * axis_spacing_mm),
                    "slab_step_vox": int(step),
                    "slab_step_mm": float(step * axis_spacing_mm),
                    "slab_range_index": [int(max(0,i0)), int(min(L,i1))],
                    "slab_range_mm": [float(max(0,i0) * axis_spacing_mm), float(min(L,i1) * axis_spacing_mm)],
                    "proj_axis": int(proj_axis),
                    "proj_axis_spacing_mm": axis_spacing_mm,
                }, f, indent=2)

    click.echo(f"✅ Wrote pseudo-lateral dataset to {out_dir}")
