from __future__ import annotations
import json, math, random, csv
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


def _ct_window_to_density(vol_hu: np.ndarray, lo=-1000, hi=1000) -> np.ndarray:
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


def _draw_dashed_line(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int],
                      thickness: int = 2, dash: int = 8, gap: int = 6, line_type: int = cv2.LINE_AA) -> None:
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return
    vx = dx / dist
    vy = dy / dist
    pos = 0.0
    while pos < dist:
        seg_len = min(dash, dist - pos)
        x_start = int(round(x1 + vx * pos))
        y_start = int(round(y1 + vy * pos))
        x_end = int(round(x1 + vx * (pos + seg_len)))
        y_end = int(round(y1 + vy * (pos + seg_len)))
        cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness, lineType=line_type)
        pos += dash + gap


def _draw_dashed_circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int],
                        thickness: int = 2, dash_deg: float = 12.0, gap_deg: float = 10.0) -> None:
    cx, cy = int(center[0]), int(center[1])
    a = 0.0
    while a < 360.0:
        a2 = min(a + dash_deg, 360.0)
        cv2.ellipse(img, (cx, cy), (radius, radius), 0.0, a, a2, color, thickness, lineType=cv2.LINE_AA)
        a += dash_deg + gap_deg


def _draw_axis_helper(overlay: np.ndarray, plane: str, rot90: int, flip_h: bool, flip_v: bool,
                      yaw: float | None, pitch: float | None, roll: float | None) -> None:
    """Draw XYZ triad (X=red, Y=green, Z=blue) following the exact transforms
    applied to the volume: yaw→pitch→roll, then plane projection, rot90 and flips.
    Positive directions are solid; negative directions are dashed.
    """
    h, w = overlay.shape[:2]
    L = max(20, int(0.08 * min(h, w)))
    ox = int(0.12 * w)
    oy = int(0.88 * h)

    # Rotation matrix R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    y = math.radians(float(yaw or 0.0))
    p = math.radians(float(pitch or 0.0))
    r = math.radians(float(roll or 0.0))
    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(p), math.sin(p)
    cx, sx = math.cos(r), math.sin(r)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    R = Rz @ Ry @ Rx

    # Basis vectors in world X,Y,Z
    basis = {
        'X': np.array([1.0, 0.0, 0.0], dtype=float),
        'Y': np.array([0.0, 1.0, 0.0], dtype=float),
        'Z': np.array([0.0, 0.0, 1.0], dtype=float),
    }

    # Color map (BGR)
    colors = {'X': (0, 0, 255), 'Y': (0, 255, 0), 'Z': (255, 0, 0)}

    # Project rotated vectors to 2D before rot90/flips
    plane = plane.lower()

    def project_to_2d(v3: np.ndarray) -> tuple[float, float, float]:
        # v3 is a 3D direction vector in world; rotate by R
        v = (R @ v3.reshape(3, 1)).ravel()
        # Map to image pre-rot90/flips (dx right, dy down), and keep out-of-plane component magnitude
        if plane == 'sag':  # keep (Z,Y), drop X
            dx, dy = v[1], v[0]  # cols <- +Y, rows <- +Z
            oop = abs(v[2])      # magnitude along dropped axis (X)
        elif plane == 'cor':    # keep (Z,X), drop Y
            dx, dy = v[2], v[0]  # cols <- +X, rows <- +Z
            oop = abs(v[1])
        elif plane == 'ax':     # keep (Y,X), drop Z
            dx, dy = v[2], v[1]  # cols <- +X, rows <- +Y
            oop = abs(v[0])
        else:
            dx, dy, oop = 0.0, 0.0, 1.0
        return float(dx), float(dy), float(oop)

    def apply_2d_transforms(dx: float, dy: float) -> tuple[float, float]:
        # Apply rot90 CCW k times (numpy semantics)
        k = int(rot90) % 4
        for _ in range(k):
            dx, dy = -dy, dx
        if bool(flip_h):
            dx = -dx
        if bool(flip_v):
            dy = -dy
        return dx, dy

    def draw_axis(name: str):
        dx, dy, oop = project_to_2d(basis[name])
        dx, dy = apply_2d_transforms(dx, dy)
        # Normalize in-plane length
        norm = math.hypot(dx, dy)
        if norm < 1e-6 or oop > 0.98:
            # Nearly out-of-plane: draw dot (positive) and dashed ring (negative)
            r_outer = max(6, L // 6)
            r_inner = max(3, r_outer - 2)
            _draw_dashed_circle(overlay, (ox, oy), r_outer, colors[name], thickness=2)
            cv2.circle(overlay, (ox, oy), r_inner, colors[name], thickness=-1, lineType=cv2.LINE_AA)
            return
        ux, uy = dx / norm, dy / norm
        # Positive (solid)
        p1 = (ox, oy)
        p2 = (int(round(ox + ux * L)), int(round(oy + uy * L)))
        cv2.line(overlay, p1, p2, colors[name], thickness=2, lineType=cv2.LINE_AA)
        # Negative (dashed)
        q2 = (int(round(ox - ux * L)), int(round(oy - uy * L)))
        _draw_dashed_line(overlay, p1, q2, colors[name], thickness=2)

    for a in ('X', 'Y', 'Z'):
        draw_axis(a)

    legend = [
        ('Yaw', (0.88, 0.92), 'Z'),
        ('Pitch', (0.88, 0.86), 'Y'),
        ('Roll', (0.88, 0.80), 'X'),
    ]
    for label, (nx, ny), comp in legend:
        color = colors.get(comp, (255, 255, 255))
        px = int(nx * w)
        py = int(ny * h)
        cv2.putText(overlay, label, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness=1, lineType=cv2.LINE_AA)


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


def _shift_volume(arr: np.ndarray, pan_x_mm: float|None, pan_y_mm: float|None, pan_z_mm: float|None, spacing_xyz: np.ndarray, order: int) -> np.ndarray:
    """
    Translate a (Z,Y,X) array by mm along X/Y/Z before projection.
    spacing_xyz is (sx, sy, sz) in mm/voxel.
    """
    dx = float(pan_x_mm or 0.0)
    dy = float(pan_y_mm or 0.0)
    dz = float(pan_z_mm or 0.0)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6:
        return arr
    sx, sy, sz = [float(v) for v in spacing_xyz]
    # array axes = (Z,Y,X) → shifts in voxels along (0,1,2)
    shift_vox = [dz / max(sz, 1e-6), dy / max(sy, 1e-6), dx / max(sx, 1e-6)]
    return ndi.shift(arr, shift=shift_vox, order=order, mode='nearest', prefilter=(order != 0))


@cli.command("build-pseudo-lateral")
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/CTSpine1K"), show_default=True)
@click.option("--subset", default="HNSCC-3DCT-RT", show_default=True)
@click.option("--limit-cases", type=int, default=0, show_default=True, help="0 = all")
@click.option("--labels", default="1-7", show_default=True, help="Label IDs to include (e.g., '1-7')")
@click.option("--height", type=int, default=512, show_default=True)
@click.option("--native-resolution/--no-native-resolution", default=False, show_default=True,
              help="Keep the CT's native in-plane resolution (ignores --height and disables post-resize)")
@click.option("--plane", type=click.Choice(["sag","cor","ax"], case_sensitive=False), default="sag", show_default=True)
@click.option("--rot90", type=int, default=0, show_default=True, help="Rotate output by k*90 degrees (0..3)")
@click.option("--ct-window-lo", type=float, default=-1000.0, show_default=True,
              help="Lower HU bound used before projection (clip below this)")
@click.option("--ct-window-hi", type=float, default=1000.0, show_default=True,
              help="Upper HU bound used before projection (clip above this)")
@click.option("--projection-power", type=float, default=1.0, show_default=True,
              help="Exponent applied to normalized HU before summation (1.0 mimics raw CT)")
@click.option("--tone-style", type=click.Choice(["ct", "fluoro"], case_sensitive=False), default="ct", show_default=True,
              help="'ct' keeps clean CT tone; 'fluoro' adds gamma/blur/noise like C-arm")
@click.option("--auto-crop/--no-auto-crop", default=False, show_default=True,
              help="Crop outputs to the union mask bounding box (plus margin)")
@click.option("--crop-margin-mm", type=float, default=20.0, show_default=True,
              help="Margin (mm) to retain around the mask when auto-cropping")
@click.option("--resize-after-crop/--no-resize-after-crop", default=True, show_default=True,
              help="Re-scale cropped image back to the requested height to keep output resolution consistent")
@click.option("--flip-h/--no-flip-h", default=False, show_default=True)
@click.option("--flip-v/--no-flip-v", default=False, show_default=True)
@click.option("--aperture/--no-aperture", default=True, show_default=True)
@click.option("--axis-helper/--no-axis-helper", default=False, show_default=True, help="Draw XYZ axis helper on overlay (X=red, Y=green, Z=blue)")
@click.option("--yaw", type=float, default=None, help="Rotate around Z axis (deg)")
@click.option("--pitch", type=float, default=None, help="Rotate around Y axis (deg)")
@click.option("--roll", type=float, default=None, help="Rotate around X axis (deg)")
@click.option("--slab-vox", type=int, default=0, show_default=True, help="If >0, generate multiple slabs along the projection axis with this thickness (voxels)")
@click.option("--slab-mm", type=float, default=0.0, show_default=True, help="If >0, slab thickness in millimeters (takes precedence over --slab-vox)")
@click.option("--slab-count", type=int, default=3, show_default=True, help="How many slabs to sample (centered around mid-depth)")
@click.option("--slab-step-vox", type=int, default=0, show_default=True, help="Step between slabs (voxels). Default: slab-vox (non-overlapping)")
@click.option("--slab-step-mm", type=float, default=0.0, show_default=True, help="Step between slabs in millimeters (takes precedence over --slab-step-vox)")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("data/pseudo_lateral"), show_default=True)
@click.option("--write-manifest/--no-write-manifest", default=True, show_default=True, help="Write manifest.csv summarizing outputs")
@click.option("--clear-dir/--no-clear-dir", default=False, show_default=True, help="Delete --out-dir before generating (DANGEROUS)")
@click.option("--keep-slabs", default="", show_default=True, help="Comma/range list of slab indices to keep (e.g., '13-15,20'). Empty = all.")
@click.option("--keep-mm", type=float, default=0.0, show_default=True, help="Keep only slabs whose center lies within ±mm of mid-depth. If --window-count > 0, resample exactly that many slabs within the window.")
@click.option("--window-count", type=int, default=0, show_default=True, help="If >0 with --keep-mm, generate exactly N slabs evenly spaced within ±keep-mm window around mid-depth.")
@click.option("--pan-x-mm", type=float, default=0.0, show_default=True, help="Translate along X (mm) before projection")
@click.option("--pan-y-mm", type=float, default=0.0, show_default=True, help="Translate along Y (mm) before projection")
@click.option("--pan-z-mm", type=float, default=0.0, show_default=True, help="Translate along Z (mm) before projection")
@click.option("--override-existing/--no-override-existing", default=False, show_default=True,
              help="Skip recomputing samples if the per-sample folder already exists with all outputs")
def build_pseudo_lateral(data_root, subset, limit_cases, labels, height, native_resolution, plane, rot90,
                         ct_window_lo, ct_window_hi, projection_power, tone_style,
                         auto_crop, crop_margin_mm, resize_after_crop,
                         flip_h, flip_v, aperture, axis_helper,
                         yaw, pitch, roll, slab_vox, slab_mm, slab_count, slab_step_vox, slab_step_mm, out_dir, write_manifest, clear_dir, keep_slabs, keep_mm, window_count, pan_x_mm, pan_y_mm, pan_z_mm, override_existing):
    """Deprecated entry point retained for CLI compatibility."""

    raise click.ClickException(
        "'build-pseudo-lateral' is deprecated and no longer maintained. "
        "Please switch to 'spine build-hf-projection'."
    )
    tone_style = tone_style.lower()
    if ct_window_hi <= ct_window_lo:
        raise click.ClickException("--ct-window-hi must be greater than --ct-window-lo")
    proj_power = float(projection_power)
    if proj_power <= 0:
        raise click.ClickException("--projection-power must be > 0")
    if not native_resolution and height <= 0:
        raise click.ClickException("--height must be > 0 unless --native-resolution is enabled")

    out_dir = Path(out_dir)
    # New structure: one folder per generated image (sample)
    if clear_dir and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
        click.echo(f"🧹 Cleared directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


    images_dir = out_dir / "images"
    masks_dir = out_dir / "mask"
    mask_labels_dir = out_dir / "mask_labels"
    overlays_dir = out_dir / "overlays"
    overlay_recolored_dir = out_dir / "overlay_recolored"
    labels_dir = out_dir / "labels-json"
    for d in (images_dir, masks_dir, mask_labels_dir, overlays_dir, overlay_recolored_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest_rows = []  # kept for backward compatibility (we also stream rows)
    # Prepare streaming manifest writer
    mf = out_dir / "manifest.csv"
    headers = ["image", "mask_labels"]
    # lids list will be known after labels parsing; append their header fields when first row is written
    manifest_inited = mf.exists()
    if write_manifest and not manifest_inited:
        with open(mf, "w", newline="") as f:
            pass  # header will be written after we know lids

    lids = _labels_from_spec(labels)
    pairs = find_pairs(Path(data_root), subset, limit=(None if limit_cases == 0 else limit_cases))
    if not pairs:
        raise click.ClickException("No cases found.")

    # parse keep_slabs spec into a set of indices (0-based) or None for all
    def _parse_keep(spec: str):
        spec = str(spec or "").strip()
        if not spec:
            return None
        keep = set()
        for tok in spec.split(','):
            tok = tok.strip()
            if not tok:
                continue
            if '-' in tok:
                a, b = tok.split('-')
                a = int(a); b = int(b)
                for k in range(min(a,b), max(a,b)+1):
                    keep.add(k)
            else:
                keep.add(int(tok))
        return keep

    keep_set = _parse_keep(keep_slabs)

    for i, (ct, seg) in enumerate(pairs, 1):
        if seg is None:
            click.echo(f"[{i}] {ct.name}: no seg; skipping")
            continue
        click.echo(f"[{i}] {ct.name}")
        vol_hu, sp, _ = load_nii(ct)
        lab, _, _ = load_nii(seg)
        lab = lab.astype(np.int32)
        # CT -> density in [0,1] using configurable window (defaults mimic inspect 00_* outputs)
        ct_den = _ct_window_to_density(vol_hu, lo=ct_window_lo, hi=ct_window_hi)
        # Optional 3D rotations to simulate camera orientation
        if any(v is not None and abs(float(v)) > 1e-3 for v in (yaw, pitch, roll)):
            ct_den = _rotate_volume(ct_den, yaw=yaw, pitch=pitch, roll=roll, order=1)
            lab = _rotate_volume(lab, yaw=yaw, pitch=pitch, roll=roll, order=0).astype(np.int32)
            click.echo(f"   ↻ rotated (yaw={yaw or 0:.1f}, pitch={pitch or 0:.1f}, roll={roll or 0:.1f})")
        # Optional 3D translation (pan) in mm prior to projection
        if any(abs(float(v or 0.0)) > 1e-6 for v in (pan_x_mm, pan_y_mm, pan_z_mm)):
            ct_den = _shift_volume(ct_den, pan_x_mm, pan_y_mm, pan_z_mm, sp, order=1)
            lab = _shift_volume(lab, pan_x_mm, pan_y_mm, pan_z_mm, sp, order=0).astype(np.int32)
            click.echo(f"   ⇢ panned (x={pan_x_mm:.1f}mm, y={pan_y_mm:.1f}mm, z={pan_z_mm:.1f}mm)")
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
            den = np.clip(part, 0.0, 1.0) ** proj_power
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
        mid = L // 2
        if slab_vox <= 0:
            # single full projection
            starts = [(0, L)]
        else:
            half = slab_vox // 2
            if keep_mm and keep_mm > 0 and window_count and window_count > 0:
                # Evenly spaced centers within ±keep_mm window
                half_window_vox = float(keep_mm) / max(axis_spacing_mm, 1e-6)
                centers = np.linspace(mid - half_window_vox, mid + half_window_vox, int(window_count))
                for c in centers:
                    c = int(round(c))
                    i0 = c - half
                    i1 = i0 + slab_vox
                    # clamp to bounds while preserving thickness
                    if i0 < 0:
                        i1 -= i0
                        i0 = 0
                    if i1 > L:
                        i0 -= (i1 - L)
                        i1 = L
                    starts.append((i0, i1))
                # de-dup and sort
                starts = sorted(list({(int(a), int(b)) for a, b in starts}), key=lambda t: t[0])
            else:
                # Default symmetric sampling around center with step
                starts.append((mid - half, mid - half + slab_vox))
                k_left = (slab_count - 1) // 2
                k_right = slab_count - 1 - k_left
                for j in range(1, k_left + 1):
                    i0 = mid - half - j * step
                    starts.append((i0, i0 + slab_vox))
                for j in range(1, k_right + 1):
                    i0 = mid - half + j * step
                    starts.append((i0, i0 + slab_vox))
                starts.sort(key=lambda t: t[0])

        stem_base = f"{subset}_{ct.stem}"

        for si, (i0, i1) in enumerate(starts):
            if keep_set is not None and si not in keep_set:
                continue
            # Early skip if sample folder already exists and override is false
            stem = f"{stem_base}_slab{si:02d}"
            sample_dir = out_dir / stem
            if (not override_existing) and sample_dir.exists():
                need = [sample_dir/"image.png", sample_dir/"mask.png", sample_dir/"overlay.png", sample_dir/"labels.json"]
                if all(p.exists() for p in need):
                    click.echo(f"   ⏭️  skip existing {stem}")
                    continue
            # Project CT slab
            base = proj_slab_sum(ct_den, i0, i1)
            h0, w0 = base.shape[:2]
            if native_resolution:
                new_h = h0
                new_w = w0
                scale = 1.0
            else:
                new_h = int(height)
                scale = new_h / float(max(h0, 1))
                new_w = int(max(1, round(w0 * scale)))
                if new_h != h0 or new_w != w0:
                    base = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if tone_style == "fluoro":
                # Fluoroscopy-style tone via random gamma, blur, and sensor noise
                gamma = random.uniform(0.85, 1.15)
                blur_k = random.choice([3, 5])
                noise_std = random.uniform(2.0, 6.0)
                img = _augment_tone(base, gamma=gamma, blur_k=blur_k, noise_std=noise_std)
            else:
                # Keep clean CT appearance similar to inspect's 00_* outputs
                img = _ensure_gray_u8(base)

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

            label_map = np.zeros_like(union, dtype=np.uint8)
            for lid, m in masks2d.items():
                if m.max() == 0:
                    continue
                label_map = np.where(m > 0, np.uint8(lid), label_map)

            # Pixel spacing after resize/rot: divide by scale; swap if rot90 is odd
            row_mm = row_spacing / max(scale, 1e-6)
            col_mm = col_spacing / max(scale, 1e-6)
            if (k % 2) == 1:
                row_mm, col_mm = col_mm, row_mm

            crop_y0 = 0
            crop_x0 = 0
            crop_h = img.shape[0]
            crop_w = img.shape[1]
            crop_h_raw = crop_h
            crop_w_raw = crop_w
            crop_margin_mm = max(float(crop_margin_mm), 0.0)
            did_crop = False
            if auto_crop:
                mask_src = union
                if mask_src.max() == 0:
                    # fallback: use simple intensity support
                    mask_src = (img > 0).astype(np.uint8) * 255
                ys, xs = np.where(mask_src > 0)
                if ys.size > 0 and xs.size > 0:
                    pad_r = int(round(crop_margin_mm / max(row_mm, 1e-6)))
                    pad_c = int(round(crop_margin_mm / max(col_mm, 1e-6)))
                    y0 = max(0, int(ys.min()) - pad_r)
                    y1 = min(img.shape[0], int(ys.max()) + pad_r + 1)
                    x0 = max(0, int(xs.min()) - pad_c)
                    x1 = min(img.shape[1], int(xs.max()) + pad_c + 1)
                    if (y1 - y0) > 0 and (x1 - x0) > 0:
                        img = img[y0:y1, x0:x1]
                        union = union[y0:y1, x0:x1]
                        masks2d = {kk: vv[y0:y1, x0:x1] for kk, vv in masks2d.items()}
                        label_map = label_map[y0:y1, x0:x1]
                        crop_y0, crop_x0 = y0, x0
                        crop_h, crop_w = img.shape[:2]
                        crop_h_raw, crop_w_raw = crop_h, crop_w
                        did_crop = True

            allow_post_resize = resize_after_crop and (not native_resolution)
            if allow_post_resize and did_crop and new_h > 0 and img.shape[0] != new_h:
                scale_post = new_h / float(max(img.shape[0], 1))
                new_w_post = int(max(1, round(img.shape[1] * scale_post)))
                img = cv2.resize(img, (new_w_post, new_h), interpolation=cv2.INTER_AREA)
                union = cv2.resize(union, (new_w_post, new_h), interpolation=cv2.INTER_NEAREST)
                masks2d = {kk: cv2.resize(vv, (new_w_post, new_h), interpolation=cv2.INTER_NEAREST) for kk, vv in masks2d.items()}
                label_map = cv2.resize(label_map, (new_w_post, new_h), interpolation=cv2.INTER_NEAREST)
                row_mm /= max(scale_post, 1e-6)
                col_mm /= max(scale_post, 1e-6)
                crop_h = img.shape[0]
                crop_w = img.shape[1]

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

            # Axis helper (optional)
            if axis_helper:
                try:
                    _draw_axis_helper(overlay, plane=plane, rot90=rot90, flip_h=flip_h, flip_v=flip_v,
                                      yaw=yaw, pitch=pitch, roll=roll)
                except Exception as e:
                    # Keep robust; do not fail generation if helper drawing had an issue
                    click.echo(f"      (axis-helper error: {e})")

            # write artefacts into structured directories
            out_img = images_dir / f"{stem}.png"
            out_msk = masks_dir / f"{stem}.png"
            out_lbl = mask_labels_dir / f"{stem}.png"
            out_ovr = overlays_dir / f"{stem}.png"
            out_lbljson = labels_dir / f"{stem}.json"
            out_recolored = overlay_recolored_dir / f"{stem}.png"

            cv2.imwrite(str(out_img), _ensure_gray_u8(img))
            cv2.imwrite(str(out_msk), _ensure_gray_u8(union))
            cv2.imwrite(str(out_lbl), label_map)
            cv2.imwrite(str(out_ovr), overlay)

            color_mask = np.zeros((*img.shape, 3), dtype=np.uint8)
            for lid, m in masks2d.items():
                if m.max() == 0:
                    continue
                color_mask[m.astype(bool)] = colors.get(int(lid), (255, 255, 255))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay_recolored = cv2.addWeighted(color_mask, 0.45, img_rgb, 0.55, 0.0)
            cv2.imwrite(str(out_recolored), overlay_recolored)

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

            with open(out_lbljson, "w") as f:
                json.dump({
                    "case": ct.stem,
                    "subset": subset,
                    "image": str(out_img),
                    "mask": str(out_msk),
                    "mask_labels": str(out_lbl),
                    "overlay": str(out_ovr),
                    "overlay_recolored": str(out_recolored),
                    "labels": stats,
                    "plane": plane,
                    "rot90": int(k),
                    "flip_h": bool(flip_h),
                    "flip_v": bool(flip_v),
                    "crop_origin_index": [int(crop_y0), int(crop_x0)],
                    "crop_size_index": [int(crop_h_raw), int(crop_w_raw)],
                    "image_size_index": [int(img.shape[0]), int(img.shape[1])],
                    "pan_mm": [float(pan_x_mm), float(pan_y_mm), float(pan_z_mm)],
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

            # Append manifest row
            row = {
                "image": str(out_img),
                "mask_labels": str(out_lbl),
            }
            # Stream manifest row immediately
            manifest_rows.append(row)
            if write_manifest:
                if not manifest_inited:
                    with open(mf, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=headers)
                        w.writeheader()
                    manifest_inited = True
                with open(mf, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=headers)
                    w.writerow(row)
                    f.flush()

    click.echo(f"✅ Wrote pseudo-lateral dataset to {out_dir}")
