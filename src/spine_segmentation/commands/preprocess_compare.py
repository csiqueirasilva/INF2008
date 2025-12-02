from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

import click
import cv2
import numpy as np

from .root import cli
from ..core.deepdrr_bridge import render_deepdrr_projection


def _double_clahe(
    img: np.ndarray,
    clip1: float,
    tile1: int,
    clip2: float,
    tile2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply two CLAHE passes back-to-back to lift local contrast."""
    clahe1 = cv2.createCLAHE(clipLimit=clip1, tileGridSize=(tile1, tile1))
    first = clahe1.apply(img)
    clahe2 = cv2.createCLAHE(clipLimit=clip2, tileGridSize=(tile2, tile2))
    second = clahe2.apply(first)
    return first, second


def _otsu_mask(img: np.ndarray, blur_kernel: int) -> np.ndarray:
    """Compute an Otsu mask (optional Gaussian blur for stability)."""
    proc = img
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    _, mask = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _overlay_mask(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a binary mask on top of a grayscale base."""
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(overlay)
    color_mask[mask > 0] = (0, 0, 255)  # red tint in BGR
    return cv2.addWeighted(overlay, 0.65, color_mask, 0.35, 0.0)


def _hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Match the histogram of `source` to `template` (grayscale uint8).
    Uses simple CDF mapping; returns uint8 image.
    """
    src = source.ravel()
    tmpl = template.ravel()

    src_hist, _ = np.histogram(src, bins=256, range=(0, 255))
    tmpl_hist, _ = np.histogram(tmpl, bins=256, range=(0, 255))

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    tmpl_cdf = np.cumsum(tmpl_hist).astype(np.float64)
    if src_cdf[-1] == 0 or tmpl_cdf[-1] == 0:
        return source.copy()
    src_cdf /= src_cdf[-1]
    tmpl_cdf /= tmpl_cdf[-1]

    mapping = np.interp(src_cdf, tmpl_cdf, np.arange(256))
    matched = mapping[src].reshape(source.shape)
    return np.clip(matched, 0, 255).astype(np.uint8)


def _unsharp_enhance(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """
    Simple unsharp mask: blur then boost high frequencies to accent edges/bone.
    """
    if sigma <= 0 or amount <= 0:
        return img
    # Kernel size derived from sigma; ensure odd and >=3.
    k = max(3, int(round(sigma * 6)) | 1)
    blur = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    # Classic unsharp: img * (1 + amount) - blur * amount
    enhanced = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def _label_bbox(labels: np.ndarray, label_ids: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """
    Compute tight bbox (x0,y0,x1,y1 inclusive) for given label ids in a 2D label map.
    Returns None if no pixels are found.
    """
    mask = np.isin(labels, label_ids)
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _letterbox(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    color: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, dict]:
    """
    Resize with aspect ratio preserved and pad to (target_h, target_w).
    Returns the letterboxed image and padding metadata.
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if img.ndim == 2:
        letter = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=int(color)
        )
    else:
        letter = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(int(color),) * 3
        )
    meta = {
        "scale": scale,
        "pad": (pad_left, pad_top, pad_right, pad_bottom),
        "new_size": (new_w, new_h),
    }
    return letter, meta


def _pad_to_canvas(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    color: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, dict]:
    """
    Pad to (target_h, target_w) without scaling when the crop is smaller.
    Falls back to letterbox resize if the crop exceeds the canvas.
    """
    h, w = img.shape[:2]
    if h > target_h or w > target_w:
        return _letterbox(img, target_w, target_h, color=color, interpolation=interpolation)
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    if img.ndim == 2:
        padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=int(color)
        )
    else:
        padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(int(color),) * 3
        )
    meta = {
        "scale": 1.0,
        "pad": (pad_left, pad_top, pad_right, pad_bottom),
        "new_size": (w, h),
    }
    return padded, meta


def _stack_horizontal(img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
    """Place two images side by side, padding to the same height."""
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    max_h = max(h_left, h_right)

    def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, _ = img.shape[:2]
        if h == target_h:
            return img
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        if img.ndim == 2:
            return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    left = pad_to_height(img_left, max_h)
    right = pad_to_height(img_right, max_h)
    return np.hstack([left, right])


def _parse_guide(spec: str | None) -> tuple[float, float, float, float] | None:
    """
    Parse a guide string formatted as 'x1,y1,x2,y2' in normalized coords [0,1].
    Returns None if not provided.
    """
    if not spec:
        return None
    try:
        parts = [float(v) for v in spec.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)  # type: ignore[return-value]
    except Exception as exc:
        raise click.ClickException("Guide must be 'x1,y1,x2,y2' in normalized coords (0-1).") from exc


def _draw_guide(img: np.ndarray, guide: tuple[float, float, float, float], color=(0, 0, 255), thickness: int = 3) -> np.ndarray:
    """Draw a guide line on a BGR image using normalized coordinates."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = guide
    pt1 = (int(round(x1 * w)), int(round(y1 * h)))
    pt2 = (int(round(x2 * w)), int(round(y2 * h)))
    out = img.copy()
    cv2.line(out, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    return out


def _guide_from_mask(mask_path: Path, labels: tuple[int, ...]) -> tuple[float, float, float, float]:
    """Derive a guide line from a labelled mask by fitting a line through per-label centroids."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise click.ClickException(f"Failed to read mask: {mask_path}")
    h, w = mask.shape[:2]
    if len(labels) == 0:
        raise click.ClickException("No labels provided for guide computation.")
    pts = []
    for lid in labels:
        ys, xs = np.nonzero(mask == lid)
        if ys.size == 0:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())
        pts.append((cy, cx))
    if not pts:
        raise click.ClickException(f"No pixels found for labels {labels} in {mask_path}")

    ys = np.array([p[0] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)

    if ys.size >= 2:
        m, b = np.polyfit(ys, xs, 1)
    else:
        # single point: use vertical line through centroid
        m, b = 0.0, float(xs.mean())

    y1 = float(ys.min())
    y2 = float(ys.max())
    x1 = m * y1 + b
    x2 = m * y2 + b
    return (
        float(np.clip(x1 / w, 0.0, 1.0)),
        float(np.clip(y1 / h, 0.0, 1.0)),
        float(np.clip(x2 / w, 0.0, 1.0)),
        float(np.clip(y2 / h, 0.0, 1.0)),
    )


def _guide_from_labelmap(label_map: np.ndarray, labels: tuple[int, ...]) -> tuple[float, float, float, float]:
    """Derive a guide line from an in-memory label map."""
    h, w = label_map.shape[:2]
    pts = []
    for lid in labels:
        ys, xs = np.nonzero(label_map == lid)
        if ys.size == 0:
            continue
        cy = float(ys.mean())
        cx = float(xs.mean())
        pts.append((cy, cx))
    if not pts:
        raise click.ClickException("No pixels found for requested labels in label map.")
    ys = np.array([p[0] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)
    if ys.size >= 2:
        m, b = np.polyfit(ys, xs, 1)
    else:
        m, b = 0.0, float(xs.mean())
    y1 = float(ys.min())
    y2 = float(ys.max())
    x1 = m * y1 + b
    x2 = m * y2 + b
    return (
        float(np.clip(x1 / w, 0.0, 1.0)),
        float(np.clip(y1 / h, 0.0, 1.0)),
        float(np.clip(x2 / w, 0.0, 1.0)),
        float(np.clip(y2 / h, 0.0, 1.0)),
    )


def _apply_c_arm_aperture(img: np.ndarray, center_xy: tuple[float, float], radius: float,
                          edge_softness: float = 0.0, inside_gain: float = 1.0,
                          outside_gain: float = 0.0, blur_px: int = 0,
                          ellipse_y_scale: float = 1.0) -> np.ndarray:
    """Apply a hard circular/elliptical mask (optionally softened) with tunable gains."""
    mask = _make_aperture_mask(img.shape[:2], center_xy, radius, edge_softness, blur_px, ellipse_y_scale)
    base = img.astype(np.float32)
    gain = outside_gain + (inside_gain - outside_gain) * mask
    out = base * gain
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _make_aperture_mask(shape: tuple[int, int], center_xy: tuple[float, float], radius: float,
                        edge_softness: float, blur_px: int, ellipse_y_scale: float) -> np.ndarray:
    """Build a [0,1] aperture mask."""
    h, w = shape
    cx, cy = center_xy
    ry = max(ellipse_y_scale, 1e-3)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + ((yy - cy) / ry) ** 2)
    if edge_softness <= 0:
        mask = (rr <= radius).astype(np.float32)
    else:
        mask = np.clip((radius - rr) / (radius * edge_softness), 0.0, 1.0)
    if blur_px > 0:
        k = max(1, int(blur_px) // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _parse_label_ids(spec: str) -> tuple[int, ...]:
    """
    Parse label ids from a string like "1-7,9" into a tuple of ints.
    """
    parts = []
    for token in spec.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            try:
                a, b = token.split("-", 1)
                a, b = int(a), int(b)
            except Exception as exc:
                raise click.ClickException(f"Invalid label range: {token}") from exc
            step = 1 if a <= b else -1
            parts.extend(range(a, b + step, step))
        else:
            try:
                parts.append(int(token))
            except Exception as exc:
                raise click.ClickException(f"Invalid label id: {token}") from exc
    dedup = sorted(set(parts))
    if not dedup:
        raise click.ClickException("No valid label IDs parsed for guide computation.")
    return tuple(dedup)


def _project_label_volume(
    label_volume_path: Path | None,
    yaw: float,
    pitch: float,
    roll: float,
    target_shape: tuple[int, int],
    native_resolution: bool,
    sensor_height: int,
    sensor_width: int | None,
    label_volume: np.ndarray | None = None,
    slice_offset_mm: float = 0.0,
    slice_thickness_mm: float = 0.0,
    spacing_z: float = 1.0,
) -> np.ndarray:
    """Project a 3D label volume with the same rotations as the DeepDRR render."""
    import nibabel as nib
    import scipy.ndimage as ndi

    if label_volume is None:
        nii = nib.load(str(label_volume_path))
        lbl = np.asarray(nii.get_fdata(), dtype=np.int16)
    else:
        lbl = np.asarray(label_volume, dtype=np.int16)
    # Reorder to (Z, Y, X) matching deepdrr_bridge
    lbl = np.transpose(lbl, (2, 1, 0))

    def rotate(arr: np.ndarray, order: int) -> np.ndarray:
        out = arr
        if yaw and abs(float(yaw)) > 1e-3:
            out = ndi.rotate(out, angle=float(yaw), axes=(1, 2), reshape=True, order=order, mode="nearest")
        if pitch and abs(float(pitch)) > 1e-3:
            out = ndi.rotate(out, angle=float(pitch), axes=(0, 2), reshape=True, order=order, mode="nearest")
        if roll and abs(float(roll)) > 1e-3:
            out = ndi.rotate(out, angle=float(roll), axes=(0, 1), reshape=True, order=order, mode="nearest")
        return out

    lbl = rotate(lbl, order=0)

    # Apply slab mask along projection axis (axis=2) after rotation
    if slice_thickness_mm > 0:
        nz = lbl.shape[2]
        mid = (nz - 1) / 2.0
        center_idx = mid + slice_offset_mm / max(spacing_z, 1e-6)
        half = (slice_thickness_mm / 2.0) / max(spacing_z, 1e-6)
        z0 = int(np.floor(center_idx - half))
        z1 = int(np.ceil(center_idx + half))
        z0 = max(0, z0)
        z1 = min(nz, z1)
        mask = np.zeros_like(lbl, dtype=bool)
        if z1 > z0:
            mask[..., z0:z1] = True
        lbl = np.where(mask, lbl, 0)
    # Project along +X (axis=2)
    proj = lbl.max(axis=2)

    proj_img = proj.astype(np.int16)
    if not native_resolution:
        target_h = int(sensor_height)
        target_w = int(sensor_width if sensor_width is not None else sensor_height)
        proj_img = cv2.resize(proj_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        h, w = proj_img.shape[:2]
        th, tw = target_shape
        if (h, w) != (th, tw):
            proj_img = cv2.resize(proj_img, (tw, th), interpolation=cv2.INTER_NEAREST)
    return proj_img


def _process_group(
    paths: Iterable[Path],
    label: str,
    out_root: Path,
    clip1: float,
    tile1: int,
    clip2: float,
    tile2: int,
    blur_kernel: int,
) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        img_path = path.expanduser().resolve()
        if not img_path.exists():
            raise click.ClickException(f"Input not found: {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise click.ClickException(f"Failed to read image: {img_path}")

        clahe1, clahe2 = _double_clahe(img, clip1, tile1, clip2, tile2)
        mask = _otsu_mask(clahe2, blur_kernel)
        overlay = _overlay_mask(clahe2, mask)

        out_dir = out_root / label / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "orig": out_dir / "orig_gray.png",
            "clahe1": out_dir / "clahe1.png",
            "clahe2": out_dir / "clahe2.png",
            "otsu_mask": out_dir / "otsu_mask.png",
            "otsu_overlay": out_dir / "otsu_overlay.png",
        }

        cv2.imwrite(str(outputs["orig"]), img)
        cv2.imwrite(str(outputs["clahe1"]), clahe1)
        cv2.imwrite(str(outputs["clahe2"]), clahe2)
        cv2.imwrite(str(outputs["otsu_mask"]), mask)
        cv2.imwrite(str(outputs["otsu_overlay"]), overlay)

        records.append(
            {
                "dataset": label,
                "stem": img_path.stem,
                "input": str(img_path),
                "orig_gray": str(outputs["orig"]),
                "clahe1": str(outputs["clahe1"]),
                "clahe2": str(outputs["clahe2"]),
                "otsu_mask": str(outputs["otsu_mask"]),
                "otsu_overlay": str(outputs["otsu_overlay"]),
            }
        )
    return records


@cli.command("preprocess-compare")
@click.option(
    "--frame",
    "frames",
    multiple=True,
    type=click.Path(path_type=Path),
    default=("data/frames/50/v50_f145.png",),
    show_default=True,
    help="Frame(s) to preprocess (grayscale assumed).",
)
@click.option(
    "--pseudo",
    "pseudos",
    multiple=True,
    type=click.Path(path_type=Path),
    default=(),
    show_default=True,
    help="Optional pseudo-lateral image(s) to preprocess for visual comparison.",
)
@click.option(
    "--deepdrr",
    "deepdrrs",
    multiple=True,
    type=click.Path(path_type=Path),
    default=("outputs/deepdrr/fake-vfss/images/HNSCC-3DCT-RT_HN_P001_pitch0.png",),
    show_default=True,
    help="DeepDRR projection(s) to preprocess alongside frames/pseudo images.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/preprocess_compare"),
    show_default=True,
    help="Root directory for processed outputs.",
)
@click.option(
    "--clip-limit1",
    type=float,
    default=2.0,
    show_default=True,
    help="Clip limit for the first CLAHE pass.",
)
@click.option(
    "--clip-limit2",
    type=float,
    default=2.0,
    show_default=True,
    help="Clip limit for the second CLAHE pass.",
)
@click.option(
    "--tile-size1",
    type=int,
    default=8,
    show_default=True,
    help="Tile grid size for the first CLAHE pass.",
)
@click.option(
    "--tile-size2",
    type=int,
    default=8,
    show_default=True,
    help="Tile grid size for the second CLAHE pass.",
)
@click.option(
    "--blur-kernel",
    type=int,
    default=3,
    show_default=True,
    help="Odd Gaussian kernel size applied before Otsu (set to 1 to disable).",
)
@click.option(
    "--stack/--no-stack",
    default=True,
    show_default=True,
    help="Write side-by-side composites (reference left, DeepDRR right).",
)
@click.option(
    "--guide-left",
    type=str,
    default=None,
    help="Optional normalized guide line for the left image: 'x1,y1,x2,y2' in [0,1] coords.",
)
@click.option(
    "--guide-right",
    type=str,
    default=None,
    help="Optional normalized guide line for the right image: 'x1,y1,x2,y2' in [0,1] coords.",
)
@click.option(
    "--guide-left-from-mask",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional mask to derive the left guide from (multi-class labels).",
)
@click.option(
    "--guide-right-from-mask",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional mask to derive the right guide from (multi-class labels).",
)
@click.option(
    "--guide-labels",
    type=str,
    default="1-7",
    show_default=True,
    help="Label IDs to use when deriving guides from masks (comma or dash separated).",
)
def preprocess_compare(
    frames: tuple[Path, ...],
    pseudos: tuple[Path, ...],
    deepdrrs: tuple[Path, ...],
    out_dir: Path,
    clip_limit1: float,
    clip_limit2: float,
    tile_size1: int,
    tile_size2: int,
    blur_kernel: int,
    stack: bool,
    guide_left: str | None,
    guide_right: str | None,
    guide_left_from_mask: Path | None,
    guide_right_from_mask: Path | None,
    guide_labels: str,
) -> None:
    """
    Apply double-CLAHE + Otsu to frames and pseudo-lateral projections for side-by-side review.
    """

    if tile_size1 <= 0 or tile_size2 <= 0:
        raise click.ClickException("Tile sizes must be positive.")
    if clip_limit1 <= 0 or clip_limit2 <= 0:
        raise click.ClickException("Clip limits must be positive.")
    if blur_kernel <= 0:
        blur_kernel = 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1  # enforce odd kernel for GaussianBlur

    out_root = out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records_frames = _process_group(
        paths=frames,
        label="frames",
        out_root=out_root,
        clip1=clip_limit1,
        tile1=tile_size1,
        clip2=clip_limit2,
        tile2=tile_size2,
        blur_kernel=blur_kernel,
    )
    records_pseudo = _process_group(
        paths=pseudos,
        label="pseudo",
        out_root=out_root,
        clip1=clip_limit1,
        tile1=tile_size1,
        clip2=clip_limit2,
        tile2=tile_size2,
        blur_kernel=blur_kernel,
    )
    records_deepdrr = _process_group(
        paths=deepdrrs,
        label="deepdrr",
        out_root=out_root,
        clip1=clip_limit1,
        tile1=tile_size1,
        clip2=clip_limit2,
        tile2=tile_size2,
        blur_kernel=blur_kernel,
    )
    records: list[dict] = []
    records.extend(records_frames)
    records.extend(records_pseudo)
    records.extend(records_deepdrr)

    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "stem",
                "input",
                "orig_gray",
                "clahe1",
                "clahe2",
                "otsu_mask",
                "otsu_overlay",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    labels_tuple = _parse_label_ids(guide_labels) if guide_labels else (1, 2, 3, 4, 5, 6, 7)
    guide_left_parsed = None
    guide_right_parsed = None
    if guide_left_from_mask:
        guide_left_parsed = _guide_from_mask(guide_left_from_mask, labels_tuple)
    elif guide_left:
        guide_left_parsed = _parse_guide(guide_left)
    label_proj = None
    inferred_label = None
    if label_ct is None:
        if "/volumes/" in str(ct_path):
            inferred_label = Path(str(ct_path).replace("/volumes/", "/labels/"))
        elif "volumes" in ct_path.parts:
            parts = list(ct_path.parts)
            try:
                idx = parts.index("volumes")
                parts[idx] = "labels"
                inferred_label = Path(*parts)
            except ValueError:
                inferred_label = None
    else:
        inferred_label = label_ct

    if guide_right_from_mask:
        guide_right_parsed = _guide_from_mask(guide_right_from_mask, labels_tuple)
    elif inferred_label and Path(inferred_label).exists():
        try:
            label_proj = _project_label_volume(
                label_volume_path=Path(inferred_label),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                target_shape=drr_img.shape,
                native_resolution=native_resolution,
                sensor_height=sensor_height,
                sensor_width=sensor_width,
            )
            guide_right_parsed = _guide_from_labelmap(label_proj, labels_tuple)
        except Exception:
            guide_right_parsed = None
    elif guide_right:
        guide_right_parsed = _parse_guide(guide_right)

    if stack and records_frames and records_deepdrr:
        combos_dir = out_root / "combined"
        combos_dir.mkdir(parents=True, exist_ok=True)
        for idx, (ref_rec, drr_rec) in enumerate(zip(records_frames, records_deepdrr), start=1):
            ref_gray = cv2.imread(ref_rec["orig_gray"], cv2.IMREAD_GRAYSCALE)
            drr_gray = cv2.imread(drr_rec["orig_gray"], cv2.IMREAD_GRAYSCALE)
            ref_overlay = cv2.imread(ref_rec["otsu_overlay"])
            drr_overlay = cv2.imread(drr_rec["otsu_overlay"])
            clahe_ref = cv2.imread(ref_rec["clahe2"], cv2.IMREAD_GRAYSCALE)
            clahe_drr = cv2.imread(drr_rec["clahe2"], cv2.IMREAD_GRAYSCALE)

            if ref_gray is None or drr_gray is None or ref_overlay is None or drr_overlay is None:
                continue

            ref_bgr = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
            drr_bgr = cv2.cvtColor(drr_gray, cv2.COLOR_GRAY2BGR)
            if guide_left_parsed:
                ref_bgr = _draw_guide(ref_bgr, guide_left_parsed)
            if guide_right_parsed:
                drr_bgr = _draw_guide(drr_bgr, guide_right_parsed)

            orig_pair = _stack_horizontal(ref_bgr, drr_bgr)
            cv2.imwrite(str(combos_dir / f"orig_pair{idx}.png"), orig_pair)

            ref_overlay_draw = ref_overlay
            drr_overlay_draw = drr_overlay
            if guide_left_parsed:
                ref_overlay_draw = _draw_guide(ref_overlay_draw, guide_left_parsed)
            if guide_right_parsed:
                drr_overlay_draw = _draw_guide(drr_overlay_draw, guide_right_parsed)

            overlay_pair = _stack_horizontal(ref_overlay_draw, drr_overlay_draw)
            cv2.imwrite(str(combos_dir / f"overlay_pair{idx}.png"), overlay_pair)

            if clahe_ref is not None and clahe_drr is not None:
                clahe_ref_bgr = cv2.cvtColor(clahe_ref, cv2.COLOR_GRAY2BGR)
                clahe_drr_bgr = cv2.cvtColor(clahe_drr, cv2.COLOR_GRAY2BGR)
                if guide_left_parsed:
                    clahe_ref_bgr = _draw_guide(clahe_ref_bgr, guide_left_parsed)
                if guide_right_parsed:
                    clahe_drr_bgr = _draw_guide(clahe_drr_bgr, guide_right_parsed)
                clahe_pair = _stack_horizontal(clahe_ref_bgr, clahe_drr_bgr)
                cv2.imwrite(str(combos_dir / f"clahe2_pair{idx}.png"), clahe_pair)

    click.echo(f"Processed {len(records)} images → {out_root}")


@cli.command("stack-pair")
@click.option(
    "--left",
    type=click.Path(path_type=Path),
    required=True,
    help="Reference image on the left (e.g., fluoroscopy frame).",
)
@click.option(
    "--right",
    type=click.Path(path_type=Path),
    required=True,
    help="Comparison image on the right (e.g., DeepDRR projection).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/pair_compare"),
    show_default=True,
    help="Directory to store stacked panels.",
)
@click.option("--clip-limit1", type=float, default=2.0, show_default=True)
@click.option("--clip-limit2", type=float, default=2.0, show_default=True)
@click.option("--tile-size1", type=int, default=8, show_default=True)
@click.option("--tile-size2", type=int, default=8, show_default=True)
@click.option(
    "--blur-kernel",
    type=int,
    default=3,
    show_default=True,
    help="Odd Gaussian kernel size applied before Otsu (set to 1 to disable).",
)
@click.option(
    "--guide-left",
    type=str,
    default=None,
    help="Optional normalized guide line for the left image: 'x1,y1,x2,y2' in [0,1] coords.",
)
@click.option(
    "--guide-right",
    type=str,
    default=None,
    help="Optional normalized guide line for the right image: 'x1,y1,x2,y2' in [0,1] coords.",
)
@click.option(
    "--guide-left-from-mask",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional mask to derive the left guide from (multi-class labels).",
)
@click.option(
    "--guide-right-from-mask",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional mask to derive the right guide from (multi-class labels).",
)
@click.option(
    "--guide-labels",
    type=str,
    default="1-7",
    show_default=True,
    help="Label IDs to use when deriving guides from masks (comma or dash separated).",
)
def stack_pair(
    left: Path,
    right: Path,
    out_dir: Path,
    clip_limit1: float,
    clip_limit2: float,
    tile_size1: int,
    tile_size2: int,
    blur_kernel: int,
    guide_left: str | None,
    guide_right: str | None,
    guide_left_from_mask: Path | None,
    guide_right_from_mask: Path | None,
    guide_labels: str,
) -> None:
    """
    Generate side-by-side panels (orig, CLAHE, overlay) for a single pair:
    left = reference frame, right = DeepDRR (or other) image.
    """

    if tile_size1 <= 0 or tile_size2 <= 0:
        raise click.ClickException("Tile sizes must be positive.")
    if clip_limit1 <= 0 or clip_limit2 <= 0:
        raise click.ClickException("Clip limits must be positive.")
    if blur_kernel <= 0:
        blur_kernel = 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    out_root = out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for label, path in (("left", left), ("right", right)):
        if not path.expanduser().resolve().exists():
            raise click.ClickException(f"{label} not found: {path}")

    left_img = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        raise click.ClickException("Failed to read one of the images.")

    left_clahe1, left_clahe2 = _double_clahe(left_img, clip_limit1, tile_size1, clip_limit2, tile_size2)
    right_clahe1, right_clahe2 = _double_clahe(right_img, clip_limit1, tile_size1, clip_limit2, tile_size2)

    left_mask = _otsu_mask(left_clahe2, blur_kernel)
    right_mask = _otsu_mask(right_clahe2, blur_kernel)
    left_overlay = _overlay_mask(left_clahe2, left_mask)
    right_overlay = _overlay_mask(right_clahe2, right_mask)

    labels_tuple = _parse_label_ids(guide_labels) if guide_labels else (1, 2, 3, 4, 5, 6, 7)
    guide_left_parsed = None
    guide_right_parsed = None
    if guide_left_from_mask:
        guide_left_parsed = _guide_from_mask(guide_left_from_mask, labels_tuple)
    elif guide_left:
        guide_left_parsed = _parse_guide(guide_left)
    if guide_right_from_mask:
        guide_right_parsed = _guide_from_mask(guide_right_from_mask, labels_tuple)
    elif guide_right:
        guide_right_parsed = _parse_guide(guide_right)

    panels = {
        "orig_pair.png": _stack_horizontal(
            _draw_guide(cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR), guide_left_parsed) if guide_left_parsed else cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR),
            _draw_guide(cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR), guide_right_parsed) if guide_right_parsed else cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR),
        ),
        "clahe2_pair.png": _stack_horizontal(
            _draw_guide(cv2.cvtColor(left_clahe2, cv2.COLOR_GRAY2BGR), guide_left_parsed) if guide_left_parsed else cv2.cvtColor(left_clahe2, cv2.COLOR_GRAY2BGR),
            _draw_guide(cv2.cvtColor(right_clahe2, cv2.COLOR_GRAY2BGR), guide_right_parsed) if guide_right_parsed else cv2.cvtColor(right_clahe2, cv2.COLOR_GRAY2BGR),
        ),
        "overlay_pair.png": _stack_horizontal(
            _draw_guide(left_overlay, guide_left_parsed) if guide_left_parsed else left_overlay,
            _draw_guide(right_overlay, guide_right_parsed) if guide_right_parsed else right_overlay,
        ),
    }

    for name, panel in panels.items():
        cv2.imwrite(str(out_root / name), panel)

    click.echo(f"Wrote stacked panels to {out_root}")


@cli.command("deepdrr-pair")
@click.option("--ct", "ct_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Path to the CT volume (NIfTI) in Hounsfield units.")
@click.option("--frame", "frame_path", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Reference frame to place on the left side.")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/pair_compare_gen"),
              show_default=True, help="Directory to store the generated DRR and stacked panels.")
@click.option("--yaw", type=float, default=0.0, show_default=True)
@click.option("--pitch", type=float, default=0.0, show_default=True)
@click.option("--roll", type=float, default=0.0, show_default=True)
@click.option("--size", "sensor_height", type=int, default=512, show_default=True,
              help="Detector height in pixels (legacy square output size).")
@click.option("--sensor-width", type=int, default=None,
              help="Detector width in pixels (defaults to --size when omitted).")
@click.option("--pixel-mm", type=float, default=1.2, show_default=True,
              help="Detector pixel size (mm/pixel).")
@click.option("--sdd", "source_to_detector_distance", type=float, default=1600.0, show_default=True,
              help="Source-to-detector distance (mm).")
@click.option("--no-noise", is_flag=True, help="Disable DeepDRR noise injection.")
@click.option("--spectrum", default="90KV_AL40", show_default=True,
              help="Named X-ray spectrum to use (DeepDRR preset).")
@click.option("--tone", type=click.Choice(["smooth", "raw"], case_sensitive=False), default="smooth",
              show_default=True, help="Tone-mapping style (default: smooth gamma).")
@click.option("--clahe/--no-clahe", default=False, show_default=True,
              help="Apply CLAHE after tone mapping.")
@click.option("--native-resolution/--no-native-resolution", default=True, show_default=True,
              help="Keep the CT-derived projection grid (disable to force a square detector).")
@click.option(
    "--clip-limit1",
    type=float,
    default=2.0,
    show_default=True,
    help="Clip limit for the first CLAHE pass (for the stacked panels).",
)
@click.option(
    "--clip-limit2",
    type=float,
    default=2.0,
    show_default=True,
    help="Clip limit for the second CLAHE pass (for the stacked panels).",
)
@click.option(
    "--tile-size1",
    type=int,
    default=8,
    show_default=True,
    help="Tile grid size for the first CLAHE pass (for the stacked panels).",
)
@click.option(
    "--tile-size2",
    type=int,
    default=8,
    show_default=True,
    help="Tile grid size for the second CLAHE pass (for the stacked panels).",
)
@click.option(
    "--hist-match/--no-hist-match",
    default=False,
    show_default=True,
    help="Match the DRR histogram to the reference frame before post-processing.",
)
@click.option(
    "--bone-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Multiplier on bone attenuation (DeepDRR); >1 brightens bone.",
)
@click.option(
    "--slice-offset-mm",
    type=float,
    default=0.0,
    show_default=True,
    help="Offset (mm) of the thin slab center along the projection axis (use with --slice-thickness-mm).",
)
@click.option(
    "--slice-thickness-mm",
    type=float,
    default=0.0,
    show_default=True,
    help="Slab thickness (mm) summed along the projection axis; <=0 keeps the full volume.",
)
@click.option(
    "--edge-enhance/--no-edge-enhance",
    default=False,
    show_default=True,
    help="Apply an unsharp mask to emphasize edges/bone on the DeepDRR side.",
)
@click.option(
    "--edge-sigma",
    type=float,
    default=1.2,
    show_default=True,
    help="Gaussian sigma for the unsharp mask blur (controls edge scale).",
)
@click.option(
    "--edge-amount",
    type=float,
    default=1.0,
    show_default=True,
    help="Strength of the unsharp mask boost.",
)
@click.option(
    "--blur-kernel",
    type=int,
    default=3,
    show_default=True,
    help="Odd Gaussian kernel size applied before Otsu (set to 1 to disable).",
)
@click.option(
    "--guide-left",
    type=str,
    default=None,
    help="Optional normalized guide line for the left image: 'x1,y1,x2,y2' in [0,1] coords.",
)
@click.option(
    "--guide-right",
    type=str,
    default=None,
    help="Optional normalized guide line for the right image: 'x1,y1,x2,y2' in [0,1] coords.",
)
@click.option(
    "--guide-left-from-mask",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional mask to derive the left guide from (multi-class labels).",
)
@click.option(
    "--guide-right-from-mask",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional mask to derive the right guide from (multi-class labels).",
)
@click.option(
    "--guide-labels",
    type=str,
    default="1-7",
    show_default=True,
    help="Label IDs to use when deriving guides from masks (comma or dash separated).",
)
@click.option(
    "--guides/--no-guides",
    default=True,
    show_default=True,
    help="Enable/disable drawing guide lines on the stacked panels.",
)
@click.option(
    "--label-ct",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional CT label volume (NIfTI) to derive the right guide by projecting labels with the same yaw/pitch/roll.",
)
@click.option("--aperture/--no-aperture", default=False, show_default=True,
              help="Apply a circular c-arm vignette to the DeepDRR image.")
@click.option("--aperture-radius-scale", type=float, default=0.48, show_default=True,
              help="Radius as a fraction of min(H,W) for the vignette.")
@click.option("--aperture-softness", type=float, default=0.22, show_default=True,
              help="Edge softness for the vignette falloff.")
@click.option("--aperture-inside", type=float, default=1.0, show_default=True,
              help="Gain multiplier inside the vignette (default 1.0).")
@click.option("--aperture-outside", type=float, default=0.0, show_default=True,
              help="Gain multiplier outside the vignette (default 0 = black).")
@click.option("--aperture-blur", type=int, default=3, show_default=True,
              help="Gaussian blur (px) applied to the vignette mask edges (set 0 for none).")
@click.option("--aperture-ellipse-y-scale", type=float, default=1.0, show_default=True,
              help="Vertical scale for the aperture (use <1 to flatten the circle).")
@click.option("--aperture-mask-labels/--no-aperture-mask-labels", default=True, show_default=True,
              help="Apply the aperture mask to the projected labels as well (keep on for alignment).")
@click.option("--zoom-factor", type=float, default=1.0, show_default=True,
              help="Digital zoom: >1 crops tighter around center then resizes back; 1.0 disables.")
@click.option("--pan-x-px", type=float, default=0.0, show_default=True,
              help="Horizontal pan in pixels (positive = right) applied during zoom crop.")
@click.option("--pan-y-px", type=float, default=0.0, show_default=True,
              help="Vertical pan in pixels (positive = down) applied during zoom crop.")
@click.option("--crop-square/--no-crop-square", default=True, show_default=True,
              help="If enabled, center-crop the DeepDRR output to a square (1:1 aspect).")
@click.option("--match-frame-size/--no-match-frame-size", default=True, show_default=True,
              help="Resize the DeepDRR (after crop/aperture) to match the reference frame dimensions.")
@click.option("--colorize/--no-colorize", default=True, show_default=True,
              help="Draw colorized labels on the DeepDRR (requires --label-ct or --guide-right-from-mask).")
@click.option(
    "--save-bbox/--no-save-bbox",
    default=False,
    show_default=True,
    help="Save the tight bbox of guide labels to JSON in the output directory.",
)
@click.option(
    "--crop-to-bbox/--no-crop-to-bbox",
    default=False,
    show_default=True,
    help="If enabled and labels exist, crop outputs to the bbox of guide labels.",
)
@click.option(
    "--crop-margin",
    type=float,
    default=0.05,
    show_default=True,
    help="Extra margin (fraction of bbox size) when cropping to bbox.",
)
@click.option(
    "--resize-after-crop/--no-resize-after-crop",
    default=True,
    show_default=True,
    help="If cropping to bbox, resize the crop back to the original size.",
)
@click.option(
    "--letterbox-after-crop/--no-letterbox-after-crop",
    default=False,
    show_default=True,
    help="If cropping to bbox, letterbox (pad) back to original size instead of stretching.",
)
@click.option(
    "--letterbox-pad-only/--letterbox-resize",
    default=False,
    show_default=True,
    help="When letterboxing after crop, pad without scaling (default resizes with aspect).",
)
@click.option(
    "--letterbox-color",
    type=int,
    default=0,
    show_default=True,
    help="Padding color for letterbox (0–255).",
)
@click.option(
    "--otsu-source",
    type=click.Choice(["clahe2", "orig"], case_sensitive=False),
    default="clahe2",
    show_default=True,
    help="Image used for Otsu mask/overlay: 'clahe2' (default) or 'orig' (pre-CLAHE).",
)
@click.option(
    "--crop-frame/--no-crop-frame",
    default=True,
    show_default=True,
    help="Apply bbox crop to the reference frame as well (disable to keep frame untouched).",
)
@click.option(
    "--save-raw-drr/--no-save-raw-drr",
    default=True,
    show_default=True,
    help="Save the uncropped DeepDRR image before bbox/crop transforms.",
)
def deepdrr_pair(
    ct_path: Path,
    frame_path: Path,
    out_dir: Path,
    yaw: float,
    pitch: float,
    roll: float,
    sensor_height: int,
    sensor_width: int | None,
    pixel_mm: float,
    source_to_detector_distance: float,
    no_noise: bool,
    spectrum: str,
    tone: str,
    clahe: bool,
    native_resolution: bool,
    clip_limit1: float,
    clip_limit2: float,
    tile_size1: int,
    tile_size2: int,
    hist_match: bool,
    bone_scale: float,
    slice_offset_mm: float,
    slice_thickness_mm: float,
    edge_enhance: bool,
    edge_sigma: float,
    edge_amount: float,
    blur_kernel: int,
    guide_left: str | None,
    guide_right: str | None,
    guide_left_from_mask: Path | None,
    guide_right_from_mask: Path | None,
    guide_labels: str,
    guides: bool,
    label_ct: Path | None,
    aperture: bool,
    aperture_radius_scale: float,
    aperture_softness: float,
    aperture_inside: float,
    aperture_outside: float,
    aperture_blur: int,
    aperture_ellipse_y_scale: float,
    aperture_mask_labels: bool,
    zoom_factor: float,
    pan_x_px: float,
    pan_y_px: float,
    crop_square: bool,
    match_frame_size: bool,
    colorize: bool,
    save_bbox: bool,
    crop_to_bbox: bool,
    crop_margin: float,
    resize_after_crop: bool,
    letterbox_after_crop: bool,
    letterbox_pad_only: bool,
    letterbox_color: int,
    otsu_source: str,
    crop_frame: bool,
    save_raw_drr: bool,
) -> None:
    """
    Render a DeepDRR projection from a CT, then stack it side-by-side with a reference frame.
    """

    if tile_size1 <= 0 or tile_size2 <= 0:
        raise click.ClickException("Tile sizes must be positive.")
    if clip_limit1 <= 0 or clip_limit2 <= 0:
        raise click.ClickException("Clip limits must be positive.")
    if blur_kernel <= 0:
        blur_kernel = 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    out_root = out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Load frame early so we can size-match later.
    frame_img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame_img is None:
        raise click.ClickException(f"Failed to read frame: {frame_path}")

    projected_labels = None
    projected_labels_raw = None

    # Generate DeepDRR image.
    drr_img = render_deepdrr_projection(
        ct_path=ct_path,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        sensor_size_px=sensor_height,
        sensor_width_px=sensor_width,
        native_resolution=native_resolution,
        pixel_size_mm=pixel_mm,
        source_to_detector_distance_mm=source_to_detector_distance,
        add_noise=not no_noise,
        spectrum=spectrum,
        tone_style=tone,
        apply_clahe=clahe,
        bone_scale=bone_scale,
        slice_offset_mm=slice_offset_mm,
        slice_thickness_mm=slice_thickness_mm,
    )
    drr_raw = drr_img.copy()

    labels_tuple = _parse_label_ids(guide_labels) if guide_labels else (1, 2, 3, 4, 5, 6, 7)
    guide_left_parsed = None
    guide_right_parsed = None

    # Project labels before any spatial transforms so we can apply the same crop/resize later.
    if label_ct:
        label_ct = label_ct.expanduser().resolve()
        if not label_ct.exists():
            raise click.ClickException(f"Label CT not found: {label_ct}")
        import nibabel as nib  # local import to avoid hard dep if not used
        lbl_nii = nib.load(str(label_ct))
        spacing_z = lbl_nii.header.get_zooms()[2] if len(lbl_nii.header.get_zooms()) > 2 else 1.0
        lbl_vol = lbl_nii.get_fdata().astype(np.int16)

        projected_labels = _project_label_volume(
            label_volume_path=None,
            label_volume=lbl_vol,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            target_shape=drr_img.shape,
            native_resolution=native_resolution,
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            slice_offset_mm=slice_offset_mm,
            slice_thickness_mm=slice_thickness_mm,
            spacing_z=spacing_z,
        )
        projected_labels_raw = projected_labels.copy()

    # If we have projected labels, keep them in sync with drr transforms below.
    if crop_square:
        h, w = drr_img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        drr_img = drr_img[y0:y0 + side, x0:x0 + side]
        if projected_labels is not None and projected_labels.shape[:2] == (h, w):
            projected_labels = projected_labels[y0:y0 + side, x0:x0 + side]

    # Optional: digital zoom (center crop + pan, then resize back to same size)
    if zoom_factor and zoom_factor > 1.0:
        h, w = drr_img.shape[:2]
        side = int(round(min(h, w) / zoom_factor))
        side = max(1, side)

        # Allow manual panning of the zoomed crop; clamp so we stay in-bounds.
        half = side / 2.0
        cx = np.clip(w / 2.0 + pan_x_px, half, w - half)
        cy = np.clip(h / 2.0 + pan_y_px, half, h - half)
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)

        drr_img = drr_img[y0:y1, x0:x1]
        drr_img = cv2.resize(drr_img, (w, h), interpolation=cv2.INTER_CUBIC)
        if projected_labels is not None:
            lbl_crop = projected_labels[y0:y1, x0:x1]
            projected_labels = cv2.resize(
                lbl_crop.astype(np.int32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(projected_labels.dtype)
    if save_raw_drr:
        (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "deepdrr" / "deepdrr_postzoom.png"), drr_img)

    # Optional: force same dimensions as the reference frame for 1:1 visual comparison
    if match_frame_size and frame_img.shape[:2] != drr_img.shape[:2]:
        drr_img = cv2.resize(drr_img, (frame_img.shape[1], frame_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if projected_labels is not None:
            projected_labels = cv2.resize(
                projected_labels.astype(np.int32),
                (frame_img.shape[1], frame_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(projected_labels.dtype)

    # Optional: match DRR histogram to the frame to better align overall tone.
    if hist_match:
        drr_img = _hist_match(drr_img, frame_img)

    # Optional: edge/bone enhancement via unsharp mask.
    if edge_enhance:
        drr_img = _unsharp_enhance(drr_img, sigma=edge_sigma, amount=edge_amount)

    # Save overlay after crop/resize but before aperture to debug drift.
    if colorize and projected_labels is not None:
        (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)
        pre_ap_overlay = _overlay_labels_color(drr_img, projected_labels)
        cv2.imwrite(str(out_root / "deepdrr" / "label_overlay_pre_aperture.png"), pre_ap_overlay)

    # Apply aperture last at final resolution to minimize aliasing.
    if aperture:
        h, w = drr_img.shape[:2]
        radius = min(h, w) * max(aperture_radius_scale, 1e-3)
        drr_img = _apply_c_arm_aperture(
            drr_img,
            center_xy=(w / 2.0, h / 2.0),
            radius=radius,
            edge_softness=aperture_softness,
            inside_gain=aperture_inside,
            outside_gain=aperture_outside,
            blur_px=aperture_blur,
            ellipse_y_scale=aperture_ellipse_y_scale,
        )
        if projected_labels is not None and aperture_mask_labels:
            yy, xx = np.mgrid[0:h, 0:w]
            ry = max(aperture_ellipse_y_scale, 1e-3)
            rr = np.sqrt((xx - w / 2.0) ** 2 + ((yy - h / 2.0) / ry) ** 2)
            mask = (rr <= radius).astype(projected_labels.dtype)
            projected_labels = projected_labels * mask

    # Keep a copy before any bbox crop/resize so we can build circular exports at the original aperture view.
    drr_precrop = drr_img.copy()
    projected_labels_precrop = projected_labels.copy() if projected_labels is not None else None

    # Record the post-zoom/aperture size for later padding.
    base_h, base_w = drr_img.shape[:2]

    # Save the full DeepDRR after tone/aperture but before bbox crop.
    if save_raw_drr:
        (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "deepdrr" / "deepdrr_precrop.png"), drr_img)

    # Optionally crop to bbox (with margin) across frame/DRR/labels.
    if crop_to_bbox and projected_labels is not None:
        bbox = _label_bbox(projected_labels, labels_tuple)
        if bbox is None:
            click.echo("Warning: crop-to-bbox requested but no labels found; skipping crop.")
        else:
            orig_h, orig_w = drr_img.shape[:2]
            x0, y0, x1, y1 = bbox
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1

            # Skip pathological tiny crops (e.g., thin slabs with almost no mask)
            MIN_CROP_SIZE = 64
            if bw < MIN_CROP_SIZE or bh < MIN_CROP_SIZE:
                click.echo(
                    f"Skipping bbox crop (too small: {bw}x{bh}); keeping full frame for stability."
                )
            else:
                expand_x = int(round(bw * crop_margin))
                expand_y = int(round(bh * crop_margin))
                x0 = max(0, x0 - expand_x)
                y0 = max(0, y0 - expand_y)
                x1 = min(orig_w - 1, x1 + expand_x)
                y1 = min(orig_h - 1, y1 + expand_y)

                def _crop(arr: np.ndarray) -> np.ndarray:
                    return arr[y0:y1 + 1, x0:x1 + 1]

                drr_img = _crop(drr_img)
                if crop_frame:
                    frame_img = _crop(frame_img)
                projected_labels = _crop(projected_labels)

                # We will enforce final sizing after this block.

    # Enforce final size back to original (base_h, base_w) via letterbox or resize.
    if (drr_img.shape[0], drr_img.shape[1]) != (base_h, base_w):
        if letterbox_after_crop:
            letter_fn = _pad_to_canvas if letterbox_pad_only else _letterbox
            drr_img, _ = letter_fn(drr_img, base_w, base_h, color=letterbox_color, interpolation=cv2.INTER_LINEAR)
            if crop_frame:
                frame_img, _ = letter_fn(
                    frame_img, base_w, base_h, color=letterbox_color, interpolation=cv2.INTER_LINEAR
                )
            if projected_labels is not None:
                projected_labels, _ = letter_fn(
                    projected_labels.astype(np.int32),
                    base_w,
                    base_h,
                    color=0,
                    interpolation=cv2.INTER_NEAREST,
                )
                projected_labels = projected_labels.astype(np.int32)
        elif resize_after_crop:
            drr_img = cv2.resize(drr_img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            if crop_frame:
                frame_img = cv2.resize(frame_img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            if projected_labels is not None:
                projected_labels = cv2.resize(
                    projected_labels.astype(np.int32),
                    (base_w, base_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(projected_labels.dtype)

    # Compute bbox on final labels (post crop/resize/letterbox).
    bbox_final = None
    if projected_labels is not None:
        bbox_final = _label_bbox(projected_labels, labels_tuple)

    # Save bitmask of labels (up to 16 labels packed into uint16).
    if projected_labels is not None and labels_tuple:
        bitmask = np.zeros(projected_labels.shape[:2], dtype=np.uint16)
        for idx, lid in enumerate(labels_tuple):
            if idx >= 16:
                break  # PNG uint16 only holds 16 bits; extend if needed later.
            bitmask[projected_labels == lid] |= (1 << idx)
        (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "deepdrr" / "label_bitmask_cropped_letterboxed.png"), bitmask)
    if projected_labels_precrop is not None and labels_tuple:
        bitmask_circ = np.zeros(projected_labels_precrop.shape[:2], dtype=np.uint16)
        for idx, lid in enumerate(labels_tuple):
            if idx >= 16:
                break
            bitmask_circ[projected_labels_precrop == lid] |= (1 << idx)
        (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "deepdrr" / "label_bitmask_circular_synth.png"), bitmask_circ)

    # Save bbox to JSON if requested.
    if save_bbox:
        bbox_path = out_root / "bbox_labels.json"
        payload = {
            "bbox": None if bbox_final is None else {
                "x0": bbox_final[0],
                "y0": bbox_final[1],
                "x1": bbox_final[2],
                "y1": bbox_final[3],
            },
            "image_size": {"width": int(drr_img.shape[1]), "height": int(drr_img.shape[0])},
            "labels": list(labels_tuple),
        }
        with open(bbox_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    drr_path = out_root / "deepdrr.png"
    cv2.imwrite(str(drr_path), drr_img)

    # Save an overlay on the original (pre-crop) DRR for debugging alignment.
    if colorize and projected_labels_raw is not None:
        labels_for_raw = projected_labels_raw
        if labels_for_raw.shape[:2] != drr_raw.shape[:2]:
            labels_for_raw = cv2.resize(
                labels_for_raw.astype(np.int32),
                (drr_raw.shape[1], drr_raw.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(labels_for_raw.dtype)
        orig_overlay = _overlay_labels_color(drr_raw, labels_for_raw)
        (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "deepdrr" / "original_label_overlay.png"), orig_overlay)

    def _process_single(
        img: np.ndarray, tag: str, label_overlay: np.ndarray | None = None, otsu_from: str = "clahe2"
    ) -> dict:
        clahe1, clahe2 = _double_clahe(img, clip_limit1, tile_size1, clip_limit2, tile_size2)
        if otsu_from.lower() == "orig":
            mask_base = img
        else:
            mask_base = clahe2
        mask = _otsu_mask(mask_base, blur_kernel)
        overlay = _overlay_mask(mask_base, mask)
        color_overlay = None
        if label_overlay is not None:
            color_overlay = _overlay_labels_color(mask_base, label_overlay)
        out_dir_tag = out_root / tag
        out_dir_tag.mkdir(parents=True, exist_ok=True)

        paths = {
            "orig": out_dir_tag / "orig_gray.png",
            "clahe1": out_dir_tag / "clahe1.png",
            "clahe2": out_dir_tag / "clahe2.png",
            "otsu_mask": out_dir_tag / "otsu_mask.png",
            "otsu_overlay": out_dir_tag / "otsu_overlay.png",
        }
        cv2.imwrite(str(paths["orig"]), img)
        cv2.imwrite(str(paths["clahe1"]), clahe1)
        cv2.imwrite(str(paths["clahe2"]), clahe2)
        cv2.imwrite(str(paths["otsu_mask"]), mask)
        cv2.imwrite(str(paths["otsu_overlay"]), overlay)
        if color_overlay is not None:
            paths["label_overlay"] = out_dir_tag / "label_overlay.png"
            cv2.imwrite(str(paths["label_overlay"]), color_overlay)
        return paths

    frame_paths = _process_single(frame_img, "frame", otsu_from=otsu_source)
    drr_paths = _process_single(
        drr_img,
        "deepdrr",
        label_overlay=projected_labels if (colorize and projected_labels is not None) else None,
        otsu_from=otsu_source,
    )

    if guides:
        if guide_left_from_mask:
            guide_left_parsed = _guide_from_mask(guide_left_from_mask, labels_tuple)
        elif guide_left:
            guide_left_parsed = _parse_guide(guide_left)
        if guide_right_from_mask:
            guide_right_parsed = _guide_from_mask(guide_right_from_mask, labels_tuple)
        elif projected_labels is not None:
            guide_right_parsed = _guide_from_labelmap(projected_labels, labels_tuple)
        elif guide_right:
            guide_right_parsed = _parse_guide(guide_right)

    combos_dir = out_root / "combined"
    combos_dir.mkdir(parents=True, exist_ok=True)

    left_bgr = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(drr_img, cv2.COLOR_GRAY2BGR)
    if guide_left_parsed:
        left_bgr = _draw_guide(left_bgr, guide_left_parsed)
    if guide_right_parsed:
        right_bgr = _draw_guide(right_bgr, guide_right_parsed)

    orig_pair = _stack_horizontal(left_bgr, right_bgr)
    cv2.imwrite(str(combos_dir / "orig_pair.png"), orig_pair)

    clahe_frame = cv2.imread(str(frame_paths["clahe2"]), cv2.IMREAD_GRAYSCALE)
    clahe_drr = cv2.imread(str(drr_paths["clahe2"]), cv2.IMREAD_GRAYSCALE)
    clahe_frame_bgr = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)
    clahe_drr_bgr = cv2.cvtColor(clahe_drr, cv2.COLOR_GRAY2BGR)
    if guide_left_parsed:
        clahe_frame_bgr = _draw_guide(clahe_frame_bgr, guide_left_parsed)
    if guide_right_parsed:
        clahe_drr_bgr = _draw_guide(clahe_drr_bgr, guide_right_parsed)

    clahe_pair = _stack_horizontal(clahe_frame_bgr, clahe_drr_bgr)
    cv2.imwrite(str(combos_dir / "clahe2_pair.png"), clahe_pair)

    overlay_frame = cv2.imread(str(frame_paths["otsu_overlay"]))
    overlay_drr = cv2.imread(str(drr_paths["otsu_overlay"]))
    if guide_left_parsed:
        overlay_frame = _draw_guide(overlay_frame, guide_left_parsed)
    if guide_right_parsed:
        overlay_drr = _draw_guide(overlay_drr, guide_right_parsed)

    overlay_pair = _stack_horizontal(overlay_frame, overlay_drr)
    cv2.imwrite(str(combos_dir / "overlay_pair.png"), overlay_pair)

    # Side-by-side binary masks (Otsu only) to compare thresholded structure.
    mask_frame = cv2.imread(str(frame_paths["otsu_mask"]), cv2.IMREAD_GRAYSCALE)
    mask_drr = cv2.imread(str(drr_paths["otsu_mask"]), cv2.IMREAD_GRAYSCALE)
    if mask_frame is not None and mask_drr is not None:
        mask_pair = _stack_horizontal(
            cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_drr, cv2.COLOR_GRAY2BGR),
        )
        cv2.imwrite(str(combos_dir / "binary_pair.png"), mask_pair)

    # Circle-masked combined view similar to fluoroscopy apertures.
    (out_root / "deepdrr").mkdir(parents=True, exist_ok=True)

    hh, ww = drr_precrop.shape[:2]
    circ_mask = _make_aperture_mask(
        (hh, ww),
        center_xy=(ww / 2.0, hh / 2.0),
        radius=min(hh, ww) * 0.5,
        edge_softness=0.0,
        blur_px=0,
        ellipse_y_scale=1.0,
    ).astype(np.float32)
    circ_mask_3c = circ_mask[..., None]

    # Use the pre-crop DRR (aperture view) to create the circular export.
    circular_drr = (drr_precrop.astype(np.float32) * circ_mask).astype(np.uint8)
    circular_drr_bgr = cv2.cvtColor(circular_drr, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_root / "deepdrr" / "circular-synth.png"), circular_drr)

    left_for_circle = cv2.resize(left_bgr, (ww, hh), interpolation=cv2.INTER_LINEAR)
    left_circle = (left_for_circle.astype(np.float32) * circ_mask_3c).astype(np.uint8)
    circle_pair = _stack_horizontal(left_circle, circular_drr_bgr)
    cv2.imwrite(str(combos_dir / "circle_pair.png"), circle_pair)

    click.echo(f"DeepDRR + stacked panels written to {out_root}")
def _overlay_labels_color(base: np.ndarray, labels: np.ndarray,
                          alpha: float = 0.45) -> np.ndarray:
    """Blend a colorized label map onto a grayscale/base BGR image."""
    from ..core.label_colors import label_to_color

    if labels.dtype != np.int32:
        labels = labels.astype(np.int32)
    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique = [int(v) for v in np.unique(labels) if v > 0]
    for lid in unique:
        color[labels == lid] = label_to_color(lid)
    if base.ndim == 2:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base
    return cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0.0)
