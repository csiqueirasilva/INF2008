import json
from pathlib import Path
import click
import cv2
import numpy as np

from .root import cli
from ..core.overlay import save_u8_gray
from ..core.drr_utils import mask_to_outline


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _to_binary(mask: np.ndarray, thresh: int = 1) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(mask, max(0, int(thresh)), 255, cv2.THRESH_BINARY)
    return bw


def _resize_to(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if img.shape[:2] == (h, w):
        return img
    interp = cv2.INTER_NEAREST if (img.ndim == 2 or img.shape[2] == 1) else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)


def _overlay_mask(gray: np.ndarray, mask_bw: np.ndarray, color=(0, 255, 0), alpha=0.35) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(base)
    overlay[mask_bw.astype(bool)] = color
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0.0)


def _apply_mask(gray: np.ndarray, mask_bw: np.ndarray) -> np.ndarray:
    m = (mask_bw > 0).astype(np.uint8)
    return (gray * m).astype(np.uint8)


def _metrics(a_bw: np.ndarray, b_bw: np.ndarray) -> dict:
    a = (a_bw > 0).astype(np.uint8)
    b = (b_bw > 0).astype(np.uint8)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    sa = int(a.sum()); sb = int(b.sum())
    iou = (inter / union) if union > 0 else 0.0
    dice = (2 * inter / (sa + sb)) if (sa + sb) > 0 else 0.0
    return {"area_a": sa, "area_b": sb, "intersection": inter, "union": union, "iou": iou, "dice": dice}


@cli.command("compare-masks")
@click.option("--frame", type=click.Path(path_type=Path), required=True, help="Path to frame image (png/jpg)")
@click.option("--manual", type=click.Path(path_type=Path), required=True, help="Path to ImageJ Mask.tif (or similar)")
@click.option("--other", type=click.Path(path_type=Path), default=None, help="Optional: path to another mask to compare")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("outputs/compare_manual"), show_default=True)
@click.option("--manual-thresh", type=int, default=1, show_default=True, help="Threshold to binarize manual mask")
@click.option("--other-thresh", type=int, default=1, show_default=True, help="Threshold to binarize other mask")
def compare_masks(frame, manual, other, out_dir, manual_thresh, other_thresh):
    """
    Compare a frame against an ImageJ mask (and optionally another mask).
    Produces overlays, masked frames, and IoU/Dice metrics if two masks are given.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    gray = _read_gray(frame)
    m1 = _read_gray(manual)
    m1 = _resize_to(m1, gray.shape[:2])
    m1 = _to_binary(m1, thresh=manual_thresh)

    # Save base and manual overlays
    save_u8_gray(out_dir / "00_frame.png", gray)
    ov1 = _overlay_mask(gray, m1, color=(0, 255, 0), alpha=0.35)
    cv2.imwrite(str(out_dir / "01_manual_overlay.png"), ov1)
    cv2.imwrite(str(out_dir / "02_manual_masked.png"), _apply_mask(gray, m1))

    report = {"frame": str(frame), "manual": str(manual)}

    if other:
        m2 = _read_gray(other)
        m2 = _resize_to(m2, gray.shape[:2])
        m2 = _to_binary(m2, thresh=other_thresh)

        ov2 = _overlay_mask(gray, m2, color=(0, 0, 255), alpha=0.35)
        cv2.imwrite(str(out_dir / "03_other_overlay.png"), ov2)
        cv2.imwrite(str(out_dir / "04_other_masked.png"), _apply_mask(gray, m2))

        # Outline comparison visualization
        o1 = mask_to_outline(m1, k=3)
        o2 = mask_to_outline(m2, k=3)
        comp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        comp[o1.astype(bool)] = (0, 255, 0)
        comp[o2.astype(bool)] = (0, 0, 255)
        cv2.imwrite(str(out_dir / "10_compare_outlines.png"), comp)

        # Combined overlay (alpha-blended)
        both = cv2.addWeighted(ov1, 0.5, ov2, 0.5, 0.0)
        cv2.imwrite(str(out_dir / "11_compare_overlay_both.png"), both)

        mets = _metrics(m1, m2)
        report.update({"other": str(other), "metrics": mets})

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    click.echo(f"📂 Outputs in {out_dir}")

